#!/usr/bin/env python3
"""
features.py
-----------
Build a baseline feature set from raw_<game_id>.json produced by ingest.py.

Usage
-----
python src/features.py 0022400001
# default game id = 0022400001 if omitted

Output
------
data/baseline_<game_id>.csv
"""
import json, os, sys
import pandas as pd

# ---------------- helpers ----------------------------------------------------


def clock_to_seconds(clock: str) -> int:
    """
    Convert a clock string like 'PT11M32.00S' (ISO8601) to total seconds.
    pbpstats uses this ISO format for possession time remaining.
    """
    if clock.startswith("PT") and "M" in clock:
        minutes = int(clock.split("PT")[1].split("M")[0])
        seconds = float(clock.split("M")[1].replace("S", ""))
        return int(minutes * 60 + seconds)
    return 0


def shot_bucket(distance_ft: float) -> str:
    """
    Very coarse location bucket. Good enough for the baseline model.
    """
    if distance_ft is None:
        return "no_shot"
    if distance_ft <= 3:
        return "restricted_area"
    if distance_ft <= 14:
        return "paint"
    if distance_ft <= 18:
        return "midrange"
    # Distances up through the corner three range
    if distance_ft <= 24:
        return "corner_three"
    # Anything longer is a non‑corner three
    return "non_corner_three"


# ---------------- main -------------------------------------------------------


def build_baseline(game_id: str) -> str:
    raw_path = f"data/raw_{game_id}.json"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found. Run ingest.py first.")

    with open(raw_path) as f:
        raw = json.load(f)

    # possessions is a list of dicts (we created it in ingest.py)
    poss_df = pd.DataFrame(raw["possessions"])

    # Basic derived columns
    poss_df["period"] = poss_df["period"]
    poss_df["clock_start_sec"] = poss_df["time_remaining_in_period"].apply(
        clock_to_seconds
    )
    poss_df["clock_end_sec"] = poss_df["clock_start_sec"] - poss_df["duration"]

    # Score differential at possession start (offense minus defense)
    poss_df["score_diff_start"] = (
        poss_df["offense_start_score"] - poss_df["defense_start_score"]
    )

    # Label we want to predict later
    poss_df["points_scored"] = poss_df["points"]

    # ---------------- shot location bucket ----------------------------------
    # link each possession to its last shot distance if a shot occurred
    shot_df = pd.DataFrame(raw["shots"])[
        ["event_num", "distance", "period"]
    ].rename(columns={"distance": "shot_distance_ft"})

    # use event_num max per possession_id to get the *last* shot
    last_shots = (
        poss_df[["poss_id", "last_event_num"]]
        .merge(shot_df, left_on="last_event_num", right_on="event_num", how="left")
        .set_index("poss_id")
    )

    poss_df["shot_distance_ft"] = last_shots["shot_distance_ft"]
    poss_df["shot_bucket"] = poss_df["shot_distance_ft"].apply(
        lambda d: shot_bucket(d) if pd.notna(d) else "no_shot"
    )

    # ---------------- select baseline columns --------------------------------
    keep_cols = [
        "poss_id",
        "period",
        "clock_start_sec",
        "clock_end_sec",
        "offense_team_id",
        "defense_team_id",
        "score_diff_start",
        "shot_bucket",
        "points_scored",  # ← label
    ]
    baseline = poss_df[keep_cols].copy()

    out_csv = f"data/baseline_{game_id}.csv"
    baseline.to_csv(out_csv, index=False)
    print(f"✅  Saved {out_csv}  ({len(baseline)} rows, {baseline.shape[1]} cols)")
    return out_csv


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "0022400001"
    build_baseline(gid)
