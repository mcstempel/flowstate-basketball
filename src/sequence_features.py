#!/usr/bin/env python3
"""
sequence_features.py
--------------------
Adds short‑memory (m = 3 possessions) features on top of the baseline CSV.

Usage
-----
python src/sequence_features.py 0022400001
# default game id = 0022400001 if omitted

Output
------
data/sequence_<game_id>.csv
"""
import sys, os
import pandas as pd


def add_sequence_feats(game_id: str):
    base_path = f"data/baseline_{game_id}.csv"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"{base_path} not found. Run features.py first.")

    df = pd.read_csv(base_path).sort_values("poss_id").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 1. previous points (numeric)                                       #
    # ------------------------------------------------------------------ #
    for k in (1, 2, 3):
        df[f"prev_pts_{k}"] = df["points_scored"].shift(k).fillna(0).astype(int)

    # ------------------------------------------------------------------ #
    # 2. previous shot bucket (categorical → one label)                  #
    # ------------------------------------------------------------------ #
    df["prev_bucket_1"] = df["shot_bucket"].shift(1).fillna("none")

    # ------------------------------------------------------------------ #
    # 3. tempo metrics                                                   #
    # ------------------------------------------------------------------ #
    df["tempo_sec"] = df["clock_start_sec"] - df["clock_end_sec"]
    df["tempo_mean_last3"] = df["tempo_sec"].rolling(window=3, min_periods=1).mean().shift(1)

    # ------------------------------------------------------------------ #
    # 4. simple momentum flag                                            #
    # ------------------------------------------------------------------ #
    df["streak_scored_last3"] = (
        (df["prev_pts_1"] > 0)
        & (df["prev_pts_2"] > 0)
        & (df["prev_pts_3"] > 0)
    ).astype(int)

    # drop any rows that lost context (first 3) if you prefer
    # df = df[df["poss_id"] > 3]

    out_path = f"data/sequence_{game_id}.csv"
    df.to_csv(out_path, index=False)
    print(
        f"✅  Saved {out_path}  "
        f"({len(df)} rows, {df.shape[1]} cols)"
    )
    return out_path


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "0022400001"
    add_sequence_feats(gid)
