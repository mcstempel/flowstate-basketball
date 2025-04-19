#!/usr/bin/env python3
"""
ingest.py
----------
Download play‑by‑play, shot chart, and possessions for a single NBA game
using pbpstats and save them as one JSON blob:

    data/raw_<game_id>.json

Usage
-----
python src/ingest.py 0022300031
# (game_id defaults to 0022400001 if omitted)
"""
import json
import os
import sys
from pbpstats.client import Client


def _obj_to_dict(obj):
    """Return the dict representation regardless of pbpstats version."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise AttributeError("Pbpstats item has no .to_dict() or .as_dict() method")


def fetch_game(game_id: str, out_dir: str = "data") -> str:
    """Download one game and write data/raw_<game_id>.json. Return path."""
    settings = {
        "Pbp":         {"source": "web", "data_provider": "stats_nba"},
        "Shots":       {"source": "web", "data_provider": "stats_nba"},
        "Possessions": {"source": "web", "data_provider": "stats_nba"},
    }

    client = Client(settings)
    game = client.Game(game_id)

    raw = {
        "pbp":         [_obj_to_dict(e) for e in game.pbp.items],
        "shots":       [_obj_to_dict(s) for s in game.shots.items],
        "possessions": [_obj_to_dict(p) for p in game.possessions.items],
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"raw_{game_id}.json")
    with open(out_path, "w") as f:
        json.dump(raw, f)

    print(
        f"✅  Saved {out_path}  "
        f"({len(raw['possessions'])} possessions, "
        f"{len(raw['pbp'])} pbp events, {len(raw['shots'])} shots)"
    )
    return out_path


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "0022400001"
    fetch_game(gid)

