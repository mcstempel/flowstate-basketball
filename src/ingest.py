#!/usr/bin/env python3
"""
ingest.py
----------

Pulls play‑by‑play, shot chart, and possession data for a single NBA game
from stats.nba.com (via the `pbpstats` library) and saves everything to:

    data/raw_<game_id>.json

Usage
-----
$ python src/ingest.py 0022400001
# (defaults to 0022400001 if you omit the argument)

Dependencies
------------
pip install pbpstats
"""

import json
import os
import sys
from pbpstats.client import Client


def fetch_game(game_id: str, out_dir: str = "data") -> str:
    """Download one game and return the path of the saved JSON file."""
    settings = {
        # Each resource: where to load it & which provider to use
        "Pbp":         {"source": "web", "data_provider": "stats_nba"},
        "Shots":       {"source": "web", "data_provider": "stats_nba"},
        "Possessions": {"source": "web", "data_provider": "stats_nba"},
    }

    client = Client(settings)
    game = client.Game(game_id)

raw = {
    "pbp":         [e.to_dict() for e in game.pbp.items],          # <- change here
    "shots":       [s.to_dict() for s in game.shots.items],        # <- and here
    "possessions": [p.to_dict() for p in game.possessions.items],
}

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"raw_{game_id}.json")
    with open(out_path, "w") as f:
        json.dump(raw, f)

    print(
        f"✅  Saved {out_path}  "
        f"({len(raw['pbp'])} pbp events, {len(raw['shots'])} shots, "
        f"{len(raw['possessions'])} possessions)"
    )
    return out_path


if __name__ == "__main__":
    gid = sys.argv[1] if len(sys.argv) > 1 else "0022400001"
    fetch_game(gid)
