#!/usr/bin/env python3
"""
ingest_bulk.py
--------------
Fetch play-by-play, shots and possessions for multiple NBA games.

Usage
-----
python src/ingest_bulk.py <game_id1> <game_id2> ...
"""
import sys
from ingest import fetch_game


def main(game_ids):
    if not game_ids:
        print("Usage: python src/ingest_bulk.py <game_id1> <game_id2> ...")
        return 1
    for gid in game_ids:
        try:
            fetch_game(gid)
        except Exception as e:
            print(f"\u26A0\ufe0f  Failed to ingest {gid}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
