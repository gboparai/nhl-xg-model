"""
build_shift_lookup.py
=====================
Reads all raw shift JSON files and player position files, then builds a
parquet lookup table with an is_goalie flag per player.

Schema:
  game_id    int64
  period     int64
  player_id  int64
  team_id    int64
  start_s    int32   (seconds within period)
  end_s      int32
  is_goalie  bool

Run once:
    python src/models/build_shift_lookup.py

Output: data/shots/shift_lookup.parquet
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

_ROOT        = Path(__file__).parent.parent.parent   # project root
SHIFTS_DIR   = _ROOT / "data" / "raw" / "shifts"
PLAYERS_DIR  = _ROOT / "data" / "raw" / "players"
OUTPUT_PATH  = _ROOT / "data" / "shots" / "shift_lookup.parquet"

def mmss_to_seconds(s: str) -> int:
    """Convert 'MM:SS' to total seconds."""
    try:
        m, sec = s.strip().split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return -1

def build_goalie_set() -> set:
    """Return a set of player IDs whose position is 'G'."""
    goalies: set = set()
    for pf in PLAYERS_DIR.glob("player_*.json"):
        try:
            with open(pf) as f:
                d = json.load(f)
            if str(d.get("position", "")).upper() == "G":
                goalies.add(int(d["playerId"]))
        except Exception:
            pass
    print(f"  Found {len(goalies):,} goalies in player files")
    return goalies


def build():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading goalie IDs from player files...")
    goalies = build_goalie_set()

    rows = []
    shift_dirs = sorted(SHIFTS_DIR.iterdir())
    total = sum(len(list(d.iterdir())) for d in shift_dirs)
    done = 0

    print(f"Processing {total:,} shift files...")
    for year_dir in shift_dirs:
        for shift_file in sorted(year_dir.iterdir()):
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{total:,} files...")
            try:
                with open(shift_file) as f:
                    d = json.load(f)
            except Exception:
                continue

            game_id = d.get("game_id")
            for shift in d.get("data", []):
                start = mmss_to_seconds(shift.get("startTime", ""))
                end   = mmss_to_seconds(shift.get("endTime",   ""))
                if start < 0 or end < 0 or end < start:
                    continue
                pid = shift.get("playerId")
                rows.append({
                    "game_id":   game_id,
                    "period":    shift.get("period"),
                    "player_id": pid,
                    "team_id":   shift.get("teamId"),
                    "start_s":   start,
                    "end_s":     end,
                    "is_goalie": (int(pid) in goalies) if pid is not None else False,
                })

    print(f"  Total shifts loaded: {len(rows):,}")
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["game_id", "period", "player_id", "team_id"])
    df = df.astype({
        "game_id":   "int64",
        "period":    "int64",
        "player_id": "int64",
        "team_id":   "int64",
        "start_s":   "int32",
        "end_s":     "int32",
        "is_goalie": bool,
    })
    n_goalie_rows = df["is_goalie"].sum()
    print(f"  Goalie shift rows: {n_goalie_rows:,} ({n_goalie_rows/len(df)*100:.1f}%)")
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved -> {OUTPUT_PATH}  ({len(df):,} rows)")

if __name__ == "__main__":
    print("Building shift lookup table...")
    build()
    print("Done.")
