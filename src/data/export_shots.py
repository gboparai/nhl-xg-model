"""
export_shots.py
===============
Builds the shot-event CSV directly from the raw NHL game JSON files.

Output: data/shots/xg_table.csv.gz  (default)

Run:
    .venv/Scripts/python.exe export_shots.py
    .venv/Scripts/python.exe export_shots.py --shot-population no-blocked
    .venv/Scripts/python.exe export_shots.py --shot-population all-attempts
    .venv/Scripts/python.exe export_shots.py --shot-population sog-missed

Shot population presets
-----------------------
  default      shot-on-goal + blocked-shot + goal          (baseline)
  no-blocked   shot-on-goal + goal
  all-attempts shot-on-goal + blocked-shot + missed-shot + goal
  sog-missed   shot-on-goal + missed-shot + goal

Key schema decisions:
  - empty_net = 1 for blocked-shots and empty-net goals, 0 otherwise
  - homeScore / awayScore are NULL until the first goal of the game (not backfilled)
  - ALL events update the prev_event tracker (not just shots)
  - prev_event_type is recorded
  - is_home flag included (1 = shooting team is home team)
  - preseason (gameType=1) IS included in the output (notebook filters by season >= 20132014)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # project root
from src.database import get_db_path

# ── paths ────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).parent.parent.parent   # project root


RAW_DIR  = _ROOT / "data" / "raw" / "games"
OUT_PATH = _ROOT / "data" / "shots" / "xg_table.csv.gz"

# Default shot population: shot-on-goal + blocked-shot + goal (NO missed-shot)
_DEFAULT_SHOT_TYPES = frozenset({"shot-on-goal", "blocked-shot", "goal"})

# Named presets for --shot-population
SHOT_POPULATIONS: dict[str, tuple[frozenset, str]] = {
    "default":      (frozenset({"shot-on-goal", "blocked-shot", "goal"}),
                     "xg_table.csv.gz"),
    "no-blocked":   (frozenset({"shot-on-goal", "goal"}),
                     "xg_table_no_blocked.csv.gz"),
    "all-attempts": (frozenset({"shot-on-goal", "blocked-shot", "missed-shot", "goal"}),
                     "xg_table_all_attempts.csv.gz"),
    "sog-missed":   (frozenset({"shot-on-goal", "missed-shot", "goal"}),
                     "xg_table_sog_missed.csv.gz"),
}

# Backwards-compatible module-level constant (used when called without args)
SHOT_TYPES = _DEFAULT_SHOT_TYPES

# Include all gameTypes (notebook filters by season, not gameType)
SKIP_GAME_TYPES: set[int] = set()



def _load_player_positions(db_path: Path) -> dict[int, str]:
    """Return {player_id: position} from the players table."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT player_id, position FROM players").fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows if r[1]}
    except Exception:
        return {}


def _parse_game_data(data: dict, positions: dict[int, str],
                     shot_types: frozenset | set = _DEFAULT_SHOT_TYPES) -> list[dict]:
    """
    Parse a game JSON dict and return shot-event dicts matching the CSV schema.

    Key behaviours:
      - blocked-shot rows get empty_net=1, shot_made=0
      - homeScore/awayScore stay None until first goal fires (not backfilled from 0)
      - Every event (not just shots) updates the prev_event tracker
      - is_home = 1 if shooting team == home team
    """

    game_id   = data.get("id")
    season    = data.get("season")
    game_type = data.get("gameType", 2)

    if game_type in SKIP_GAME_TYPES or not game_id or not season:
        return []

    # Home / away team IDs
    home_team_id = (data.get("homeTeam") or {}).get("id")
    away_team_id = (data.get("awayTeam") or {}).get("id")

    plays = data.get("plays", [])
    if not plays:
        return []

    rows = []
    prev_x:        float | None = None
    prev_y:        float | None = None
    prev_time:     str   | None = None
    prev_period:   int   | None = None
    prev_type:     str   | None = None
    prev_team_id:  int   | None = None

    # homeScore / awayScore: None until the first goal is recorded
    home_score: int | None = None
    away_score: int | None = None

    event_id_counter = 0

    for play in plays:
        t_key        = play.get("typeDescKey", "")
        details      = play.get("details") or {}
        period       = play.get("periodDescriptor", {}).get("number")
        tip          = play.get("timeInPeriod", "")
        sit_code     = play.get("situationCode")
        nhl_event_id = play.get("eventId")

        # Skip shootouts (period 5+)
        if period is not None and period >= 5:
            prev_x = prev_y = prev_time = prev_period = None
            prev_type = prev_team_id = None
            continue

        x = details.get("xCoord")
        y = details.get("yCoord")

        # Determine shooting / blocking team for all relevant events
        event_team_id = details.get("eventOwnerTeamId")

        if t_key in shot_types:
            event_id_counter += 1
            is_goal = 1 if t_key == "goal" else 0

            # blocked-shot: always empty_net=1 (it's a shot attempt that was blocked)
            # missed-shot: empty_net=0, shot_made=0 (treated like shot-on-goal)
            # goal: check if goalie was pulled (goalieInNetId absent/None)
            if t_key == "blocked-shot":
                empty_net = 1
                shot_type = details.get("shotType", "")
                # blocked-shot: shootingPlayerId is the shooter
                shooter_id = details.get("shootingPlayerId")
                shooting_team_id = event_team_id
            elif t_key == "goal":
                goalie_id = details.get("goalieInNetId")
                empty_net = 0 if goalie_id else 1
                shot_type = details.get("shotType", "")
                shooter_id = details.get("scoringPlayerId")
                shooting_team_id = event_team_id
            else:  # shot-on-goal or missed-shot
                empty_net = 0
                shot_type = details.get("shotType", "")
                shooter_id = details.get("shootingPlayerId")
                shooting_team_id = event_team_id

            pos = positions.get(shooter_id, "") if shooter_id else ""
            is_forward = 1 if pos in ("L", "R", "C") else 0

            # is_home: 1 if shooting team is home team
            if shooting_team_id and home_team_id:
                is_home = 1 if shooting_team_id == home_team_id else 0
            else:
                is_home = None

            # Score at moment of shot (None if no goal yet in this game)
            # For goals: score BEFORE this goal (same as running total before update)
            rows.append({
                "game_id":           game_id,
                "nhl_event_id":      nhl_event_id,
                "event_id":          event_id_counter,
                "season":            season,
                "gameType":          game_type,
                "homeTeam_id":       home_team_id,
                "awayTeam_id":       away_team_id,
                "period_number":     period,
                "time_in_period":    tip,
                "type_desc_key":     t_key,
                "shotType":          shot_type,
                "situation_code":    sit_code,
                "xCoord":            x,
                "yCoord":            y,
                "shot_made":         is_goal,
                "shootingPlayerId":  shooter_id if not is_goal else None,
                "scoringPlayerId":   shooter_id if is_goal else None,
                "shooting_team_id":  shooting_team_id,
                "is_forward":        is_forward,
                "is_home":           is_home,
                "homeScore":         home_score,   # None until first goal
                "awayScore":         away_score,   # None until first goal
                "empty_net":         empty_net,
                "prev_event_type":   prev_type,
                "prev_event_team_id":prev_team_id,
                "prev_event_x":      prev_x,
                "prev_event_y":      prev_y,
                "prev_event_time":   prev_time,
                "prev_event_period": prev_period,
            })

            # Update running score AFTER recording the row (score before the event)
            if is_goal:
                hs = details.get("homeScore")
                as_ = details.get("awayScore")
                if hs is not None:
                    home_score = int(hs)
                if as_ is not None:
                    away_score = int(as_)

        elif t_key == "goal":
            # Should not happen since "goal" is in SHOT_TYPES, but guard anyway
            pass
        else:
            # Non-shot event: update running score if it's a goal carried in details
            # (shouldn't happen ΓÇö goals have typeDescKey="goal")
            pass

        # Update prev-event tracker for EVERY event (including non-shots)
        prev_time    = tip
        prev_period  = period
        prev_x       = x
        prev_y       = y
        prev_type    = t_key
        prev_team_id = event_team_id

    return rows


def _parse_game(path: Path, positions: dict[int, str],
                shot_types: frozenset | set = _DEFAULT_SHOT_TYPES) -> list[dict]:
    """Load a game JSON file and delegate to _parse_game_data."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    return _parse_game_data(data, positions, shot_types)


def main(shot_types: frozenset | set | None = None,
         out_path: Path | str | None = None) -> None:
    """
    Export shot events to CSV.

    Args:
        shot_types: Set of NHL event typeDescKey strings to include.
                    Defaults to the 'default' preset (SOG + blocked + goal).
        out_path:   Output CSV path. Defaults to data/shots/xg_table.csv.gz.
    """
    if shot_types is None:
        shot_types = _DEFAULT_SHOT_TYPES
    resolved_out = Path(out_path) if out_path else OUT_PATH

    db_path = get_db_path()
    print(f"Loading player positions from {db_path} ...")
    positions = _load_player_positions(db_path)
    print(f"  {len(positions):,} players loaded")
    print(f"  Shot population  : {sorted(shot_types)}")
    print(f"  Output           : {resolved_out}")

    season_dirs = sorted(RAW_DIR.glob("*"))
    if not season_dirs:
        print(f"No game files found under {RAW_DIR}. Exiting.")
        sys.exit(1)

    all_rows: list[dict] = []
    total_files = 0

    for season_dir in season_dirs:
        files = sorted(season_dir.glob("*.json"))
        if not files:
            continue
        season_rows_before = len(all_rows)
        for f in files:
            all_rows.extend(_parse_game(f, positions, shot_types))
            total_files += 1
        season_shots = len(all_rows) - season_rows_before
        print(f"  {season_dir.name}: {len(files):>4} games  ->  {season_shots:>7,} shots")

    print(f"\nTotal: {total_files:,} games  ->  {len(all_rows):,} shots")

    df = pd.DataFrame(all_rows)

    # Type cleanup
    df["season"]        = pd.to_numeric(df["season"],        errors="coerce").astype("Int64")
    df["period_number"] = pd.to_numeric(df["period_number"], errors="coerce").astype("Int64")
    df["shot_made"]     = df["shot_made"].astype(int)
    df["is_forward"]    = df["is_forward"].astype(int)
    df["empty_net"]     = df["empty_net"].astype(int)
    # homeScore / awayScore: keep as float so NaN survives (matches original)
    df["homeScore"]     = pd.to_numeric(df["homeScore"], errors="coerce")
    df["awayScore"]     = pd.to_numeric(df["awayScore"], errors="coerce")
    df["situation_code"] = pd.to_numeric(df["situation_code"], errors="coerce")

    print(f"\nGoal rate        : {df['shot_made'].mean()*100:.2f}%")
    print(f"homeScore NaN    : {df['homeScore'].isna().mean()*100:.1f}%  (expected ~20%)")
    print(f"empty_net=1 rate : {df['empty_net'].mean()*100:.1f}%")
    print(f"prev_x coverage  : {df['prev_event_x'].notna().mean()*100:.1f}% non-null")
    print(f"\ntype_desc_key breakdown:")
    print(df["type_desc_key"].value_counts().to_string())
    print(f"\ngameType breakdown:")
    print(df["gameType"].value_counts().sort_index().to_string())

    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {resolved_out} ...")
    df.to_csv(resolved_out, index=False, compression="gzip")
    size_mb = resolved_out.stat().st_size / 1_048_576
    print(f"Done -- {size_mb:.1f} MB  ({len(df):,} rows)")
    print(f"\nNext:  python main.py --evaluate --retrain --skip-fetch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export NHL shot events to CSV")
    parser.add_argument(
        "--shot-population",
        choices=list(SHOT_POPULATIONS.keys()),
        default="default",
        help="Shot population preset (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override output CSV path (default: derived from --shot-population)",
    )
    args = parser.parse_args()
    _types, _filename = SHOT_POPULATIONS[args.shot_population]
    _out = Path(args.out) if args.out else (_ROOT / "data" / "shots" / _filename)
    main(shot_types=_types, out_path=_out)
