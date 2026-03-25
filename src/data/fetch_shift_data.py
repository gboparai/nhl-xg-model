"""
Fetch NHL shift data (ice time) per game from the NHL REST stats API.
Saves to local JSON files under data/raw/shifts/<season>/<game_id>.json

API endpoint:
    https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<game_id>

Each shift record contains:
    - playerId, firstName, lastName
    - teamId, teamAbbrev
    - period, startTime, endTime, duration  (all as "MM:SS" strings)
    - shiftNumber, typeCode
    - gameId
"""

import json
import time
import argparse
import sqlite3
import requests
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database import get_connection

SHIFT_API_URL = "https://api.nhle.com/stats/rest/en/shiftcharts"

# typeCode 517 = regular shift, 505 = goal event marker in shift data
# We only want typeCode 517 (real ice-time shifts)
REGULAR_SHIFT_TYPECODE = 517


def mmss_to_seconds(mmss: str) -> int | None:
    """Convert 'MM:SS' string to total seconds. Returns None if blank/null."""
    if not mmss:
        return None
    parts = mmss.split(':')
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    return None


def fetch_shifts_for_game(game_id: int, session: requests.Session, retries: int = 3) -> list[dict] | None:
    """
    Fetch shift chart data for a single game from the NHL API.

    Returns:
        List of shift dicts, or None on failure.
    """
    params = {"cayenneExp": f"gameId={game_id}"}
    for attempt in range(retries):
        try:
            resp = session.get(SHIFT_API_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except requests.HTTPError as e:
            if resp.status_code == 404:
                return []  # Game not found - skip
            print(f"  HTTP error for game {game_id}: {e}  (attempt {attempt+1}/{retries})")
        except Exception as e:
            print(f"  Error fetching shifts for game {game_id}: {e}  (attempt {attempt+1}/{retries})")
        time.sleep(1.0 * (attempt + 1))
    return None


def save_shifts_for_game(game_id: int, shifts: list[dict], shifts_dir: Path) -> Path:
    """
    Save raw shift data for a game to a JSON file.

    File path: <shifts_dir>/<season>/<game_id>.json
    Season is derived from the first 8 digits of game_id  (e.g. 2025020869 -> 20252026)
    We use the 4-digit year prefix of the game_id as the folder name to match the
    existing games/ layout which uses the 4-digit season start year.
    """
    season_year = str(game_id)[:4]  # e.g. "2025"
    season_dir = shifts_dir / season_year
    season_dir.mkdir(parents=True, exist_ok=True)

    out_file = season_dir / f"{game_id}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"game_id": game_id, "data": shifts}, f, indent=2)
    return out_file


def get_game_ids_from_db(seasons: list[str] | None = None) -> list[int]:
    """
    Retrieve all game_ids from the local database, optionally filtered by season.
    Season strings match the `season` column (e.g. '20252026').
    """
    conn = get_connection()
    cursor = conn.cursor()

    if seasons:
        placeholders = ','.join('?' for _ in seasons)
        cursor.execute(
            f"SELECT game_id FROM games WHERE season IN ({placeholders}) ORDER BY game_id",
            seasons,
        )
    else:
        cursor.execute("SELECT game_id FROM games ORDER BY game_id")

    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


def get_already_fetched(shifts_dir: Path, game_ids: list[int]) -> set[int]:
    """Return the subset of game_ids that already have a local shift file."""
    fetched = set()
    for gid in game_ids:
        season_year = str(gid)[:4]
        p = shifts_dir / season_year / f"{gid}.json"
        if p.exists():
            fetched.add(gid)
    return fetched


def fetch_all_shifts(
    seasons: list[str] | None = None,
    data_dir: str = 'data/raw',
    rate_limit: float = 0.3,
    skip_existing: bool = True,
    game_ids: list[int] | None = None,
):
    """
    Fetch shift data for all games (or a subset) and save locally.

    Args:
        seasons:      List of season strings to fetch (e.g. ['20252026']).
                      If None and game_ids is None, fetches all games in the DB.
        data_dir:     Root raw data directory.
        rate_limit:   Seconds between API requests.
        skip_existing: If True, skip games that already have a local file.
        game_ids:     Explicit list of integer game IDs to fetch (overrides seasons).
    """
    shifts_dir = Path(data_dir) / 'shifts'
    shifts_dir.mkdir(parents=True, exist_ok=True)

    if game_ids is None:
        game_ids = get_game_ids_from_db(seasons)

    if not game_ids:
        print("No game IDs found. Make sure the DB is populated.")
        return

    if skip_existing:
        already = get_already_fetched(shifts_dir, game_ids)
        remaining = [gid for gid in game_ids if gid not in already]
        print(f"Games to fetch: {len(remaining)} (skipping {len(already)} already downloaded)")
    else:
        remaining = game_ids
        print(f"Games to fetch: {len(remaining)}")

    if not remaining:
        print("All shift files already present. Nothing to do.")
        return

    success = 0
    empty = 0
    failed = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "AdvancedHockeyAnalytics/1.0"})

    for game_id in tqdm(remaining, desc="Fetching shifts"):
        shifts = fetch_shifts_for_game(game_id, session)
        if shifts is None:
            failed += 1
            continue

        # Filter to real shifts only (drop goal-event markers with typeCode=505)
        real_shifts = [s for s in shifts if s.get('typeCode') == REGULAR_SHIFT_TYPECODE]

        save_shifts_for_game(game_id, real_shifts, shifts_dir)

        if real_shifts:
            success += 1
        else:
            empty += 1

        time.sleep(rate_limit)

    session.close()

    print(f"\n✅ Fetched {success} games with shifts")
    if empty:
        print(f"⚠️  {empty} games returned no shifts (pre-season / postponed?)")
    if failed:
        print(f"❌ {failed} games failed after retries")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch NHL shift (ice-time) data and save as local JSON files."
    )
    parser.add_argument(
        '--seasons', nargs='+',
        help='Season strings to fetch (e.g. 20242025 20252026). Defaults to all in DB.'
    )
    parser.add_argument(
        '--game-ids', nargs='+', type=int,
        help='Specific game IDs to fetch (overrides --seasons).'
    )
    parser.add_argument(
        '--data-dir', default='data/raw',
        help='Root raw data directory (default: data/raw).'
    )
    parser.add_argument(
        '--rate-limit', type=float, default=0.3,
        help='Seconds between API requests (default: 0.3).'
    )
    parser.add_argument(
        '--no-skip', action='store_true',
        help='Re-fetch even if local file already exists.'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NHL SHIFT DATA FETCHER")
    print("=" * 70)

    fetch_all_shifts(
        seasons=args.seasons,
        data_dir=args.data_dir,
        rate_limit=args.rate_limit,
        skip_existing=not args.no_skip,
        game_ids=args.game_ids,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
