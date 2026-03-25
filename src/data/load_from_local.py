"""
Load NHL data from local JSON files into the database.
This reads the raw JSON files downloaded by download_raw_data.py
and populates the SQLite database without making any API calls.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database import get_connection


class LocalDataLoader:
    """Loads NHL data from local JSON files into the database."""
    
    def __init__(self, data_dir='data/raw'):
        """
        Initialize the loader.
        
        Args:
            data_dir: Directory containing JSON files
        """
        self.data_dir = Path(data_dir)
        self.games_dir = self.data_dir / 'games'
        self.players_dir = self.data_dir / 'players'
        self.teams_dir = self.data_dir / 'teams'
        self.schedules_dir = self.data_dir / 'schedules'
        self.conn = get_connection()
        
        # Verify directories exist
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir.absolute()}\n"
                "Run download_raw_data.py first to download the data."
            )
    
    def load_teams(self):
        """Load teams from local JSON file."""
        teams_file = self.teams_dir / 'teams.json'
        
        if not teams_file.exists():
            print(f"⚠️  Teams file not found: {teams_file}")
            return
        
        print("Loading teams from local file...")
        cursor = self.conn.cursor()
        
        try:
            with open(teams_file, 'r', encoding='utf-8') as f:
                teams_data = json.load(f)
            
            for team_id, team in teams_data.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO teams (team_id, team_abbr, team_name, franchise_id)
                    VALUES (?, ?, ?, ?)
                """, (
                    int(team_id),
                    team.get('abbrev', ''),
                    team.get('name', {}).get('default', '') if isinstance(team.get('name'), dict) else team.get('name', ''),
                    team.get('franchiseId')
                ))
            
            self.conn.commit()
            print(f"✅ Loaded {len(teams_data)} teams")
            
        except Exception as e:
            print(f"❌ Error loading teams: {e}")
    
    def load_game_from_file(self, game_file):
        """
        Load a single game from JSON file into database.
        
        Args:
            game_file: Path to game JSON file
        
        Returns:
            Number of events stored
        """
        cursor = self.conn.cursor()
        
        try:
            with open(game_file, 'r', encoding='utf-8') as f:
                pbp = json.load(f)
            
            game_id = pbp.get('id')
            if not game_id:
                return 0
            
            # Extract game metadata
            game_date = pbp.get('gameDate', '')[:10]  # Extract date part
            season = pbp.get('season', '')
            game_type = pbp.get('gameType', 2)  # 2=regular, 3=playoff
            
            # Get team IDs
            home_team = pbp.get('homeTeam', {})
            away_team = pbp.get('awayTeam', {})
            home_team_id = home_team.get('id')
            away_team_id = away_team.get('id')
            
            # Get final score if game is final
            home_score = home_team.get('score')
            away_score = away_team.get('score')
            game_state = pbp.get('gameState', 'UNKNOWN')
            
            # Store game
            cursor.execute("""
                INSERT OR REPLACE INTO games 
                (game_id, season, game_type, game_date, home_team_id, away_team_id, 
                 home_score, away_score, game_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, season, game_type, game_date, home_team_id, away_team_id,
                  home_score, away_score, game_state))
            
            # Process plays
            plays = pbp.get('plays', [])
            events_stored = 0

            # Track running score throughout the game (updated on goal events)
            running_home_score = 0
            running_away_score = 0

            for idx, play in enumerate(plays):
                event_type = play.get('typeDescKey', '')

                if event_type not in {
                    'shot-on-goal', 'goal', 'missed-shot', 'blocked-shot',
                    'faceoff', 'hit', 'takeaway', 'giveaway',
                }:
                    continue

                details = play.get('details', {})
                period_descriptor = play.get('periodDescriptor', {})
                period = period_descriptor.get('number', 0)

                if period_descriptor.get('periodType', '') == 'SO':
                    continue

                time_in_period = play.get('timeInPeriod', '')

                time_remaining_raw = play.get('timeRemaining', 0)
                try:
                    if isinstance(time_remaining_raw, str) and ':' in time_remaining_raw:
                        mm, ss = time_remaining_raw.split(':')
                        time_remaining = int(mm) * 60 + int(ss)
                    else:
                        time_remaining = int(time_remaining_raw) if time_remaining_raw else 0
                except (ValueError, TypeError):
                    time_remaining = 0

                x_coord = details.get('xCoord')
                y_coord = details.get('yCoord')
                zone_code = details.get('zoneCode', '')
                shot_type = details.get('shotType', '')

                shooting_player_id = details.get('scoringPlayerId') if event_type == 'goal' else details.get('shootingPlayerId')
                goalie_id = details.get('goalieInNetId')
                shooting_team_id = details.get('eventOwnerTeamId')
                miss_reason = details.get('reason') if event_type == 'missed-shot' else None

                is_goal = 1 if event_type == 'goal' else 0
                if event_type == 'goal':
                    post_home = details.get('homeScore')
                    post_away = details.get('awayScore')
                    if post_home is not None and post_away is not None:
                        if shooting_team_id == home_team_id:
                            home_score_at_event = int(post_home) - 1
                            away_score_at_event = int(post_away)
                        else:
                            home_score_at_event = int(post_home)
                            away_score_at_event = int(post_away) - 1
                        running_home_score = int(post_home)
                        running_away_score = int(post_away)
                    else:
                        home_score_at_event = running_home_score - (1 if shooting_team_id == home_team_id else 0)
                        away_score_at_event = running_away_score - (1 if shooting_team_id != home_team_id else 0)
                        running_home_score += (1 if shooting_team_id == home_team_id else 0)
                        running_away_score += (1 if shooting_team_id != home_team_id else 0)
                else:
                    home_score_at_event = running_home_score
                    away_score_at_event = running_away_score

                is_empty_net = 1 if goalie_id is None and event_type in ('shot-on-goal', 'goal') else 0

                cursor.execute("""
                    INSERT OR IGNORE INTO events
                    (game_id, event_idx, period, period_time, time_remaining, event_type,
                     x_coord, y_coord, zone_code, shot_type, shooting_player_id, goalie_id,
                     shooting_team_id, home_team_id, away_team_id, home_score, away_score,
                     is_goal, is_empty_net, miss_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (game_id, idx, period, time_in_period, time_remaining, event_type,
                      x_coord, y_coord, zone_code, shot_type, shooting_player_id, goalie_id,
                      shooting_team_id, home_team_id, away_team_id,
                      home_score_at_event, away_score_at_event, is_goal, is_empty_net,
                      miss_reason))

                events_stored += 1

                if shooting_player_id:
                    cursor.execute("""
                        INSERT OR IGNORE INTO players (player_id, full_name) VALUES (?, ?)
                    """, (shooting_player_id, f"Player_{shooting_player_id}"))

                if goalie_id:
                    cursor.execute("""
                        INSERT OR IGNORE INTO players (player_id, full_name) VALUES (?, ?)
                    """, (goalie_id, f"Goalie_{goalie_id}"))
            
            self.conn.commit()
            return events_stored
            
        except Exception as e:
            print(f"❌ Error loading game from {game_file.name}: {e}")
            return 0
    
    def load_all_games(self, seasons=None):
        """
        Load all games from local JSON files.
        
        Args:
            seasons: Optional list of seasons to load (e.g., ['2023', '2024'])
                    If None, loads all available seasons
        """
        print("Loading games from local files...")
        
        # Find all game files
        if seasons:
            game_files = []
            for season in seasons:
                season_dir = self.games_dir / season
                if season_dir.exists():
                    game_files.extend(list(season_dir.glob('game_*.json')))
        else:
            game_files = list(self.games_dir.rglob('game_*.json'))
        
        if not game_files:
            print("⚠️  No game files found")
            return
        
        print(f"Found {len(game_files)} game files")
        
        total_events = 0
        success_count = 0
        error_count = 0
        failed_games = []
        
        for game_file in tqdm(game_files, desc="Loading games"):
            events = self.load_game_from_file(game_file)
            if events > 0:
                total_events += events
                success_count += 1
            else:
                error_count += 1
                # Extract game ID from filename (game_2023020001.json -> 2023020001)
                game_id = game_file.stem.replace('game_', '')
                failed_games.append(game_id)
        
        print(f"\n✅ Loaded {success_count} games with {total_events} shot events")
        if error_count > 0:
            print(f"❌ Failed to load {error_count} games")
            
            # Save failed game IDs to file
            failed_file = Path('data/failed_games.txt')
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed_games))
            print(f"📝 Failed game IDs saved to: {failed_file}")
    
    def load_player_from_file(self, player_file):
        """
        Load a single player from JSON file into database.
        
        Args:
            player_file: Path to player JSON file
        
        Returns:
            True if successful, False otherwise
        """
        cursor = self.conn.cursor()
        
        try:
            with open(player_file, 'r', encoding='utf-8') as f:
                player_data = json.load(f)
            
            # Extract player ID from filename (player_8478402.json -> 8478402)
            player_id = int(player_file.stem.split('_')[1])
            
            # Extract player details
            first_name = player_data.get('firstName', {})
            last_name = player_data.get('lastName', {})
            
            if isinstance(first_name, dict):
                first_name = first_name.get('default', '')
            if isinstance(last_name, dict):
                last_name = last_name.get('default', '')
            
            full_name = f"{first_name} {last_name}".strip()
            position = player_data.get('position')
            shoots_catches = player_data.get('shootsCatches')
            # Strip surrounding single-quotes if present (e.g. "'L'" -> "L")
            if isinstance(shoots_catches, str):
                shoots_catches = shoots_catches.strip("'")
                if shoots_catches not in ('L', 'R'):
                    shoots_catches = None
            
            # Update database
            if full_name or position or shoots_catches:
                cursor.execute("""
                    UPDATE players 
                    SET full_name = COALESCE(?, full_name),
                        position = COALESCE(?, position),
                        shoots_catches = COALESCE(?, shoots_catches)
                    WHERE player_id = ?
                """, (full_name, position, shoots_catches, player_id))
                
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def load_all_players(self):
        """Load all players from local JSON files."""
        print("Loading players from local files...")
        
        player_files = list(self.players_dir.glob('player_*.json'))
        
        if not player_files:
            print("⚠️  No player files found")
            return
        
        print(f"Found {len(player_files)} player files")
        
        success_count = 0
        error_count = 0
        failed_players = []
        
        for player_file in tqdm(player_files, desc="Loading players"):
            if self.load_player_from_file(player_file):
                success_count += 1
            else:
                error_count += 1
                # Extract player ID from filename (player_8471214.json -> 8471214)
                player_id = player_file.stem.replace('player_', '')
                failed_players.append(player_id)
            
            # Commit every 100 players
            if (success_count + error_count) % 100 == 0:
                self.conn.commit()
        
        # Final commit
        self.conn.commit()
        
        print(f"\n✅ Loaded {success_count} players")
        if error_count > 0:
            print(f"❌ Failed to load {error_count} players")
            
            # Save failed player IDs to file
            failed_file = Path('data/failed_players.txt')
            with open(failed_file, 'w') as f:
                f.write('\n'.join(failed_players))
            print(f"📝 Failed player IDs saved to: {failed_file}")
    
    # ------------------------------------------------------------------
    # Shift data (ice time)
    # ------------------------------------------------------------------

    @staticmethod
    def _mmss_to_seconds(mmss: str) -> int | None:
        """Convert 'MM:SS' string to total seconds. Returns None if blank."""
        if not mmss:
            return None
        parts = mmss.split(':')
        if len(parts) == 2:
            try:
                return int(parts[0]) * 60 + int(parts[1])
            except ValueError:
                return None
        return None

    def load_shifts_for_game(self, game_id: int, shifts_dir: str = 'data/raw/shifts') -> int:
        """
        Load shift data for a single game from its local JSON file into the
        ``player_shifts`` table.

        Args:
            game_id:    NHL game ID.
            shifts_dir: Root directory containing shift JSON files.

        Returns:
            Number of shift rows inserted.
        """
        season_year = str(game_id)[:4]
        shift_file = Path(shifts_dir) / season_year / f"{game_id}.json"

        if not shift_file.exists():
            return 0

        cursor = self.conn.cursor()
        try:
            with open(shift_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            shifts = payload.get('data', [])
            inserted = 0

            for s in shifts:
                # Skip non-regular-shift type codes (e.g. 505 = goal markers)
                if s.get('typeCode') != 517:
                    continue

                player_id = s.get('playerId')
                team_id = s.get('teamId')
                period = s.get('period')
                shift_number = s.get('shiftNumber')
                start_time = s.get('startTime', '')
                end_time = s.get('endTime', '')
                duration = s.get('duration', '')

                start_sec = self._mmss_to_seconds(start_time)
                end_sec = self._mmss_to_seconds(end_time)
                dur_sec = self._mmss_to_seconds(duration)

                cursor.execute("""
                    INSERT OR IGNORE INTO player_shifts
                        (game_id, player_id, team_id, period, shift_number,
                         start_time, end_time, duration,
                         start_seconds, end_seconds, duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id, player_id, team_id, period, shift_number,
                    start_time, end_time, duration,
                    start_sec, end_sec, dur_sec,
                ))

                # Ensure player stub exists so FK is satisfied
                if player_id:
                    first = s.get('firstName', '')
                    last = s.get('lastName', '')
                    full_name = f"{first} {last}".strip() or f"Player_{player_id}"
                    cursor.execute("""
                        INSERT OR IGNORE INTO players (player_id, full_name)
                        VALUES (?, ?)
                    """, (player_id, full_name))

                inserted += 1

            self.conn.commit()
            return inserted

        except Exception as e:
            print(f"❌ Error loading shifts for game {game_id}: {e}")
            return 0

    def load_all_shifts(self, shifts_dir: str = 'data/raw/shifts', seasons: list | None = None):
        """
        Load all locally saved shift JSON files into the ``player_shifts`` table.

        Args:
            shifts_dir: Root directory that contains per-season sub-folders of
                        shift JSON files (produced by fetch_shift_data.py).
            seasons:    Optional list of season-year strings (e.g. ['2024', '2025'])
                        to restrict which sub-folders are scanned.
        """
        shifts_path = Path(shifts_dir)
        if not shifts_path.exists():
            print(f"⚠️  Shifts directory not found: {shifts_path.absolute()}")
            print("   Run fetch_shift_data.py first to download shift data.")
            return

        # Collect all shift files
        if seasons:
            shift_files = []
            for season in seasons:
                season_dir = shifts_path / season
                if season_dir.exists():
                    shift_files.extend(list(season_dir.glob('*.json')))
        else:
            shift_files = list(shifts_path.rglob('*.json'))

        if not shift_files:
            print("⚠️  No shift files found.")
            return

        print(f"Loading shifts from {len(shift_files)} game files...")

        total_rows = 0
        success = 0
        skipped = 0

        for sf in tqdm(shift_files, desc="Loading shifts"):
            try:
                game_id = int(sf.stem)
            except ValueError:
                skipped += 1
                continue

            rows = self.load_shifts_for_game(game_id, shifts_dir=shifts_dir)
            if rows > 0:
                total_rows += rows
                success += 1
            else:
                skipped += 1

        print(f"\n✅ Loaded {total_rows:,} shift rows from {success} games")
        if skipped:
            print(f"⚠️  Skipped {skipped} files (empty or parse error)")

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Load NHL data from local JSON files into database')
    parser.add_argument('--data-dir', default='data/raw',
                       help='Directory containing JSON files')
    parser.add_argument('--seasons', nargs='+',
                       help='Specific seasons to load (e.g., 2023 2024). If not specified, loads all.')
    parser.add_argument('--skip-teams', action='store_true',
                       help='Skip loading teams')
    parser.add_argument('--skip-games', action='store_true',
                       help='Skip loading games')
    parser.add_argument('--skip-players', action='store_true',
                       help='Skip loading players')
    parser.add_argument('--load-shifts', action='store_true',
                       help='Also load shift (ice-time) data from data/raw/shifts/')
    parser.add_argument('--shifts-only', action='store_true',
                       help='Only load shift data (skip teams/games/players)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NHL LOCAL DATA LOADER")
    print("="*80)
    print(f"Data directory: {Path(args.data_dir).absolute()}")
    if args.seasons:
        print(f"Seasons: {args.seasons}")
    else:
        print("Seasons: All available")
    print("="*80)
    
    try:
        loader = LocalDataLoader(data_dir=args.data_dir)
        
        if args.shifts_only:
            shifts_dir = str(Path(args.data_dir) / 'shifts')
            loader.load_all_shifts(shifts_dir=shifts_dir, seasons=args.seasons)
        else:
            # Load teams
            if not args.skip_teams:
                loader.load_teams()
            
            # Load games
            if not args.skip_games:
                loader.load_all_games(seasons=args.seasons)
            
            # Load players
            if not args.skip_players:
                loader.load_all_players()

            # Optionally load shifts
            if args.load_shifts:
                shifts_dir = str(Path(args.data_dir) / 'shifts')
                loader.load_all_shifts(shifts_dir=shifts_dir, seasons=args.seasons)
        
        loader.close()
        
        print("\n" + "="*80)
        print("LOADING COMPLETE!")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease run download_raw_data.py first to download the data.")
        sys.exit(1)


if __name__ == "__main__":
    main()
