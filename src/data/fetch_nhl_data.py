"""
NHL API data fetching module.
Fetches play-by-play data from NHL API and stores in SQLite database.
Can also load data from local JSON files (see load_from_local.py).
"""

import time
import argparse
from datetime import datetime
from nhlpy import NHLClient
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database import get_connection
from src.data.load_from_local import LocalDataLoader


class NHLDataFetcher:
    """Fetches and stores NHL play-by-play data."""
    
    def __init__(self, rate_limit_delay=0.5, use_local=False, local_data_dir='data/raw'):
        """
        Initialize the data fetcher.
        
        Args:
            rate_limit_delay: Delay in seconds between API requests
            use_local: If True, load data from local JSON files instead of API
            local_data_dir: Directory containing local JSON files
        """
        self.rate_limit_delay = rate_limit_delay
        self.use_local = use_local
        self.local_data_dir = local_data_dir
        self.conn = get_connection()
        
        if use_local:
            self.local_loader = LocalDataLoader(data_dir=local_data_dir)
        else:
            self.client = NHLClient()
            self.local_loader = None
        
    def fetch_teams(self):
        """Fetch and store all NHL teams."""
        if self.use_local:
            print("Loading teams from local files...")
            self.local_loader.load_teams()
            return
        
        print("Fetching NHL teams from API...")
        cursor = self.conn.cursor()
        
        # NHL team IDs are typically 1-54 (with gaps)
        # We'll try to get teams from a recent game schedule
        try:
            schedule = self.client.schedule.daily_schedule()
            teams_seen = set()
            
            if schedule and 'gameWeek' in schedule:
                for day in schedule['gameWeek']:
                    if 'games' in day:
                        for game in day['games']:
                            for team_key in ['homeTeam', 'awayTeam']:
                                if team_key in game:
                                    team = game[team_key]
                                    team_id = team.get('id')
                                    if team_id and team_id not in teams_seen:
                                        teams_seen.add(team_id)
                                        cursor.execute("""
                                            INSERT OR REPLACE INTO teams (team_id, team_abbr, team_name, franchise_id)
                                            VALUES (?, ?, ?, ?)
                                        """, (
                                            team_id,
                                            team.get('abbrev', ''),
                                            team.get('name', {}).get('default', '') if isinstance(team.get('name'), dict) else team.get('name', ''),
                                            team.get('franchiseId')
                                        ))
            
            self.conn.commit()
            print(f"Stored {len(teams_seen)} teams")
        except Exception as e:
            print(f"Error fetching teams: {e}")
    
    def fetch_season_schedule(self, season, team_abbr=None):
        """
        Fetch schedule for a season.
        
        Args:
            season: Season in format "20232024"
            team_abbr: Optional team abbreviation to filter (e.g., "TOR")
        
        Returns:
            List of game IDs
        """
        game_ids = []
        
        try:
            if team_abbr:
                # Get schedule for specific team
                schedule = self.client.schedule.team_season_schedule(team_abbr=team_abbr, season=season)
                
                if schedule and 'games' in schedule:
                    for game in schedule['games']:
                        game_ids.append(game['id'])
            
        except Exception as e:
            print(f"Error fetching schedule for team {team_abbr} season {season}: {e}")
        
        return game_ids
    
    def fetch_all_season_games(self, season):
        """
        Fetch all games for a season by getting schedules for all teams.
        
        Args:
            season: Season in format "20232024"
        
        Returns:
            Set of unique game IDs
        """
        # All NHL team abbreviations (current + historical 2010-2026)
        # Includes: ATL (2010-11), PHX (2010-2014), ARI (2014-2024), UTA (2024+)
        teams = [
            'ANA', 'ARI', 'ATL', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY',
            'CHI', 'COL', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN',
            'MTL', 'NJD', 'NSH', 'NYI', 'NYR', 'OTT', 'PHI', 'PHX',
            'PIT', 'SEA', 'SJS', 'STL', 'TBL', 'TOR', 'UTA', 'VAN',
            'VGK', 'WPG', 'WSH',
        ]
        
        game_ids = set()
        
        print(f"Fetching season {season} schedule from all teams...")
        for team in tqdm(teams, desc="Teams"):
            try:
                schedule = self.client.schedule.team_season_schedule(team_abbr=team, season=season)
                
                if schedule and 'games' in schedule:
                    for game in schedule['games']:
                        game_ids.add(game['id'])
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                # Silently continue for teams that don't exist in this season (e.g., SEA before 2021)
                continue
        
        return list(game_ids)
    
    def fetch_game_data(self, game_id):
        """
        Fetch and store play-by-play data for a single game.
        
        Args:
            game_id: NHL game ID
        
        Returns:
            Number of events stored
        """
        cursor = self.conn.cursor()
        
        try:
            # Fetch play-by-play data
            pbp = self.client.game_center.play_by_play(game_id)
            
            if not pbp:
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

                STORED_EVENT_TYPES = {
                    'shot-on-goal', 'goal', 'missed-shot', 'blocked-shot',
                    'faceoff', 'hit', 'takeaway', 'giveaway',
                }
                if event_type not in STORED_EVENT_TYPES:
                    continue

                details = play.get('details', {})
                period_descriptor = play.get('periodDescriptor', {})
                period = period_descriptor.get('number', 0)

                if period_descriptor.get('periodType', '') == 'SO':
                    continue

                time_in_period = play.get('timeInPeriod', '')

                time_remaining_raw = play.get('timeRemaining', '')
                try:
                    if time_remaining_raw and ':' in str(time_remaining_raw):
                        mm, ss = str(time_remaining_raw).split(':')
                        time_remaining = int(mm) * 60 + int(ss)
                    elif time_remaining_raw:
                        time_remaining = int(time_remaining_raw)
                    else:
                        time_remaining = 0
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
                    INSERT OR REPLACE INTO events
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
            print(f"Error fetching game {game_id}: {e}")
            return 0
    
    def fetch_seasons(self, seasons):
        """
        Fetch data for multiple seasons.
        
        Args:
            seasons: List of seasons in format ["20232024", "20242025"]
        """
        if self.use_local:
            print("Loading data from local files...")
            # Convert season format from "20232024" to "2023" for directory structure
            season_dirs = [s[:4] for s in seasons]
            self.local_loader.load_all_games(seasons=season_dirs)
            self.local_loader.load_all_players()
            # Also load shift data from local files
            shifts_dir = str(Path(self.local_data_dir) / 'shifts')
            self.local_loader.load_all_shifts(shifts_dir=shifts_dir, seasons=season_dirs)
            return
        
        for season in seasons:
            print(f"\n=== Fetching season {season} from API ===")
            
            # Get all games for the season
            game_ids = self.fetch_all_season_games(season)
            print(f"Found {len(game_ids)} games for season {season}")
            
            # Fetch each game
            total_events = 0
            for game_id in tqdm(game_ids, desc=f"Games {season}"):
                events = self.fetch_game_data(game_id)
                total_events += events
                time.sleep(self.rate_limit_delay)
            
            print(f"Stored {total_events} shot events for season {season}")
    
    def close(self):
        """Close database connection."""
        if self.local_loader:
            self.local_loader.close()
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Fetch NHL play-by-play data')
    parser.add_argument('--seasons', nargs='+', default=['20232024', '20242025'],
                       help='Seasons to fetch (e.g., 20232024 20242025)')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                       help='Delay between API requests in seconds')
    parser.add_argument('--use-local', action='store_true',
                       help='Load from local JSON files instead of API')
    parser.add_argument('--local-dir', default='data/raw',
                       help='Directory containing local JSON files')
    
    args = parser.parse_args()
    
    fetcher = NHLDataFetcher(
        rate_limit_delay=args.rate_limit,
        use_local=args.use_local,
        local_data_dir=args.local_dir
    )
    
    # Fetch teams first
    fetcher.fetch_teams()
    
    # Fetch seasons
    fetcher.fetch_seasons(args.seasons)
    
    fetcher.close()
    print("\nData fetching complete!")


if __name__ == "__main__":
    main()
