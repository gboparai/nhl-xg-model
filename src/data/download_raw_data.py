"""
Download and store raw NHL API data locally.
This script fetches game and player data from NHL API and saves as JSON files.
Run this once to build a local cache, then use load_from_local.py to populate the database.
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from nhlpy import NHLClient
from tqdm import tqdm
import requests


class RawDataDownloader:
    """Downloads and stores raw NHL data as JSON files."""
    
    def __init__(self, data_dir='data/raw', rate_limit_delay=0.5):
        """
        Initialize the downloader.
        
        Args:
            data_dir: Directory to store JSON files
            rate_limit_delay: Delay in seconds between API requests
        """
        self.client = NHLClient()
        self.rate_limit_delay = rate_limit_delay
        self.data_dir = Path(data_dir)
        
        # Create directory structure
        self.games_dir = self.data_dir / 'games'
        self.players_dir = self.data_dir / 'players'
        self.schedules_dir = self.data_dir / 'schedules'
        self.teams_dir = self.data_dir / 'teams'
        
        for dir_path in [self.games_dir, self.players_dir, self.schedules_dir, self.teams_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_teams(self):
        """Download and store team data."""
        print("Downloading NHL teams...")
        
        try:
            schedule = self.client.schedule.daily_schedule()
            teams_data = {}
            
            if schedule and 'gameWeek' in schedule:
                for day in schedule['gameWeek']:
                    if 'games' in day:
                        for game in day['games']:
                            for team_key in ['homeTeam', 'awayTeam']:
                                if team_key in game:
                                    team = game[team_key]
                                    team_id = team.get('id')
                                    if team_id:
                                        teams_data[team_id] = team
            
            # Save teams data
            teams_file = self.teams_dir / 'teams.json'
            with open(teams_file, 'w', encoding='utf-8') as f:
                json.dump(teams_data, f, indent=2)
            
            print(f"✅ Saved {len(teams_data)} teams to {teams_file}")
            
        except Exception as e:
            print(f"❌ Error downloading teams: {e}")
    
    def download_season_schedule(self, season):
        """
        Download schedule for a season.
        
        Args:
            season: Season in format "20232024"
        
        Returns:
            List of game IDs
        """
        # All NHL team abbreviations (current + historical 2010-2026)
        teams = [
            'ANA', 'ARI', 'ATL', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY',
            'CHI', 'COL', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN',
            'MTL', 'NJD', 'NSH', 'NYI', 'NYR', 'OTT', 'PHI', 'PHX',
            'PIT', 'SEA', 'SJS', 'STL', 'TBL', 'TOR', 'UTA', 'VAN',
            'VGK', 'WPG', 'WSH',
        ]
        
        game_ids = set()
        season_schedules = {}
        
        print(f"Downloading season {season} schedule from all teams...")
        for team in tqdm(teams, desc="Teams"):
            try:
                schedule = self.client.schedule.team_season_schedule(team_abbr=team, season=season)
                
                if schedule and 'games' in schedule:
                    season_schedules[team] = schedule
                    for game in schedule['games']:
                        game_ids.add(game['id'])
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                # Silently continue for teams that don't exist in this season
                continue
        
        # Save schedule data
        schedule_file = self.schedules_dir / f'schedule_{season}.json'
        with open(schedule_file, 'w', encoding='utf-8') as f:
            json.dump(season_schedules, f, indent=2)
        
        print(f"✅ Saved schedule for {season} ({len(game_ids)} games) to {schedule_file}")
        
        return list(game_ids)
    
    def download_game_data(self, game_id, force=False):
        """
        Download and store play-by-play data for a single game.
        
        Args:
            game_id: NHL game ID
            force: If True, re-download even if file exists
        
        Returns:
            True if successful, False otherwise
        """
        # Organize games by season (first 4 digits of game ID)
        season = str(game_id)[:4]
        season_dir = self.games_dir / season
        season_dir.mkdir(exist_ok=True)
        
        game_file = season_dir / f'game_{game_id}.json'
        
        # Skip if already exists and not forcing
        if game_file.exists() and not force:
            return True
        
        try:
            # Fetch play-by-play data
            pbp = self.client.game_center.play_by_play(game_id)
            
            if not pbp:
                return False
            
            # Save game data
            with open(game_file, 'w', encoding='utf-8') as f:
                json.dump(pbp, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading game {game_id}: {e}")
            return False
    
    def download_seasons(self, seasons, force=False):
        """
        Download data for multiple seasons.
        
        Args:
            seasons: List of seasons in format ["20232024", "20242025"]
            force: If True, re-download existing files
        """
        for season in seasons:
            print(f"\n{'='*80}")
            print(f"DOWNLOADING SEASON {season}")
            print(f"{'='*80}")
            
            # Get all games for the season
            game_ids = self.download_season_schedule(season)
            print(f"Found {len(game_ids)} games for season {season}")
            
            # Download each game
            success_count = 0
            error_count = 0
            skipped_count = 0
            
            for game_id in tqdm(game_ids, desc=f"Downloading games {season}"):
                season_dir = self.games_dir / str(game_id)[:4]
                game_file = season_dir / f'game_{game_id}.json'
                
                if game_file.exists() and not force:
                    skipped_count += 1
                    continue
                
                success = self.download_game_data(game_id, force=force)
                if success:
                    success_count += 1
                else:
                    error_count += 1
                
                time.sleep(self.rate_limit_delay)
            
            print(f"\n✅ Downloaded: {success_count} games")
            print(f"⏭️  Skipped (already exists): {skipped_count} games")
            print(f"❌ Errors: {error_count} games")
    
    def download_player_data(self, player_ids, force=False):
        """
        Download player details from NHL API.
        
        Args:
            player_ids: List of player IDs to fetch
            force: If True, re-download existing files
        
        Returns:
            Tuple of (success_count, error_count, skipped_count)
        """
        print(f"\nDownloading player data for {len(player_ids)} players...")
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for player_id in tqdm(player_ids, desc="Downloading player details"):
            player_file = self.players_dir / f'player_{player_id}.json'
            
            # Skip if already exists and not forcing
            if player_file.exists() and not force:
                skipped_count += 1
                continue
            
            try:
                # Use the GameCenter API endpoint
                url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    player_data = response.json()
                    
                    # Save player data
                    with open(player_file, 'w', encoding='utf-8') as f:
                        json.dump(player_data, f, indent=2)
                    
                    success_count += 1
                else:
                    error_count += 1
                
                # Rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                error_count += 1
                continue
        
        print(f"\n✅ Downloaded: {success_count} players")
        print(f"⏭️  Skipped (already exists): {skipped_count} players")
        print(f"❌ Errors: {error_count} players")
        
        return success_count, error_count, skipped_count
    
    def extract_player_ids_from_games(self):
        """
        Extract unique player IDs from all downloaded game files.
        
        Returns:
            Set of player IDs
        """
        player_ids = set()
        
        print("Extracting player IDs from downloaded games...")
        game_files = list(self.games_dir.rglob('game_*.json'))
        
        for game_file in tqdm(game_files, desc="Scanning games"):
            try:
                with open(game_file, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                
                # Extract player IDs from plays
                plays = game_data.get('plays', [])
                for play in plays:
                    details = play.get('details', {})
                    
                    # Get shooter ID
                    shooter_id = details.get('scoringPlayerId') or details.get('shootingPlayerId')
                    if shooter_id:
                        player_ids.add(shooter_id)
                    
                    # Get goalie ID
                    goalie_id = details.get('goalieInNetId')
                    if goalie_id:
                        player_ids.add(goalie_id)
                        
            except Exception as e:
                continue
        
        print(f"Found {len(player_ids)} unique player IDs")
        return player_ids
    
    def download_all_player_data(self, force=False):
        """
        Download player data for all players found in game files.
        
        Args:
            force: If True, re-download existing files
        """
        player_ids = self.extract_player_ids_from_games()
        self.download_player_data(list(player_ids), force=force)


def main():
    parser = argparse.ArgumentParser(description='Download raw NHL data and save as JSON files')
    parser.add_argument('--seasons', nargs='+', 
                       default=[f"{year}{year+1}" for year in range(2010, 2026)],
                       help='Seasons to download (e.g., 20232024 20242025)')
    parser.add_argument('--data-dir', default='data/raw',
                       help='Directory to store JSON files')
    parser.add_argument('--rate-limit', type=float, default=0.5,
                       help='Delay between API requests in seconds')
    parser.add_argument('--force', action='store_true',
                       help='Re-download existing files')
    parser.add_argument('--skip-games', action='store_true',
                       help='Skip game downloads (only fetch players)')
    parser.add_argument('--skip-players', action='store_true',
                       help='Skip player downloads')
    
    args = parser.parse_args()
    
    downloader = RawDataDownloader(data_dir=args.data_dir, rate_limit_delay=args.rate_limit)
    
    print("="*80)
    print("NHL RAW DATA DOWNLOADER")
    print("="*80)
    print(f"Data directory: {downloader.data_dir.absolute()}")
    print(f"Seasons: {args.seasons}")
    print(f"Force re-download: {args.force}")
    print("="*80)
    
    # Download teams
    downloader.download_teams()
    
    # Download games
    if not args.skip_games:
        downloader.download_seasons(args.seasons, force=args.force)
    
    # Download players
    if not args.skip_players:
        downloader.download_all_player_data(force=args.force)
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"Data saved to: {downloader.data_dir.absolute()}")
    print("\nNext step: Run the database loader to populate the database from these files")


if __name__ == "__main__":
    main()
