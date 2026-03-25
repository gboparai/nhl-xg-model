"""
Fetch and populate player details (name, position, handedness) using NHL API.
Uses the GameCenter API endpoint: https://api-web.nhle.com/v1/player/{playerId}/landing
Can also load from local JSON files.
"""

import requests
import argparse
from tqdm import tqdm
from pathlib import Path
from src.database import get_connection
from src.data.load_from_local import LocalDataLoader
import time

def fetch_player_details(use_local=False, local_dir='data/raw'):
    """Fetch player details from NHL API and update database."""
    
    if use_local:
        print("Loading player data from local files...")
        loader = LocalDataLoader(data_dir=local_dir)
        loader.load_all_players()
        loader.close()
        return
    
    print("Fetching player data from API...")
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get all unique player IDs from events
    cursor.execute("""
        SELECT DISTINCT shooting_player_id 
        FROM events 
        WHERE shooting_player_id IS NOT NULL
    """)
    player_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Found {len(player_ids)} unique players to fetch")
    
    success_count = 0
    error_count = 0
    
    for player_id in tqdm(player_ids, desc="Fetching player details"):
        try:
            # Use the GameCenter API endpoint
            url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                player_data = response.json()
                
                # Extract player details
                full_name = None
                position = None
                shoots_catches = None
                
                # Get name
                first_name = player_data.get('firstName', {})
                last_name = player_data.get('lastName', {})
                
                if isinstance(first_name, dict):
                    first_name = first_name.get('default', '')
                if isinstance(last_name, dict):
                    last_name = last_name.get('default', '')
                
                full_name = f"{first_name} {last_name}".strip()
                
                # Get position
                position = player_data.get('position')
                
                # Get shoots/catches (handedness) - L or R
                shoots_catches = player_data.get('shootsCatches')
                # Strip surrounding single-quotes if present (e.g. "'L'" -> "L")
                if isinstance(shoots_catches, str):
                    shoots_catches = shoots_catches.strip("'")
                    if shoots_catches not in ('L', 'R'):
                        shoots_catches = None
                
                # Update database if we got data
                if full_name or position or shoots_catches:
                    cursor.execute("""
                        UPDATE players 
                        SET full_name = COALESCE(?, full_name),
                            position = COALESCE(?, position),
                            shoots_catches = COALESCE(?, shoots_catches)
                        WHERE player_id = ?
                    """, (full_name, position, shoots_catches, player_id))
                    
                    success_count += 1
                else:
                    error_count += 1
            else:
                error_count += 1
            
            # Rate limiting - be nice to the API
            time.sleep(0.05)
            
        except Exception as e:
            error_count += 1
            continue
        
        # Commit every 100 players
        if (success_count + error_count) % 100 == 0:
            conn.commit()
    
    # Final commit
    conn.commit()
    
    print(f"\n✅ Successfully fetched: {success_count} players")
    print(f"❌ Errors: {error_count} players")
    
    # Show sample of updated players
    cursor.execute("""
        SELECT player_id, full_name, position, shoots_catches 
        FROM players 
        WHERE shoots_catches IS NOT NULL
        LIMIT 10
    """)
    
    print("\nSample of updated players:")
    for row in cursor.fetchall():
        print(f"  {row[1]} - {row[2]} (Shoots: {row[3]})")
    
    # Show statistics
    cursor.execute("SELECT COUNT(*) FROM players WHERE shoots_catches IS NOT NULL")
    with_handedness = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM players")
    total_players = cursor.fetchone()[0]
    
    print(f"\nPlayers with handedness data: {with_handedness}/{total_players} ({with_handedness/total_players*100:.1f}%)")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch player details from NHL API or local files')
    parser.add_argument('--use-local', action='store_true',
                       help='Load from local JSON files instead of API')
    parser.add_argument('--local-dir', default='data/raw',
                       help='Directory containing local JSON files')
    
    args = parser.parse_args()
    
    fetch_player_details(use_local=args.use_local, local_dir=args.local_dir)
