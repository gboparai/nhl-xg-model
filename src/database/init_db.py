"""
Database schema initialization for NHL xG pipeline.
Creates SQLite database with tables for games, players, teams, events, features, and predictions.
"""

import sqlite3
import os
from pathlib import Path


def get_db_path():
    """Get the path to the SQLite database."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "nhl_xg.db"


def init_database():
    """Initialize the database with all required tables."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Games table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY,
            season TEXT NOT NULL,
            game_type INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            home_score INTEGER,
            away_score INTEGER,
            game_state TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Players table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            position TEXT,
            shoots_catches TEXT
        )
    """)
    
    # Teams table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_abbr TEXT NOT NULL,
            team_name TEXT NOT NULL,
            franchise_id INTEGER
        )
    """)
    
    # Events table (play-by-play shots)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            event_idx INTEGER NOT NULL,
            period INTEGER NOT NULL,
            period_time TEXT,
            time_remaining INTEGER,
            event_type TEXT NOT NULL,
            x_coord REAL,
            y_coord REAL,
            zone_code TEXT,
            shot_type TEXT,
            shooting_player_id INTEGER,
            goalie_id INTEGER,
            shooting_team_id INTEGER,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            is_goal INTEGER DEFAULT 0,
            is_empty_net INTEGER DEFAULT 0,
            miss_reason TEXT,
            FOREIGN KEY (game_id) REFERENCES games(game_id),
            FOREIGN KEY (shooting_player_id) REFERENCES players(player_id),
            FOREIGN KEY (goalie_id) REFERENCES players(player_id),
            UNIQUE (game_id, event_idx)
        )
    """)
    
    # Shot features table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shot_features (
            event_id INTEGER PRIMARY KEY,
            distance REAL,
            angle REAL,
            distance_angle_interaction REAL,
            shot_type TEXT,
            time_since_last_event REAL,
            distance_from_last_event REAL,
            last_event_type TEXT,
            is_rebound INTEGER DEFAULT 0,
            is_rush_shot INTEGER DEFAULT 0,
            score_differential INTEGER,
            period INTEGER,
            time_remaining INTEGER,
            is_home_team INTEGER,
            is_empty_net INTEGER DEFAULT 0,
            is_behind_net INTEGER DEFAULT 0,
            min_dist_to_post REAL,
            dist_to_left_post REAL,
            dist_to_right_post REAL,
            shots_last_10s INTEGER DEFAULT 0,
            shots_last_20s INTEGER DEFAULT 0,
            shots_last_40s INTEGER DEFAULT 0,
            blocks_last_10s INTEGER DEFAULT 0,
            blocks_last_20s INTEGER DEFAULT 0,
            blocks_last_40s INTEGER DEFAULT 0,
            missed_last_10s INTEGER DEFAULT 0,
            missed_last_20s INTEGER DEFAULT 0,
            missed_last_40s INTEGER DEFAULT 0,
            giveaways_last_10s INTEGER DEFAULT 0,
            giveaways_last_20s INTEGER DEFAULT 0,
            giveaways_last_40s INTEGER DEFAULT 0,
            takeaways_last_10s INTEGER DEFAULT 0,
            takeaways_last_20s INTEGER DEFAULT 0,
            takeaways_last_40s INTEGER DEFAULT 0,
            hits_last_10s INTEGER DEFAULT 0,
            hits_last_20s INTEGER DEFAULT 0,
            hits_last_40s INTEGER DEFAULT 0,
            faceoffs_last_10s INTEGER DEFAULT 0,
            faceoffs_last_20s INTEGER DEFAULT 0,
            faceoffs_last_40s INTEGER DEFAULT 0,
            is_off_wing INTEGER DEFAULT 0,
            crossed_royal_road INTEGER DEFAULT 0,
            angle_bucket TEXT,
            distance_bucket TEXT,
            rink_zone TEXT,
            shooter_toi_seconds REAL,
            avg_defender_toi_seconds REAL,
            max_defender_toi_seconds REAL,
            avg_shooting_team_toi_seconds REAL,
            strength_state_v2 TEXT,
            distance_sq REAL,
            angle_sq REAL,
            abs_y_coord REAL,
            rebound_distance REAL,
            prev_event_same_team INTEGER DEFAULT 0,
            shot_angle_from_center REAL,
            rebound_x_rush INTEGER DEFAULT 0,
            shots_last_3s INTEGER DEFAULT 0,
            blocked_last_3s INTEGER DEFAULT 0,
            missed_last_3s INTEGER DEFAULT 0,
            shots_this_period INTEGER DEFAULT 0,
            period_segment TEXT,
            defender_toi_variance REAL,
            shooter_position TEXT,
            score_state_urgency REAL,
            is_tied_late INTEGER DEFAULT 0,
            signed_angle REAL,
            shot_type_x_rebound TEXT,
            shot_type_x_zone TEXT,
            rebound_angle_change REAL DEFAULT 0.0,
            is_pre_pull_goalie INTEGER DEFAULT 0,
            puck_speed REAL,
            goalie_shots_faced INTEGER,
            velocity_x REAL,
            velocity_y REAL,
            is_stretch_rush INTEGER DEFAULT 0,
            royal_road_speed REAL,
            time_since_last_faceoff REAL,
            total_shots_last_10s INTEGER DEFAULT 0,
            total_shots_last_20s INTEGER DEFAULT 0,
            total_shots_last_40s INTEGER DEFAULT 0,
            shooting_team_b2b INTEGER DEFAULT 0,
            defending_team_b2b INTEGER DEFAULT 0,
            preceded_by_short_miss INTEGER DEFAULT 0,
            min_defender_toi_seconds REAL,
            last_event_miss_reason TEXT,
            home_plate INTEGER DEFAULT 0,
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        )
    """)
    
    # Player shifts table (ice-time per shift per player per game)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS player_shifts (
            shift_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            team_id INTEGER,
            period INTEGER NOT NULL,
            shift_number INTEGER,
            start_time TEXT,
            end_time TEXT,
            duration TEXT,
            start_seconds INTEGER,
            end_seconds INTEGER,
            duration_seconds INTEGER,
            FOREIGN KEY (game_id) REFERENCES games(game_id),
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            UNIQUE (game_id, player_id, period, shift_number)
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_shifts_game ON player_shifts(game_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_shifts_player ON player_shifts(player_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_shifts_game_player ON player_shifts(game_id, player_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_shifts_game_period ON player_shifts(game_id, period, start_seconds, end_seconds)"
    )

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            xg_probability REAL NOT NULL,
            is_high_danger INTEGER DEFAULT 0,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (event_id) REFERENCES events(event_id),
            UNIQUE(event_id, model_name, model_version)
        )
    """)
    
    # Model metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            dataset_split TEXT NOT NULL,
            evaluation_date TEXT NOT NULL
        )
    """)
    
    # Create indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_game ON events(game_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_shooter ON events(shooting_player_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_event ON predictions(event_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name)")
    
    # Create views for xG aggregations
    
    # View: xG by game and team
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS xg_by_game AS
        SELECT 
            g.game_id,
            g.game_date,
            g.season,
            g.game_type,
            e.shooting_team_id,
            t.team_abbr,
            t.team_name,
            p.model_name,
            COUNT(*) as total_shots,
            SUM(e.is_goal) as goals,
            SUM(p.xg_probability) as total_xg,
            SUM(p.is_high_danger) as high_danger_shots,
            AVG(p.xg_probability) as avg_xg_per_shot,
            SUM(p.xg_probability) - SUM(e.is_goal) as xg_difference
        FROM events e
        JOIN predictions p ON e.event_id = p.event_id
        JOIN games g ON e.game_id = g.game_id
        LEFT JOIN teams t ON e.shooting_team_id = t.team_id
        GROUP BY g.game_id, e.shooting_team_id, p.model_name
    """)
    
    # View: xG by period
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS xg_by_period AS
        SELECT 
            g.game_id,
            g.game_date,
            g.season,
            e.period,
            e.shooting_team_id,
            t.team_abbr,
            t.team_name,
            p.model_name,
            COUNT(*) as total_shots,
            SUM(e.is_goal) as goals,
            SUM(p.xg_probability) as total_xg,
            SUM(p.is_high_danger) as high_danger_shots,
            AVG(p.xg_probability) as avg_xg_per_shot
        FROM events e
        JOIN predictions p ON e.event_id = p.event_id
        JOIN games g ON e.game_id = g.game_id
        LEFT JOIN teams t ON e.shooting_team_id = t.team_id
        GROUP BY g.game_id, e.period, e.shooting_team_id, p.model_name
    """)
    
    # View: xG by team and season
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS xg_by_team_season AS
        SELECT 
            g.season,
            e.shooting_team_id,
            t.team_abbr,
            t.team_name,
            p.model_name,
            COUNT(DISTINCT g.game_id) as games_played,
            COUNT(*) as total_shots,
            SUM(e.is_goal) as goals,
            SUM(p.xg_probability) as total_xg,
            SUM(p.is_high_danger) as high_danger_shots,
            AVG(p.xg_probability) as avg_xg_per_shot,
            SUM(p.xg_probability) / COUNT(DISTINCT g.game_id) as xg_per_game,
            SUM(p.is_high_danger) / COUNT(DISTINCT g.game_id) as hd_shots_per_game,
            SUM(e.is_goal) / COUNT(DISTINCT g.game_id) as goals_per_game,
            CAST(SUM(e.is_goal) AS REAL) / COUNT(*) as shooting_percentage,
            CAST(SUM(e.is_goal) AS REAL) / SUM(p.xg_probability) as goals_vs_xg_ratio
        FROM events e
        JOIN predictions p ON e.event_id = p.event_id
        JOIN games g ON e.game_id = g.game_id
        LEFT JOIN teams t ON e.shooting_team_id = t.team_id
        GROUP BY g.season, e.shooting_team_id, p.model_name
    """)
    
    # View: xG by player (for goalies and shooters)
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS xg_by_player AS
        SELECT 
            g.season,
            e.shooting_player_id,
            ps.full_name as shooter_name,
            ps.position,
            p.model_name,
            COUNT(DISTINCT g.game_id) as games_played,
            COUNT(*) as total_shots,
            SUM(e.is_goal) as goals,
            SUM(p.xg_probability) as total_xg,
            SUM(p.is_high_danger) as high_danger_shots,
            AVG(p.xg_probability) as avg_xg_per_shot,
            CAST(SUM(e.is_goal) AS REAL) / COUNT(*) as shooting_percentage,
            SUM(e.is_goal) - SUM(p.xg_probability) as goals_above_xg
        FROM events e
        JOIN predictions p ON e.event_id = p.event_id
        JOIN games g ON e.game_id = g.game_id
        LEFT JOIN players ps ON e.shooting_player_id = ps.player_id
        WHERE e.shooting_player_id IS NOT NULL
        GROUP BY g.season, e.shooting_player_id, p.model_name
    """)
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized successfully at: {db_path}")
    print("Created views: xg_by_game, xg_by_period, xg_by_team_season, xg_by_player")
    return db_path


def get_connection():
    """Get a connection to the database."""
    db_path = get_db_path()
    return sqlite3.connect(db_path)


if __name__ == "__main__":
    init_database()
