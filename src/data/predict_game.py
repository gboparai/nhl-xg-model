"""
predict_game.py
===============
Utility for fetching a single NHL game from the API, parsing it into
the same format as xg_table.csv.gz, and running it through XGModel's
feature pipeline for live xG predictions.

Typical usage
-------------
    from src.data.predict_game import predict_xg
    from src.models.xg_model import XGModel

    model = XGModel()           # loads the saved pkl
    model.run()                 # or skip if pkl already loaded separately

    shots = predict_xg(2024020500, model)
    print(shots[["time_in_period", "period_number", "shotType", "xg"]].to_string())

Functions
---------
fetch_game_json(game_id)
    Fetch raw play-by-play JSON dict from the NHL API.

parse_game_to_shots(game_json, positions=None, shot_types=None)
    Parse the JSON dict into a shots DataFrame (same schema as xg_table.csv.gz).

predict_xg(game_id, xg_model, shot_types=None)
    End-to-end: fetch → parse → engineer features → predict.
    Returns shots DataFrame with an 'xg' column added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.export_shots import (
    _DEFAULT_SHOT_TYPES,
    _load_player_positions,
    _parse_game_data,
)
from src.database import get_db_path


# ── 1. Fetch ──────────────────────────────────────────────────────────────

def fetch_game_json(game_id: int | str) -> dict:
    """
    Fetch play-by-play JSON for *game_id* from the NHL API.

    Parameters
    ----------
    game_id : int | str
        NHL game ID, e.g. ``2024020500``.

    Returns
    -------
    dict
        Raw play-by-play JSON as returned by the NHL API.
    """
    try:
        from nhlpy import NHLClient
    except ImportError:
        raise ImportError("nhlpy is required. Run: pip install nhlpy")

    client = NHLClient()
    return client.game_center.play_by_play(int(game_id))


# ── 2. Parse ──────────────────────────────────────────────────────────────

def parse_game_to_shots(
    game_json: dict,
    positions: dict[int, str] | None = None,
    shot_types: frozenset | set | None = None,
) -> pd.DataFrame:
    """
    Parse a game JSON dict into a shots DataFrame with the same schema
    as ``xg_table.csv.gz`` (all ``prev_event_*`` columns included).

    Parameters
    ----------
    game_json : dict
        Raw play-by-play dict from :func:`fetch_game_json` or a local JSON file.
    positions : dict[int, str] | None
        ``{player_id: position}`` mapping used to set ``is_forward``.
        Loaded from the local SQLite DB if not provided.
    shot_types : frozenset | set | None
        Which NHL event types to include. Defaults to
        ``shot-on-goal + blocked-shot + goal``.

    Returns
    -------
    pd.DataFrame
        One row per qualifying shot event.
    """
    if positions is None:
        positions = _load_player_positions(get_db_path())
    if shot_types is None:
        shot_types = _DEFAULT_SHOT_TYPES

    rows = _parse_game_data(game_json, positions, shot_types)
    return pd.DataFrame(rows)


# ── 3. Full pipeline ──────────────────────────────────────────────────────

def predict_xg(
    game_id: int | str,
    xg_model,
    shot_types: frozenset | set | None = None,
) -> pd.DataFrame:
    """
    Fetch a game, run it through XGModel's feature pipeline, and return
    a shots DataFrame with an ``xg`` column added.

    Parameters
    ----------
    game_id : int | str
        NHL game ID (e.g. ``2024020500``).
    xg_model : XGModel
        A trained XGModel instance. Must have ``_model`` set — either by
        calling ``.run()`` first, or by loading the saved pkl.
    shot_types : frozenset | None
        Which event types to include. Defaults to SOG + blocked + goal.

    Returns
    -------
    pd.DataFrame
        Shot rows with engineered features and an ``xg`` column.

    Notes
    -----
    * Shift features (``shooter_toi``, ``opp_shift_*``) are skipped for
      live single-game fetches unless ``data/shots/shift_lookup.parquet``
      covers the requested game. The model handles missing shift data
      gracefully (fills with 0).
    * The feature set is automatically aligned to whatever the loaded pkl
      was trained on, so this works with both old 33-feature and new
      60+-feature models.
    """
    if xg_model._model is None:
        raise RuntimeError(
            "xg_model._model is None — call xg_model.run() first, "
            "or pass a model that has already been trained/loaded."
        )

    print(f"Fetching game {game_id} from NHL API ...")
    game_json = fetch_game_json(game_id)

    game_state = game_json.get("gameState", "unknown")
    game_date  = game_json.get("gameDate", "")
    home = (game_json.get("homeTeam") or {}).get("name", {}).get("default", "?")
    away = (game_json.get("awayTeam") or {}).get("name", {}).get("default", "?")
    print(f"  {away} @ {home}  |  {game_date}  |  state={game_state}")

    print("Parsing shot events ...")
    shots = parse_game_to_shots(game_json, shot_types=shot_types)
    if shots.empty:
        print("  No qualifying shots found.")
        return shots
    print(f"  {len(shots)} qualifying shots parsed")

    print("Engineering features ...")
    shots = xg_model._clean_coords(shots)
    shots = xg_model._add_prior_event_features(shots)
    shots = xg_model._add_shift_features(shots)
    X, _, _ = xg_model._build_feature_matrix(shots)

    # Align to the feature set the pkl was trained on
    try:
        pkl_features = list(xg_model._model["xgb"].feature_names_in_)
        X = X.reindex(columns=pkl_features, fill_value=0)
    except Exception:
        pass

    shots = shots.loc[X.index].copy()
    shots["xg"] = xg_model._model.predict_proba(X)[:, 1]

    total_xg = shots["xg"].sum()
    goals    = int(shots["shot_made"].sum())
    print(f"  xG total: {total_xg:.2f}  |  actual goals: {goals}")
    return shots
