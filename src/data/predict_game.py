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
import requests

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.export_shots import (
    _DEFAULT_SHOT_TYPES,
    _load_player_positions,
    _parse_game_data,
)
from src.database import get_db_path


# ── 0. Live shift helper ─────────────────────────────────────────────────

REGULAR_SHIFT_TYPECODE = 517
_SHIFT_API_URL = "https://api.nhle.com/stats/rest/en/shiftcharts"


def _fetch_live_shifts(game_id: int | str, positions: dict[int, str]) -> pd.DataFrame:
    """
    Fetch shift-chart data for *game_id* from the NHL stats API and return
    a DataFrame matching the ``shift_lookup.parquet`` schema::

        game_id  period  player_id  team_id  start_s  end_s  is_goalie

    Parameters
    ----------
    game_id : int | str
        NHL game ID.
    positions : dict[int, str]
        ``{player_id: position}`` mapping used to flag goalies.

    Returns
    -------
    pd.DataFrame
        Empty DataFrame (correct schema) if the API call fails or returns
        no usable shifts.
    """
    _EMPTY = pd.DataFrame(
        columns=["game_id", "period", "player_id", "team_id", "start_s", "end_s", "is_goalie"]
    )

    def _mmss_to_s(v: str) -> int:
        try:
            m, s = str(v).strip().split(":")
            return int(m) * 60 + int(s)
        except Exception:
            return -1

    try:
        resp = requests.get(
            _SHIFT_API_URL,
            params={"cayenneExp": f"gameId={game_id}"},
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json().get("data", [])
    except Exception as exc:
        print(f"  [shift] Failed to fetch live shifts for {game_id}: {exc}")
        return _EMPTY

    rows = []
    for shift in raw:
        if shift.get("typeCode") != REGULAR_SHIFT_TYPECODE:
            continue
        start = _mmss_to_s(shift.get("startTime", ""))
        end   = _mmss_to_s(shift.get("endTime",   ""))
        if start < 0 or end < 0 or end < start:
            continue
        pid = shift.get("playerId")
        rows.append({
            "game_id":   int(game_id),
            "period":    shift.get("period"),
            "player_id": pid,
            "team_id":   shift.get("teamId"),
            "start_s":   start,
            "end_s":     end,
            "is_goalie": (positions.get(int(pid), "") == "G") if pid is not None else False,
        })

    if not rows:
        return _EMPTY

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
    return df


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
    game_json: dict | None = None,
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
    game_json : dict | None
        Pre-fetched play-by-play JSON dict.  When provided the API fetch
        step is skipped (avoids a duplicate round-trip when the caller has
        already fetched the data).

    Returns
    -------
    pd.DataFrame
        Shot rows with engineered features and an ``xg`` column.

    Notes
    -----
    * Shift features are fetched live from the NHL stats API for the
      requested game so that ``shooter_toi`` / ``opp_shift_*`` are
      available even for single-game predictions.
    * The feature set is automatically aligned to whatever the loaded pkl
      was trained on, so this works with both old 33-feature and new
      60+-feature models.
    """
    if xg_model._model is None:
        raise RuntimeError(
            "xg_model._model is None — call xg_model.run() first, "
            "or pass a model that has already been trained/loaded."
        )

    if game_json is None:
        print(f"Fetching game {game_id} from NHL API ...")
        game_json = fetch_game_json(game_id)
    else:
        print(f"Using provided play-by-play JSON for game {game_id} ...")

    game_state = game_json.get("gameState", "unknown")
    game_date  = game_json.get("gameDate", "")
    home = (game_json.get("homeTeam") or {}).get("name", {}).get("default", "?")
    away = (game_json.get("awayTeam") or {}).get("name", {}).get("default", "?")
    print(f"  {away} @ {home}  |  {game_date}  |  state={game_state}")

    # Load player positions once — reused for shot parsing and shift lookup
    positions = _load_player_positions(get_db_path())

    print("Parsing shot events ...")
    shots = parse_game_to_shots(game_json, positions=positions, shot_types=shot_types)
    if shots.empty:
        print("  No qualifying shots found.")
        return shots
    print(f"  {len(shots)} qualifying shots parsed")

    print("Fetching live shift data ...")
    live_shifts = _fetch_live_shifts(game_id, positions)
    if live_shifts.empty:
        print("  No shift data available — shift features will default to 0")
        live_shifts = None
    else:
        print(f"  {len(live_shifts):,} shift rows loaded")

    print("Engineering features ...")
    shots = xg_model._clean_coords(shots)
    shots = xg_model._add_prior_event_features(shots)
    shots = xg_model._add_shift_features(shots, live_shifts=live_shifts)
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
