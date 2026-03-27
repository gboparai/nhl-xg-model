"""
Quick smoke-test for predict_game.py using game 2025021145.

Run from the project root:
    python test_predict_game.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.xg_model import XGModel, _PRETRAINED_MODEL
from src.data.predict_game import predict_xg

GAME_ID = 2025021145

# ── Load the pretrained pkl directly (skips full data pipeline) ───────────
print("=" * 60)
print(f"Loading pre-trained model from {_PRETRAINED_MODEL} ...")
assert _PRETRAINED_MODEL.exists(), f"pkl not found: {_PRETRAINED_MODEL}"

model = XGModel()
with open(_PRETRAINED_MODEL, "rb") as fh:
    model._model = pickle.load(fh)

print(f"Model loaded: {type(model._model)}")
assert model._model is not None, "Model failed to load"

# ── Run prediction ────────────────────────────────────────────────────────
print("=" * 60)
shots = predict_xg(GAME_ID, model)

assert not shots.empty, "No shots returned — check game ID or API connection"
assert "xg" in shots.columns, "'xg' column missing from output"
assert shots["xg"].between(0, 1).all(), "xG values outside [0, 1]"

# ── Print summary ─────────────────────────────────────────────────────────
print("=" * 60)
print(f"Game {GAME_ID} — shot-level xG preview (first 20 rows):\n")

display_cols = [c for c in [
    "period_number", "time_in_period", "shooting_team_id", "shooter_id",
    "shotType", "shot_distance_calc", "shot_angle_calc",
    "shot_made", "xg"
] if c in shots.columns]

print(shots[display_cols].head(20).to_string(index=False))

# Team summary — group by whatever team column is present
team_col = next((c for c in ["shooting_team_id", "team", "teamId"] if c in shots.columns), None)
if team_col:
    print("\n" + "=" * 60)
    by_team = shots.groupby(team_col)[["xg", "shot_made"]].sum().rename(
        columns={"xg": "xG", "shot_made": "Goals"}
    )
    by_team["xG"] = by_team["xG"].round(3)
    print("Team summary:")
    print(by_team.to_string())

print("\nAll assertions passed.")
