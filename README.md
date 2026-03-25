# Advanced Hockey Analytics — NHL xG Model

An XGBoost expected-goals (xG) pipeline built on raw NHL play-by-play data.  
The model predicts the probability that any given shot attempt results in a goal, broken down by game situation (5v5, PP, PK, EN).

---

## Quick Start

```bash
pip install -r requirements.txt
```

### Full pipeline (one command)

```bash
# Download, build shot CSV, build shift lookup, and train — all in one go
python main.py --full-pipeline --download --seasons 20222023 20232024 20242025

# With GPU training:
python main.py --full-pipeline --download --seasons 20222023 20232024 20242025 --retrain --gpu

# Tag the experiment run (results saved as results/xg_model_<tag>_<timestamp>.txt):
python main.py --full-pipeline --download --seasons 20222023 20232024 20242025 --retrain --tag my-run
```

### Run individual steps

```bash
# Step 1 — Download raw NHL JSON and populate database
python main.py --download --seasons 20222023 20232024 20242025

# Step 2 — Build shot CSV from raw JSON
python main.py --export-shots --skip-fetch

# Step 3 — Build shift lookup parquet
python main.py --build-shifts --skip-fetch

# Step 4 — Evaluate with pre-trained model
python main.py --evaluate --skip-fetch

# Step 4 — Retrain from scratch, then evaluate
python main.py --evaluate --retrain --skip-fetch
```

### Re-evaluate with existing model (fast)

```bash
python main.py --evaluate --skip-fetch
```

### Hyperparameter search

```bash
# Step 1 — find best params (saves to models/optuna_best_params.json)
python main.py --tune --tune-trials 60 --tune-cv-folds 3 --skip-fetch

# Step 2 — retrain using those params
python main.py --evaluate --retrain --skip-fetch --tag optuna-tuned
```

### Live game prediction

```python
from src.data.predict_game import predict_xg
from src.models.xg_model import XGModel

model = XGModel()
model.run()   # loads the saved pkl

shots = predict_xg(2024020500, model)
print(shots[["game_id", "nhl_event_id", "time_in_period", "period_number", "shotType", "shooting_team_id", "xg"]])
```

---

## Project Structure

```
AdvancedHockeyAnalytics/
├── src/
│   ├── data/
│   │   ├── export_shots.py       # raw JSON → data/shots/xg_table.csv.gz
│   │   ├── predict_game.py       # fetch a live game → xG predictions
│   │   ├── download_raw_data.py  # NHL API downloader
│   │   ├── fetch_nhl_data.py     # API fetcher
│   │   ├── fetch_shift_data.py   # shift data fetcher
│   │   └── load_from_local.py    # load from cached JSON
│   ├── models/
│   │   ├── xg_model.py           # XGBoost pipeline (XGModel class)
│   │   ├── evaluate.py           # CLI: run model, print per-situation AUC
│   │   ├── tune_hyperparams.py   # Optuna TPE hyperparameter search
│   │   └── build_shift_lookup.py # builds data/shots/shift_lookup.parquet
│   └── database/
│       └── init_db.py            # SQLite schema + helpers
├── data/
│   ├── raw/                      # downloaded NHL JSON (gitignored)
│   │   ├── games/
│   │   ├── players/
│   │   ├── schedules/
│   │   └── shifts/
│   ├── shots/
│   │   ├── xg_table.csv.gz       # shot-event table (model input)
│   │   └── shift_lookup.parquet  # on-ice skater counts per shot
│   └── nhl_xg.db                 # SQLite database
├── models/
│   ├── xgb_combined_gpu_random.pkl  # trained XGBoost model
│   └── optuna_best_params.json      # saved Optuna params (auto-loaded on retrain)
├── plots/
│   ├── roc_curve.png
│   ├── brier_score.png           # calibration plot
│   └── feature_importance.png
├── results/
│   ├── baseline_results.txt      # original baseline scores
│   └── best_results.txt          # current best model scores
├── main.py                       # unified pipeline entry point
└── requirements.txt
```

---

## Features Engineered

**Geometry**

- Shot distance, angle, distance², angle², distance × angle
- Log distance, radial distance, distance bin
- Zone flags: `in_slot`, `home_plate`, `behind_net`

**Prior event**

- Δx, Δy, distance from last event, movement speed
- Time since last event, prior event type (one-hot)

**Game state**

- Score differential buckets (down 2+, down 1, tied, up 1, up 2+)
- Period, time fraction within period
- Situation code (PP / PK / 5v5 / EN)
- On-ice skater counts via shift lookup

**Shot type** (one-hot)

- Wrist, snap, slap, backhand, tip-in, deflected, wrap-around

---

## Data

| Source                                                                            | Description            |
| --------------------------------------------------------------------------------- | ---------------------- |
| NHL API (`/v1/gamecenter/{id}/play-by-play`)                                      | Raw game events        |
| NHL API (`/v1/roster/{team}/{season}`)                                            | Player positions       |
| NHL API (`https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<id>`) | On-ice skater tracking |

Seasons available in raw data: 2013-14 → 2024-25  
Model trained on: 2013-14 → 2021-22 · Tested on: 2022-23 → 2024-25

---

## Results

See [RESULTS.md](RESULTS.md) for full model performance, feature importance, and experiment log.

---

## Live Game Predictions

`src/data/predict_game.py` provides a self-contained utility that fetches any NHL game from the API, runs the same feature engineering pipeline used during training, and returns shot-level xG probabilities.

### Functions

| Function                         | Description                                               |
| -------------------------------- | --------------------------------------------------------- |
| `fetch_game_json(game_id)`       | Fetch raw play-by-play JSON from `api-web.nhle.com`       |
| `parse_game_to_shots(game_json)` | Parse JSON → DataFrame (same schema as `xg_table.csv.gz`) |
| `predict_xg(game_id, xg_model)`  | End-to-end: fetch → parse → engineer features → predict   |

### Example

```python
from src.data.predict_game import predict_xg
from src.models.xg_model import XGModel

# Load the saved model
model = XGModel()
model.run()

# Predict xG for a specific game
shots = predict_xg(2024020500, model)

# Results include all engineered features + xg column
print(shots[["time_in_period", "period_number", "type_desc_key",
             "shotType", "shot_distance_calc", "is_rebound", "xg"]])

# Team totals
print(shots.groupby("shooting_team_id")[["shot_made", "xg"]].sum())
```

### How the pipeline maps to training

```
fetch_game_json()         — NHL API (same endpoint as bulk download)
    ↓
parse_game_to_shots()     — same parsing logic as export_shots.py
    ↓
XGModel._clean_coords()   — distance, angle, zone flags
    ↓
XGModel._add_prior_event_features()  — rebound, rush, era flags, last-event dummies
    ↓
XGModel._add_shift_features()        — shooter TOI, opponent shift stats (if available)
    ↓
XGModel._build_feature_matrix()      — assemble full feature matrix
    ↓
model.predict_proba()     — xG probability per shot
```

> **Note on shift features**: `shooter_toi` and `opp_shift_*` will be 0 for live single-game fetches unless the game is already covered by `data/shots/shift_lookup.parquet`. The model handles missing shift data gracefully.

---

## Hyperparameter Tuning

`src/models/tune_hyperparams.py` runs an **Optuna TPE** search over XGBoost using the exact same feature pipeline as training, so tuned params transfer directly to `--evaluate --retrain`.

### Workflow

```bash
# Step 1 — search (saves best params to models/optuna_best_params.json)
python main.py --tune --tune-trials 60 --tune-cv-folds 3 --skip-fetch

# Step 2 — retrain using those params, then evaluate
python main.py --evaluate --retrain --skip-fetch --tag optuna-tuned
```

`XGModel._train()` automatically detects `models/optuna_best_params.json` and uses it in place of its internal random search. Delete the file to revert to random search.

### CLI options

| Flag              | Default | Description                   |
| ----------------- | ------- | ----------------------------- |
| `--tune-trials`   | 40      | Number of Optuna trials       |
| `--tune-cv-folds` | 3       | Stratified CV folds per trial |
| `--gpu`           | off     | Use `device=cuda` for XGBoost |
| `--seed`          | 42      | Random seed                   |

### Search space

| Parameter          | Range             |
| ------------------ | ----------------- |
| `max_depth`        | 2 – 10            |
| `learning_rate`    | 0.005 – 0.3 (log) |
| `n_estimators`     | 50 – 600          |
| `subsample`        | 0.3 – 1.0         |
| `colsample_bytree` | 0.3 – 1.0         |
| `min_child_weight` | 1 – 50            |
| `reg_alpha`        | 0.0 – 1.0         |
| `reg_lambda`       | 0.5 – 3.0         |
| `gamma`            | 0.0 – 5.0         |

### How it works

```
XGModel._load_data()
    ↓
_clean_coords() → _add_prior_event_features() → _add_shift_features()
    ↓
_build_feature_matrix()   — identical to training
    ↓
Optuna TPE (n_trials × StratifiedKFold CV)   — optimises neg-log-loss
    ↓
models/optuna_best_params.json
    ↓
--evaluate --retrain   — full retrain with tuned params
```

> **Note**: Tuning runs on the training seasons only (2013-14 → 2021-22). The holdout seasons (2022-23 → 2024-25) are never seen during the search.

---
