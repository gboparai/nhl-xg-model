"""
tune_hyperparams.py
===================
Runs an Optuna TPE hyperparameter search over XGBoost using the
EXACT same feature matrix as xg_model.py (shift data, era flags,
rebound flags, etc.) by delegating to XGModel's private pipeline.

Best params are saved to models/optuna_best_params.json.
XGModel._train() automatically picks up that file when you run
--evaluate --retrain, replacing its internal random search with
the Optuna-tuned params.

Workflow:
    # 1. Find best hyperparameters (60 trials, 3-fold CV)
    python main.py --tune --tune-trials 60 --tune-cv-folds 3 --skip-fetch

    # 2. Retrain and evaluate using those params
    python main.py --evaluate --retrain --skip-fetch --tag optuna-60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ──────────────────────────────────────────────────────────────────
HERE             = Path(__file__).parent           # src/models/
ROOT             = HERE.parent.parent              # project root
CSV_PATH         = ROOT / "data" / "shots" / "xg_table.csv.gz"
BEST_PARAMS_PATH = ROOT / "models" / "optuna_best_params.json"

HOLDOUT_SEASONS = [20222023, 20232024, 20242025]


# ── feature matrix via XGModel's pipeline ─────────────────────────────────

def _get_feature_matrix(csv_path: Path, holdout_seasons: list[int]):
    """
    Build X_train, y_train, X_hold, y_hold using XGModel's exact pipeline
    (shift data, era flags, rebound flags, etc.) so tuned params transfer
    directly to --evaluate --retrain.
    """
    import pandas as pd
    from src.models.xg_model import XGModel

    model = XGModel(use_pretrained=False, verbose=True, data_path=str(csv_path))

    shots = model._load_data()
    shots = model._clean_coords(shots)
    shots = model._add_prior_event_features(shots)
    shots = model._add_shift_features(shots)
    X, y, _ = model._build_feature_matrix(shots)

    # Replace any inf that arise from division-by-zero
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    is_holdout = shots["season"].isin(holdout_seasons)
    X_train, y_train = X[~is_holdout], y[~is_holdout]
    X_hold,  y_hold  = X[is_holdout],  y[is_holdout]
    X_hold = X_hold.reindex(columns=X_train.columns, fill_value=0)

    print(f"  Train: {len(X_train):,}  |  Holdout: {len(X_hold):,}")
    print(f"  Features: {X_train.shape[1]}")
    return X_train, y_train, X_hold, y_hold


# ── Optuna TPE search ──────────────────────────────────────────────────────

def optuna_search_xgb(
    X_train,
    y_train,
    n_trials: int = 40,
    random_state: int = 42,
    use_gpu: bool = False,
    cv_folds: int = 3,
) -> dict:
    """
    Run Optuna TPE search. Returns the best params dict (XGBClassifier kwargs).
    Does NOT fit a final model — the caller saves params to JSON and the
    full retrain happens via --evaluate --retrain.
    """
    xgb_base = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
    )
    if use_gpu:
        xgb_base["device"] = "cuda"

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **xgb_base,
            "max_depth":        trial.suggest_int("max_depth", 2, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 50, 600),
            "subsample":        trial.suggest_float("subsample", 0.3, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        }
        pipeline = ImbPipeline([("xgb", XGBClassifier(**params))])
        scores = cross_val_score(
            pipeline, X_train, y_train,
            scoring="neg_log_loss", cv=skf, n_jobs=-1,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"  {'Trial':>6}  {'neg-logloss':>12}  {'best so far':>12}")
    print("  " + "-" * 36)

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        marker = "  *** best ***" if trial.number == study.best_trial.number else ""
        print(
            f"  {trial.number + 1:>6}  {trial.value:>12.4f}"
            f"  {study.best_value:>12.4f}{marker}"
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])

    print(f"\nBest neg-logloss : {study.best_value:.4f}")
    print(f"Best params      : {study.best_params}")

    # Return the full params dict (base + tuned)
    return {**xgb_base, **study.best_params}


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna TPE hyperparameter search for NHL xG model"
    )
    parser.add_argument("--trials",   type=int,  default=40,  help="Optuna trials (default: 40)")
    parser.add_argument("--gpu",      action="store_true",    help="Use GPU (XGBoost device=cuda)")
    parser.add_argument("--cv-folds", type=int,  default=3,   help="CV folds (default: 3)")
    parser.add_argument("--seed",     type=int,  default=42,  help="Random seed (default: 42)")
    parser.add_argument("--data",     type=str,  default=None,
                        help="Path to xg_table.csv.gz (default: data/shots/xg_table.csv.gz)")
    args = parser.parse_args()

    csv_path = Path(args.data) if args.data else CSV_PATH
    if not csv_path.exists():
        print(f"ERROR: data file not found: {csv_path}")
        sys.exit(1)

    print(f"Building feature matrix via XGModel pipeline ({csv_path.name}) ...")
    X_train, y_train, _X_hold, _y_hold = _get_feature_matrix(csv_path, HOLDOUT_SEASONS)

    device_label = "GPU (cuda)" if args.gpu else "CPU"
    print(
        f"\nStarting Optuna TPE search "
        f"({args.trials} trials, {args.cv_folds}-fold CV, {device_label}) ..."
    )
    best_params = optuna_search_xgb(
        X_train, y_train,
        n_trials=args.trials,
        random_state=args.seed,
        use_gpu=args.gpu,
        cv_folds=args.cv_folds,
    )

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as fh:
        json.dump(best_params, fh, indent=2)
    print(f"\nBest params saved to {BEST_PARAMS_PATH}")
    print("\nNext step — retrain and evaluate using these params:")
    print("  .venv\\Scripts\\python main.py --evaluate --retrain --skip-fetch --tag optuna-tuned")


if __name__ == "__main__":
    main()
