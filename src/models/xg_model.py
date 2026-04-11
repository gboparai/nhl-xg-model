"""
xg_model.py
===========
XGBoost expected-goals pipeline for NHL shot data.

This is a pure-Python / XGBoost pipeline that reads a shot-event CSV,
engineers features, trains or loads a pre-trained model, evaluates on
holdout seasons, and writes plots.

To use your own shot-event data pass a custom path:
    XGModel(data_path='...')

Required columns in the shot table
-----------------------------------
game_id, season, period, time, event_type, x_coord, y_coord,
shot_type, score_diff, strength_state, last_event_type,
last_event_x, last_event_y, time_since_last_event

The wrapper adds the same derived features as the original pipeline:
  - shot_distance_calc, shot_angle_calc, distance_sq, angle_sq,
    dist_x_angle, in_slot, home_plate, behind_net, radial_distance,
    dist_bin, log_distance, time_fraction,
    delta_x, delta_y, distance_from_last_event, movement_speed,
    score_diff buckets (down2+, down1, tie, up1, up2+),
    one-hot shotType columns
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── module-level helpers ────────────────────────────────────────────────────

def _mmss_to_seconds(value: str | int | float) -> float:
    """Convert MM:SS string *or* bare seconds to a float in seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parts = str(value).split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return float(value)
    except (ValueError, AttributeError):
        return 0.0


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (integer coordinates OK)."""
    n = len(polygon)
    inside = False
    px, py = x, y
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ── paths ──────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent          # src/models/
_ROOT     = _THIS_DIR.parent.parent        # project root
_DEFAULT_DATA     = _ROOT / "data" / "shots" / "xg_table.csv.gz"
_PRETRAINED_MODEL = _ROOT / "models" / "xgb_combined_gpu_random.pkl"
_OPTUNA_PARAMS    = _ROOT / "models" / "optuna_best_params.json"
_PLOTS_DIR        = _ROOT / "plots"


class XGModel:
    """
    End-to-end XGBoost expected-goals pipeline.

    Parameters
    ----------
    data_path : str | Path | None
        Path to a shot-event CSV (gzipped or plain).  Defaults to
        ``data/shots/xg_table.csv.gz``.
    use_pretrained : bool
        If ``True`` (default) load the pre-trained pickle and skip
        training.  Set to ``False`` to retrain from
        scratch using the data at ``data_path``.
    test_seasons : list[str] | None
        Seasons reserved for evaluation, e.g. ``["20222023","20232024",
        "20242025"]``.  Defaults to the original paper split.
    use_gpu : bool
        Pass ``tree_method="gpu_hist"`` to XGBoost.  Falls back to CPU
        automatically if no CUDA device is found.
    plots_dir : str | Path | None
        Directory to write evaluation plots.  Defaults to ``plots/``.
    verbose : bool
        Print progress messages.
    """

    # ---------- benchmark numbers from the paper ----------
    BENCHMARK = {
        "roc_auc": 0.799,
        "log_loss": 0.416,
        "brier_score": 0.140,
    }

    def __init__(
        self,
        data_path: str | Path | None = None,
        use_pretrained: bool = True,
        test_seasons: list[str] | None = None,
        use_gpu: bool = False,
        plots_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.data_path = Path(data_path) if data_path else _DEFAULT_DATA
        self.use_pretrained = use_pretrained
        self.test_seasons = test_seasons or ["20222023", "20232024", "20242025"]
        self.use_gpu = use_gpu
        self.plots_dir = Path(plots_dir) if plots_dir else _PLOTS_DIR
        self.verbose = verbose

        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Execute the full pipeline and return a results dictionary.

        Returns
        -------
        dict with keys:
            predictions : pd.DataFrame  – shot-level xG probabilities
            metrics     : dict          – roc_auc, log_loss, brier_score
            feature_importance : pd.Series
            plots       : dict[str, Path]
            benchmark   : dict          – original paper numbers
        """
        self._log("=" * 60)
        self._log("NHL xG Model")
        self._log("=" * 60)

        # 1. Load data
        shots = self._load_data()

        # 2. Feature engineering
        shots = self._clean_coords(shots)
        shots = self._add_prior_event_features(shots)
        shots = self._add_shift_features(shots)
        X, y, feature_cols = self._build_feature_matrix(shots)
        self._feature_cols = feature_cols

        # ── Holdout seasons ───────────────────────────────────────────────────
        HOLDOUT_SEASONS = [20222023, 20232024, 20242025]
        holdout_seasons_int = [int(s) for s in self.test_seasons] \
            if self.test_seasons else HOLDOUT_SEASONS

        # 4. Model
        if self.use_pretrained and _PRETRAINED_MODEL.exists():
            self._log(f"Loading pre-trained model from {_PRETRAINED_MODEL}")
            with open(_PRETRAINED_MODEL, "rb") as fh:
                self._model = pickle.load(fh)

            is_holdout = shots["season"].isin(holdout_seasons_int)
            X_hold = X[is_holdout]
            y_hold = y[is_holdout]
            shots_hold = shots[is_holdout].copy()

            # Align feature columns to whatever the loaded pkl was trained on,
            # adding any missing columns as 0 and dropping any extras.
            try:
                pkl_features = list(self._model["xgb"].feature_names_in_)
                X_hold = X_hold.reindex(columns=pkl_features, fill_value=0)
            except Exception:
                pass

            self._log(f"  Holdout seasons: {holdout_seasons_int}")
            self._log(f"  Holdout shots  : {len(X_hold):,}")
            self._log(f"  Features       : {X_hold.shape[1]}")

            try:
                probs = self._model.predict_proba(X_hold)[:, 1]
            except Exception:
                probs = self._model.predict_proba(
                    np.nan_to_num(X_hold.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                )[:, 1]
            test_df = shots_hold
            test_df["xg"] = probs
            y_eval = y_hold

            # 6a. Evaluate
            metrics = self._evaluate(y_hold, probs)

            # Season-by-season breakdown
            self._season_breakdown(shots_hold, probs, y_hold)

        else:
            # 3. Train / test split (time-based, no leakage)
            is_test = shots["season"].isin(holdout_seasons_int)
            X_train, X_test = X[~is_test], X[is_test]
            y_train, y_test = y[~is_test], y[is_test]
            self._log(f"Train: {len(X_train):,} shots | Test: {len(X_test):,} shots")

            self._log("Training XGBoost model (random search, 20 trials)…")
            self._model = self._train(X_train, y_train)

            # Save the newly trained model so --tag re-runs don't need --retrain
            with open(_PRETRAINED_MODEL, "wb") as _fh:
                pickle.dump(self._model, _fh)
            self._log(f"Model saved -> {_PRETRAINED_MODEL}")

            # 5b. Predict
            probs = self._model.predict_proba(X_test)[:, 1]
            test_df = shots[is_test].copy()
            test_df["xg"] = probs
            y_eval = y_test

            # 6b. Evaluate
            metrics = self._evaluate(y_test, probs)

            # Season-by-season breakdown
            self._season_breakdown(test_df, probs, y_test)

        # 7. Feature importance
        fi = self._feature_importance()

        # 8. Plots
        plot_paths = self._make_plots(y_eval, probs, fi)

        self._log("\n── Results ──────────────────────────────────")
        self._log(f"  ROC AUC    : {metrics['roc_auc']:.4f}  (paper: {self.BENCHMARK['roc_auc']:.3f})")
        self._log(f"  Log-loss   : {metrics['log_loss']:.4f}  (paper: {self.BENCHMARK['log_loss']:.3f})")
        self._log(f"  Brier score: {metrics['brier_score']:.4f}  (paper: {self.BENCHMARK['brier_score']:.3f})")
        self._log("─────────────────────────────────────────────\n")

        return {
            "predictions": test_df,
            "metrics": metrics,
            "feature_importance": fi,
            "plots": plot_paths,
            "benchmark": self.BENCHMARK,
        }

    def install_dependencies(self) -> None:
        """Install Python dependencies listed in requirements.txt."""
        req_file = _ROOT / "requirements.txt"
        if not req_file.exists():
            self._log("No requirements.txt found – skipping install.")
            return
        self._log(f"Installing dependencies from {req_file} …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"]
        )
        self._log("Dependencies installed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── 1. Data loading ────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Shot data not found at {self.data_path}.\n"
                "Either run the main pipeline first to export shot data, "
                "or point data_path to a compatible CSV/gz file."
            )
        self._log(f"Loading data from {self.data_path} …")
        df = pd.read_csv(self.data_path, compression="infer", low_memory=False)
        self._log(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

        # Drop pre-2013-14 seasons (rink-layout noise – original paper)
        if "season" in df.columns:
            before = len(df)
            # season may be stored as int (20132014) or string
            df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)
            df = df[df["season"] >= 20132014].reset_index(drop=True)
            self._log(f"  After season filter (>=20132014): {len(df):,} rows (dropped {before - len(df):,})")

        return df

    # ── 2. Geometry cleaning ────────────────────────────────────────────
    # Mirrors the notebook's clean_and_calculate_coords() exactly.

    def _clean_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Accept both our export names (xCoord/yCoord) and generic fallbacks
        x_col = self._find_col(df, ["xCoord", "x_coord", "x"])
        y_col = self._find_col(df, ["yCoord", "y_coord", "y"])

        if x_col is None or y_col is None:
            raise KeyError("Cannot find x/y coordinate columns in shot data.")

        # Rename to canonical names used throughout
        df["xCoord"] = pd.to_numeric(df[x_col], errors="coerce")
        df["yCoord"] = pd.to_numeric(df[y_col], errors="coerce")

        # Mirror: all shots attack rightward (x > 0)
        mask = df["xCoord"] < 0
        df.loc[mask, "xCoord"] = -df.loc[mask, "xCoord"]
        df.loc[mask, "yCoord"] = -df.loc[mask, "yCoord"]

        # Mirror prev_event coords using the SAME mask so delta_x/delta_y are
        # computed in the same (attack-rightward) coordinate frame.
        if "prev_event_x" in df.columns:
            df.loc[mask, "prev_event_x"] = -df.loc[mask, "prev_event_x"]
        if "prev_event_y" in df.columns:
            df.loc[mask, "prev_event_y"] = -df.loc[mask, "prev_event_y"]

        # Drop obvious outliers
        df = df[
            df["xCoord"].between(-99, 99) & df["yCoord"].between(-42, 42)
        ].dropna(subset=["xCoord", "yCoord"]).reset_index(drop=True)

        x_abs = df["xCoord"].abs()
        df["shot_distance_calc"] = np.sqrt((89 - x_abs) ** 2 + df["yCoord"] ** 2).round(2)
        df["shot_angle_calc"] = np.degrees(
            np.arctan2(df["yCoord"], (89 - df["xCoord"]))
        ).round(2)

        # Non-linear terms
        df["distance_sq"]  = df["shot_distance_calc"] ** 2
        df["angle_sq"]     = df["shot_angle_calc"] ** 2
        df["dist_x_angle"] = df["shot_distance_calc"] * df["shot_angle_calc"]
        df["log_distance"] = np.log1p(df["shot_distance_calc"])
        df["radial_distance"] = df["yCoord"].abs()

        # Spatial zone flags
        df["in_slot"] = (
            (df["shot_distance_calc"] < 25) & (df["shot_angle_calc"].abs() < 30)
        ).astype(int)
        df["behind_net"] = ((df["xCoord"] > 89) | (df["xCoord"] < -89)).astype(int)

        # Home-plate polygon (notebook definition)
        home_plate_polygon = [(89, -3.5), (89, 3.5), (69, 22), (52, 0), (69, -22)]
        df["home_plate"] = [
            int(_point_in_polygon(x, y, home_plate_polygon))
            for x, y in zip(df["xCoord"], df["yCoord"])
        ]

        # Binned distance (numeric labels, same as notebook)
        df["dist_bin"] = pd.cut(
            df["shot_distance_calc"],
            bins=[0, 10, 20, 30, 50, 200],
            labels=[1, 2, 3, 4, 5],
        ).astype(float).fillna(0)

        return df

    # ── 3. Context features ─────────────────────────────────────────────
    # Mirrors the notebook's add_prior_event_features() exactly.

    def _add_prior_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def _to_s(v):
            try:
                parts = str(v).split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                return float(v)
            except Exception:
                return np.nan

        # Recompute exactly as the notebook does using prev_event_* columns
        # that export_shots.py now provides from the raw JSON lag() pattern.
        df["current_time_s"] = pd.to_numeric(df["time_in_period"].apply(_to_s), errors="coerce")
        df["prev_time_s"] = pd.to_numeric(df["prev_event_time"].apply(_to_s), errors="coerce")

        same_period = df["period_number"] == df["prev_event_period"]

        df["time_since_last_event"] = np.where(
            same_period,
            df["current_time_s"] - df["prev_time_s"],
            np.nan,
        )

        valid_coords = (
            same_period
            & df[["xCoord", "yCoord", "prev_event_x", "prev_event_y"]].notnull().all(axis=1)
        )

        df["distance_from_last_event"] = np.where(
            valid_coords,
            np.sqrt(
                (df["xCoord"] - df["prev_event_x"]) ** 2
                + (df["yCoord"] - df["prev_event_y"]) ** 2
            ),
            np.nan,
        )

        df["delta_x"] = np.where(valid_coords, df["xCoord"] - df["prev_event_x"], np.nan)
        df["delta_y"] = np.where(valid_coords, df["yCoord"] - df["prev_event_y"], np.nan)

        df["movement_angle"] = np.degrees(np.arctan2(df["delta_y"], df["delta_x"]))
        df["movement_speed"] = df["distance_from_last_event"] / df["time_since_last_event"]

        # time fraction within period
        df["time_fraction"] = (df["current_time_s"] / 1200).clip(0, 1)

        # ── NEW: rebound flag ────────────────────────────────────────────────
        # Last event was a shot/missed-shot, same period, <3 s ago, <10 ft away
        prev_type = self._find_col(df, ["prev_event_type"])
        shot_event_types = {"shot-on-goal", "missed-shot", "blocked-shot",
                            "shot", "missed_shot", "blocked_shot",
                            "SHOT", "MISSED_SHOT", "BLOCKED_SHOT"}
        if prev_type:
            prev_was_shot = df[prev_type].isin(shot_event_types)
        else:
            prev_was_shot = pd.Series(False, index=df.index)

        df["is_rebound"] = (
            same_period
            & prev_was_shot
            & (df["time_since_last_event"].fillna(99) <= 3.0)
            & (df["distance_from_last_event"].fillna(99) <= 20.0)
        ).astype(int)

        # ── NEW: rush flag ───────────────────────────────────────────────────
        # Puck travelled >30 ft in <3 s AND moved forward (delta_x > 10)
        df["is_rush"] = (
            same_period
            & (df["distance_from_last_event"].fillna(0) > 30.0)
            & (df["time_since_last_event"].fillna(99) < 3.0)
            & (df["delta_x"].fillna(0) > 10.0)
        ).astype(int)

        # ── NEW: cross-ice pass flag ─────────────────────────────────────────────
        # Large lateral puck movement in the same coordinate frame (both xCoord
        # and prev_event_x/y are now mirrored to attack-rightward).
        # Gates: previous event in OZ (prev_event_x > 25), within 5s, and
        # lateral movement > 20ft — captures goalie-scramble one-timer situations.
        prev_in_oz = valid_coords & (df["prev_event_x"].fillna(0) > 25)
        df["is_cross_ice"] = (
            prev_in_oz
            & (df["time_since_last_event"].fillna(99) <= 5.0)
            & (df["delta_y"].fillna(0).abs() > 20.0)
        ).astype(int)

        # ── NEW: last-event-type one-hot dummies ─────────────────────────────
        last_event_types = [
            "shot-on-goal", "missed-shot", "blocked-shot",
            "faceoff", "hit", "giveaway", "takeaway",
            "stoppage", "goal", "penalty",
        ]
        # Normalise to lower-case hyphenated form
        _type_map = {
            "shot": "shot-on-goal", "shot_on_goal": "shot-on-goal",
            "SHOT": "shot-on-goal",
            "missed_shot": "missed-shot", "MISSED_SHOT": "missed-shot",
            "blocked_shot": "blocked-shot", "BLOCKED_SHOT": "blocked-shot",
            "stop": "stoppage", "STOP": "stoppage",
            "FACEOFF": "faceoff", "HIT": "hit",
            "GIVEAWAY": "giveaway", "TAKEAWAY": "takeaway",
            "GOAL": "goal", "PENALTY": "penalty",
        }
        if prev_type:
            normalised = df[prev_type].map(lambda v: _type_map.get(str(v), str(v).lower()) if pd.notna(v) else np.nan)
        else:
            normalised = pd.Series(np.nan, index=df.index)

        normalised = normalised.where(normalised.isin(last_event_types), other=np.nan)
        last_event_dummies = pd.get_dummies(normalised, prefix="last_event")
        # Rename hyphenated dummy columns to use underscores (consistent naming)
        last_event_dummies.columns = [c.replace("-", "_") for c in last_event_dummies.columns]
        # Ensure all expected dummy columns exist even if a category is absent
        for et in last_event_types:
            col = f"last_event_{et.replace('-', '_')}"
            if col not in last_event_dummies.columns:
                last_event_dummies[col] = 0
        df = pd.concat([df.reset_index(drop=True), last_event_dummies.reset_index(drop=True)], axis=1)

        # ── NEW: same_event_team ─────────────────────────────────────────────
        # 1 if the previous event was by the same team as the shooter (sustained
        # possession); 0 if it was a turnover/opponent event.
        prev_team_col = self._find_col(df, ["prev_event_team_id"])
        shoot_team_col = self._find_col(df, ["shooting_team_id"])
        if prev_team_col and shoot_team_col:
            pt = pd.to_numeric(df[prev_team_col], errors="coerce")
            st = pd.to_numeric(df[shoot_team_col], errors="coerce")
            df["same_event_team"] = (same_period & (pt == st)).astype(int)
        else:
            df["same_event_team"] = 0

        # ── NEW: era flags (hockeyR-style rule-change eras) ──────────────────
        # Accounts for shifting NHL goal rates driven by rule/equipment changes.
        #   era_pre2014  : 2010-11 through 2012-13 (pre goalie pad reduction)
        #   era_2014     : 2013-14 through 2017-18 (first pad reduction)
        #   era_2019     : 2018-19 through 2020-21 (chest/arm pad reduction)
        #   era_2022     : 2021-22 onward           (cross-checking emphasis)
        szn = df["season"] if "season" in df.columns else pd.Series(0, index=df.index)
        szn_int = pd.to_numeric(szn, errors="coerce").fillna(0).astype(int)
        df["era_pre2014"] = (szn_int < 20132014).astype(int)
        df["era_2014"]    = ((szn_int >= 20132014) & (szn_int <= 20172018)).astype(int)
        df["era_2019"]    = ((szn_int >= 20182019) & (szn_int <= 20202021)).astype(int)
        df["era_2022"]    = (szn_int >= 20212022).astype(int)

        # score differential buckets (exactly as notebook's compute_binned_score_diff)
        # IMPORTANT: homeScore/awayScore are NaN until the first goal of a game.
        # NaN comparisons are all False in Python, so NaN diff falls through to "up2+".
        # We replicate this exactly (do NOT fillna before computing the diff).
        home_col = self._find_col(df, ["homeScore", "home_score"])
        away_col = self._find_col(df, ["awayScore", "away_score"])

        if home_col and away_col:
            home_s = pd.to_numeric(df[home_col], errors="coerce")
            away_s = pd.to_numeric(df[away_col], errors="coerce")
            diff   = home_s - away_s  # NaN where score not yet recorded
        else:
            diff = pd.Series(np.nan, index=df.index)

        def _bucket(d):
            # Matches notebook's compute_binned_score_diff row-by-row behaviour:
            # NaN comparisons are False → NaN falls through to "up2+"
            import math
            if isinstance(d, float) and math.isnan(d):
                return "up2+"
            if d <= -2: return "down2+"
            if d == -1: return "down1"
            if d == 0:  return "tie"
            if d == 1:  return "up1"
            return "up2+"

        df["score_diff_cat"] = diff.map(_bucket)

        return df

    # ── 3b. Shift features ────────────────────────────────────────────

    def _add_shift_features(
        self,
        df: pd.DataFrame,
        live_shifts: "pd.DataFrame | None" = None,
    ) -> pd.DataFrame:
        """
        Join shift data to compute (fully vectorised):
          shooter_toi   : seconds the shooter has been on ice when they shoot
          opp_shift_min : min TOI of any on-ice opponent at shot time
          opp_shift_avg : avg TOI of on-ice opponents at shot time
          opp_shift_max : max TOI of any on-ice opponent at shot time

        Strategy: merge shots → shifts on game_id+period, filter to shifts that
        overlap the shot time, then aggregate per shot.

        Parameters
        ----------
        live_shifts : pd.DataFrame | None
            Pre-fetched shift DataFrame (same schema as shift_lookup.parquet).
            When provided, the parquet file is not read.  Pass ``None`` to
            fall back to the on-disk parquet.
        """
        _SHIFT_LOOKUP = _ROOT / "data" / "shots" / "shift_lookup.parquet"
        if live_shifts is not None:
            self._log("  Using live shift data ...")
            shifts = live_shifts
        elif not _SHIFT_LOOKUP.exists():
            self._log("  [shift] shift_lookup.parquet not found — skipping shift features")
            for col in ["shooter_toi", "opp_shift_min", "opp_shift_avg", "opp_shift_max"]:
                df[col] = np.nan
            return df
        else:
            self._log("  Loading shift lookup table...")
            shifts = pd.read_parquet(_SHIFT_LOOKUP)

        def _mmss_to_s(v):
            try:
                parts = str(v).split(":")
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                return float(v)
            except Exception:
                return np.nan

        df = df.copy()
        df["_shot_s"]      = df["time_in_period"].apply(_mmss_to_s)
        df["_game_id_int"] = pd.to_numeric(df["game_id"],           errors="coerce").astype("Int64")
        df["_period_int"]  = pd.to_numeric(df["period_number"],     errors="coerce").astype("Int64")
        # Goals use scoringPlayerId; shots use shootingPlayerId — coalesce them
        shooting = pd.to_numeric(df["shootingPlayerId"], errors="coerce")
        scoring  = pd.to_numeric(df.get("scoringPlayerId", pd.Series(dtype=float)), errors="coerce")
        df["_shooter_id"]  = shooting.fillna(scoring).astype("Int64")
        df["_shoot_team"]  = pd.to_numeric(df["shooting_team_id"],  errors="coerce").astype("Int64")
        df["_shot_idx"]    = np.arange(len(df))

        shifts["game_id"]   = shifts["game_id"].astype("Int64")
        shifts["period"]    = shifts["period"].astype("Int64")
        shifts["player_id"] = shifts["player_id"].astype("Int64")
        shifts["team_id"]   = shifts["team_id"].astype("Int64")

        self._log(f"  Computing shift features for {len(df):,} shots (chunked by season)...")

        # Process season-by-season to avoid a 375M-row cross-product OOM.
        # Within each season we merge shots→shifts on game_id+period, then
        # immediately filter to the overlapping window, keeping only ~20-30
        # rows per shot rather than all shifts in the period.
        shifts_renamed = shifts.rename(columns={
            "game_id": "_game_id_int",
            "period":  "_period_int",
        })

        shot_cols = df[["_shot_idx", "_game_id_int", "_period_int", "_shot_s",
                        "_shooter_id", "_shoot_team", "season"]].copy()

        all_shooter_toi = []
        all_opp_stats   = []

        seasons = sorted(shot_cols["season"].dropna().unique())
        for szn in seasons:
            s_shots = shot_cols[shot_cols["season"] == szn]
            # Games in this season
            game_ids = s_shots["_game_id_int"].dropna().unique()
            s_shifts = shifts_renamed[shifts_renamed["_game_id_int"].isin(game_ids)]

            if s_shifts.empty or s_shots.empty:
                continue

            merged = s_shots.merge(s_shifts, on=["_game_id_int", "_period_int"], how="left")

            # Keep only shifts that were already active before the shot.
            # Exclude shifts where start_s == shot_s: these are post-event line changes
            # whose shift record begins at the exact event timestamp (NHL data artefact),
            # giving toi=0 and creating a spurious opp_shift_min signal for goals.
            merged = merged[
                (merged["start_s"] < merged["_shot_s"]) &
                (merged["end_s"]  >= merged["_shot_s"])
            ]
            merged = merged.copy()
            merged["toi"] = merged["_shot_s"] - merged["start_s"]

            # Shooter TOI
            st = merged[merged["player_id"] == merged["_shooter_id"]] \
                       .groupby("_shot_idx")["toi"].first().rename("shooter_toi")
            all_shooter_toi.append(st)

            # Opponent stats (skaters only)
            opp = merged[(merged["team_id"] != merged["_shoot_team"]) & ~merged["is_goalie"]]
            os = opp.groupby("_shot_idx")["toi"] \
                    .agg(opp_shift_min="min", opp_shift_avg="mean", opp_shift_max="max")
            all_opp_stats.append(os)

        shooter_toi = pd.concat(all_shooter_toi) if all_shooter_toi else pd.Series(dtype=float, name="shooter_toi")
        opp_stats   = pd.concat(all_opp_stats)   if all_opp_stats   else pd.DataFrame(columns=["opp_shift_min","opp_shift_avg","opp_shift_max"])

        # Join back to df
        df = df.join(shooter_toi, on="_shot_idx")
        df = df.join(opp_stats,   on="_shot_idx")

        # Drop temp columns
        df = df.drop(columns=["_shot_s", "_game_id_int", "_period_int",
                               "_shooter_id", "_shoot_team", "_shot_idx"])

        pct = df["shooter_toi"].notna().mean() * 100
        self._log(f"  shooter_toi coverage: {pct:.1f}%")
        return df

    # ── 4. Feature matrix ───────────────────────────────────────────────
    # Mirrors the notebook's build_feature_matrix() exactly.

    def _build_feature_matrix(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        # ─ target ───────────────────────────────────────────────────────────────
        target_col = self._find_col(df, ["shot_made", "is_goal", "goal"])
        if target_col is None:
            raise KeyError("Cannot find target column (shot_made / is_goal / goal).")
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

        # ─ is_forward ──────────────────────────────────────────────────────────
        fwd_col = self._find_col(df, ["is_forward", "shooter_position"])
        if fwd_col == "shooter_position":
            df["is_forward"] = df[fwd_col].isin(["L", "R", "C"]).astype(int)
        elif fwd_col == "is_forward":
            df["is_forward"] = pd.to_numeric(df[fwd_col], errors="coerce").fillna(0).astype(int)
        else:
            df["is_forward"] = 0

        # ─ numeric columns (same list as notebook) ────────────────────────
        numeric_cols = [
            "shot_distance_calc", "shot_angle_calc", "is_forward",
            "time_since_last_event", "distance_from_last_event",
            "delta_x", "delta_y", "movement_angle", "movement_speed",
        ]
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0

        X_num = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).copy()

        # Non-linear / cross terms
        X_num["distance_sq"]              = X_num["shot_distance_calc"] ** 2
        X_num["log_distance"]             = np.log1p(X_num["shot_distance_calc"])
        X_num["angle_sq"]                 = X_num["shot_angle_calc"] ** 2
        X_num["dist_x_angle"]             = X_num["shot_distance_calc"] * X_num["shot_angle_calc"]
        X_num["movement_speed_sq"]        = X_num["movement_speed"] ** 2
        X_num["time_since_last_event_sq"] = X_num["time_since_last_event"] ** 2
        X_num["dist_x_speed"]             = X_num["shot_distance_calc"] * X_num["movement_speed"]

        # Spatial indicators from geometry step
        for flag in ["dist_bin", "in_slot", "behind_net", "radial_distance",
                     "home_plate", "time_fraction"]:
            X_num[flag] = df.get(flag, pd.Series(0, index=df.index)).fillna(0)

        # NEW: rebound / rush / cross-ice flags
        for flag in ["is_rebound", "is_rush", "is_cross_ice"]:
            X_num[flag] = df.get(flag, pd.Series(0, index=df.index)).fillna(0)

        # NEW: interaction — rebound × distance (rebounds close = very dangerous)
        X_num["rebound_x_dist"] = X_num["is_rebound"] * X_num["shot_distance_calc"]

        # NEW (hockeyR): raw previous-event coordinates
        # Complements delta_x/delta_y by giving the model the absolute OZ position
        # of the prior event, which helps distinguish slot passes from rim passes.
        for flag in ["prev_event_x", "prev_event_y"]:
            X_num[flag] = df.get(flag, pd.Series(0.0, index=df.index)).fillna(0)

        # NEW (hockeyR): same_event_team — sustained possession vs. turnover shot
        X_num["same_event_team"] = df.get("same_event_team", pd.Series(0, index=df.index)).fillna(0)

        # NEW (hockeyR): era flags — rule-change era dummies
        for flag in ["era_pre2014", "era_2014", "era_2019", "era_2022"]:
            X_num[flag] = df.get(flag, pd.Series(0, index=df.index)).fillna(0)

        # Shift TOI features (opponent skaters only, goalies excluded).
        # shooter_toi uses coalesced shootingPlayerId/scoringPlayerId so goals are covered.
        for flag in ["shooter_toi", "opp_shift_min", "opp_shift_avg", "opp_shift_max"]:
            X_num[flag] = df.get(flag, pd.Series(np.nan, index=df.index)).fillna(0)

        # ── Situation encoding ────────────────────────────────────────────────
        # Encode game situation as a numeric feature so the model can learn
        # that the same shot location is worth more/less depending on context.
        # 0=5v5, 1=PP, 2=PK, 3=EN, 4=other  (maps situation_code integer clusters)
        sit_col = self._find_col(df, ["situation", "strength_state"])
        _SIT_MAP = {"5v5": 0, "PP": 1, "PK": 2, "EN": 3, "other": 4}
        if sit_col and sit_col == "situation":
            X_num["situation_num"] = df["situation"].map(_SIT_MAP).fillna(4).astype(int)
        elif self._find_col(df, ["situation_code"]):
            # Derive from situation_code when the labelled column isn't present yet.
            # Use is_home to distinguish PP (shooter on man-advantage) from PK.
            code = pd.to_numeric(df["situation_code"], errors="coerce").fillna(0).astype(int)
            away_g  = (code // 1000) % 10
            home_g  = code % 10
            home_sk = (code // 10) % 10
            away_sk = (code // 100) % 10
            diff    = home_sk - away_sk  # >0 = home has more skaters, <0 = away does

            ih_col = self._find_col(df, ["is_home"])
            ih = pd.to_numeric(df[ih_col], errors="coerce").fillna(-1).astype(int) if ih_col else pd.Series(-1, index=df.index)

            sit_num = pd.Series(4, index=df.index)  # default: other
            en_mask = (away_g == 0) | (home_g == 0)
            sit_num[en_mask] = 3
            sit_num[~en_mask & (diff == 0)] = 0   # 5v5 / 4v4 / 3v3 (even)
            # PP: shooter is on the team with MORE skaters
            pp_mask = ~en_mask & (diff != 0)
            # diff>0 → home has more → home shot is PP, away shot is PK
            # diff<0 → away has more → away shot is PP, home shot is PK
            shooter_is_pp = (
                ((diff > 0) & (ih == 1)) |  # home PP, home shooting
                ((diff < 0) & (ih == 0))    # away PP, away shooting
            )
            sit_num[pp_mask & shooter_is_pp]  = 1  # PP
            sit_num[pp_mask & ~shooter_is_pp] = 2  # PK
            X_num["situation_num"] = sit_num
        else:
            X_num["situation_num"] = 0

        # Period
        p_col = self._find_col(df, ["period_number", "period"])
        X_num["period"] = pd.to_numeric(df.get(p_col, 0), errors="coerce").fillna(0).astype(int)

        # ─ score diff one-hot ──────────────────────────────────────────────
        score_dummies = pd.get_dummies(
            df.get("score_diff_cat", pd.Series("tie", index=df.index)),
            prefix="scoreDiff",
        )
        expected_score_cols = [
            "scoreDiff_down2+",
            "scoreDiff_down1",
            "scoreDiff_tie",
            "scoreDiff_up1",
            "scoreDiff_up2+",
        ]
        for col in expected_score_cols:
            if col not in score_dummies.columns:
                score_dummies[col] = 0
        score_dummies = score_dummies[expected_score_cols]

        # ─ shot-type one-hot (4 types, same as notebook) ────────────────
        st_col = self._find_col(df, ["shotType", "shot_type", "secondary_type"])
        valid_types = ["wrist", "snap", "slap", "backhand"]
        if st_col:
            raw = df[st_col].str.lower().where(
                df[st_col].str.lower().isin(valid_types), other=np.nan
            )
        else:
            raw = pd.Series(np.nan, index=df.index)
        shot_dummies = pd.get_dummies(raw, prefix="shotType")
        expected_shot_cols = [f"shotType_{t}" for t in valid_types]
        for col in expected_shot_cols:
            if col not in shot_dummies.columns:
                shot_dummies[col] = 0
        shot_dummies = shot_dummies[expected_shot_cols]

        # NEW: last-event-type dummies (already in df from _add_prior_event_features)
        last_event_cols = [c for c in df.columns if c.startswith("last_event_")]
        last_event_df = df[last_event_cols].fillna(0).astype(int).reset_index(drop=True) if last_event_cols else pd.DataFrame(index=df.index)

        X = pd.concat([X_num, score_dummies, shot_dummies, last_event_df], axis=1)
        feature_cols = list(X.columns)
        return X, y, feature_cols

    # ── 5. Training ─────────────────────────────────────────────────────
    # Mirrors the notebook's random_search_xgb_gpu() exactly.

    def _train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train XGBoost: uses Optuna params if available, else 20-trial random search."""
        import json
        import math
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is required. Run: pip install xgboost")

        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore[import-untyped]

        # ── fast path: Optuna-tuned params from tune_hyperparams.py ──────
        if _OPTUNA_PARAMS.exists():
            with open(_OPTUNA_PARAMS) as _fh:
                best_params = json.load(_fh)
            self._log(f"  Using Optuna-tuned params from {_OPTUNA_PARAMS.name}")
            final = ImbPipeline([("xgb", XGBClassifier(**best_params))])
            final.fit(X_train, y_train)
            return final

        tree_method = "gpu_hist" if self.use_gpu else "hist"
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

        best_score  = -np.inf
        best_params = None
        n_trials    = 20

        self._log(f"Starting {n_trials}-trial random search (tree_method={tree_method}) ...")
        rng = np.random.default_rng(42)

        for i in range(n_trials):
            params = {
                "objective":         "binary:logistic",
                "eval_metric":       "logloss",
                "use_label_encoder": False,
                "random_state":      42,
                "tree_method":       tree_method,
                "max_depth":         int(rng.integers(2, 11)),
                "learning_rate":     float(np.exp(rng.uniform(math.log(0.005), math.log(0.3)))),
                "n_estimators":      int(rng.integers(50, 601)),
                "subsample":         float(rng.uniform(0.3, 1.0)),
                "colsample_bytree":  float(rng.uniform(0.3, 1.0)),
                "min_child_weight":  int(rng.integers(1, 51)),
                "reg_alpha":         float(rng.uniform(0.0, 1.0)),
                "reg_lambda":        float(rng.uniform(0.5, 3.0)),
                "gamma":             float(rng.uniform(0.0, 5.0)),
            }
            try:
                pipeline = ImbPipeline([("xgb", XGBClassifier(**params))])
                scores   = cross_val_score(
                    pipeline, X_train, y_train,
                    scoring="neg_log_loss", cv=skf, n_jobs=-1,
                )
            except Exception as gpu_err:
                if "gpu" in str(gpu_err).lower() and tree_method == "gpu_hist":
                    self._log("GPU not available — falling back to CPU hist")
                    tree_method = "hist"
                    params["tree_method"] = "hist"
                    pipeline = ImbPipeline([("xgb", XGBClassifier(**params))])
                    scores   = cross_val_score(
                        pipeline, X_train, y_train,
                        scoring="neg_log_loss", cv=skf, n_jobs=-1,
                    )
                else:
                    raise

            mean_score = scores.mean()
            self._log(f"  Trial {i+1:>2}/{n_trials}  neg_logloss={mean_score:.4f}")
            if mean_score > best_score:
                best_score  = mean_score
                best_params = params

        self._log(f"Best params: {best_params}")
        final = ImbPipeline([("xgb", XGBClassifier(**best_params))])
        final.fit(X_train, y_train)
        return final

    # ── 6. Evaluation ────────────────────────────────────────────────────

    def _evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

        goals = float(y_true.sum())
        calibration = float(np.sum(y_pred)) / goals * 100 if goals > 0 else float("nan")
        return {
            "roc_auc":     round(float(roc_auc_score(y_true, y_pred)), 4),
            "log_loss":    round(float(log_loss(y_true, y_pred)), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_pred)), 4),
            "calibration": round(calibration, 1),
        }

    def _season_breakdown(
        self,
        shots_df: pd.DataFrame,
        probs: np.ndarray,
        y_true: pd.Series,
    ) -> None:
        """Print a season-by-season AUC / log-loss / Brier breakdown."""
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

        shots_df = shots_df.copy()
        shots_df["_probs"]  = probs
        shots_df["_actual"] = y_true.values

        self._log("\n── Season-by-Season Breakdown ───────────────────────────────")
        self._log(f"  {'Season':<12} {'Shots':>8} {'Goals':>6} {'Goal%':>6}  {'AUC':>6}  {'LogLoss':>8}  {'Brier':>7}")
        self._log(f"  {'-'*12} {'-'*8} {'-'*6} {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}")

        for season in sorted(shots_df["season"].unique()):
            mask = shots_df["season"] == season
            y_s  = shots_df.loc[mask, "_actual"]
            p_s  = shots_df.loc[mask, "_probs"]
            n    = len(y_s)
            g    = int(y_s.sum())
            pct  = g / n * 100 if n else 0
            try:
                auc_s    = roc_auc_score(y_s, p_s)
                ll_s     = log_loss(y_s, p_s)
                brier_s  = brier_score_loss(y_s, p_s)
                self._log(
                    f"  {season:<12} {n:>8,} {g:>6,} {pct:>5.1f}%"
                    f"  {auc_s:>6.4f}  {ll_s:>8.4f}  {brier_s:>7.4f}"
                )
            except Exception:
                self._log(f"  {season:<12} {n:>8,} {g:>6,} {pct:>5.1f}%  (not enough data)")

        self._log(f"  {'─'*70}")
        # Overall
        auc_o   = roc_auc_score(y_true, probs)
        ll_o    = log_loss(y_true, probs)
        brier_o = brier_score_loss(y_true, probs)
        n_o, g_o = len(y_true), int(y_true.sum())
        self._log(
            f"  {'TOTAL':<12} {n_o:>8,} {g_o:>6,} {g_o/n_o*100:>5.1f}%"
            f"  {auc_o:>6.4f}  {ll_o:>8.4f}  {brier_o:>7.4f}"
        )
        self._log("")

    # ── 7. Feature importance ────────────────────────────────────────────

    def _feature_importance(self) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float)
        try:
            # Model may be a bare XGBClassifier or wrapped in an ImbPipeline
            xgb_step = (
                self._model["xgb"]
                if hasattr(self._model, "__getitem__")
                else self._model
            )
            # Use total_gain (cumulative information gain across all splits) rather
            # than the default feature_importances_ which returns split-count (weight).
            # total_gain better reflects how much each feature actually reduces loss.
            raw = xgb_step.get_booster().get_score(importance_type="total_gain")
            total = sum(raw.values()) or 1.0
            # Map f0/f1/... keys back to named columns when available
            n = xgb_step.n_features_in_
            idx_to_name = (
                {f"f{i}": c for i, c in enumerate(self._feature_cols)}
                if len(self._feature_cols) == n
                else {}
            )
            named = {idx_to_name.get(k, k): v / total for k, v in raw.items()}
            return pd.Series(named).sort_values(ascending=False)
        except Exception:
            return pd.Series(dtype=float)

    # ── 8. Plots ─────────────────────────────────────────────────────────

    def _make_plots(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        fi: pd.Series,
    ) -> dict[str, Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, auc
            from sklearn.calibration import calibration_curve
        except ImportError:
            self._log("matplotlib / sklearn not available – skipping plots.")
            return {}

        paths: dict[str, Path] = {}

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("NHL xG Model – ROC Curve")
        ax.legend()
        fig.tight_layout()
        p = self.plots_dir / "roc_curve.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["roc_curve"] = p

        # Calibration / reliability diagram
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=20, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "o-", label="xG model")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed goal rate")
        ax.set_title("NHL xG Model – Calibration")
        ax.legend()
        fig.tight_layout()
        p = self.plots_dir / "brier_score.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths["calibration"] = p

        # Feature importance
        if not fi.empty:
            top = fi.head(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            top[::-1].plot.barh(ax=ax)
            ax.set_title("NHL xG – Feature Importance (top 20)")
            ax.set_xlabel("Total gain share")
            fig.tight_layout()
            p = self.plots_dir / "feature_importance.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths["feature_importance"] = p

        self._log(f"Plots saved to {self.plots_dir}")
        return paths

    # ── Utility ──────────────────────────────────────────────────────────

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None
