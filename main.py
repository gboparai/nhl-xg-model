"""
Main execution script for NHL xG pipeline.

Workflow (step by step)
-----------------------
    1. python main.py --download --seasons 20222023 20232024 20242025
       (downloads raw NHL JSON files and stores player/team info in the DB)

    2. python main.py --use-local --export-shots
       (parses raw JSON → data/shots/xg_table.csv.gz)

    3. python main.py --build-shifts
       (builds data/shots/shift_lookup.parquet from raw shift JSON files)

    4. python main.py --evaluate --retrain
       (trains and evaluates the xG model, saves results to results/)

Full pipeline (one command)
---------------------------
    python main.py --full-pipeline --download --seasons 20222023 20232024 20242025

    # Use existing local data (no re-download):
    python main.py --full-pipeline --use-local --retrain
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure UTF-8 output on Windows (needed for unicode characters like ✅ ✓ ⚠)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

from src.database import init_database
from src.data import NHLDataFetcher
from src.data.download_raw_data import RawDataDownloader


def run_pipeline(seasons=None, fetch_data=True,
                use_local=False, local_dir='data/raw',
                download_raw=False, force_download=False,
                export_shots=False, build_shifts=False,
                evaluate=False, retrain=False, use_gpu=False, tag='',
                shot_population='default', test_seasons=None,
                tune=False, tune_trials=40, tune_cv_folds=3):
    """
    Run the NHL xG pipeline (all or selected steps).

    Args:
        seasons: List of seasons to fetch (e.g., ['20222023', '20232024'])
        fetch_data: Whether to fetch new data into the DB from NHL API or local files
        use_local: If True, load data from local JSON files instead of API
        local_dir: Directory containing local JSON files
        download_raw: If True, download raw JSON files first (Step 0)
        force_download: If True, re-download existing files
        export_shots: If True, parse raw JSON → data/shots/xg_table.csv.gz (Step 2)
        build_shifts: If True, build the shift lookup parquet file (Step 3)
        evaluate: If True, evaluate (and optionally retrain) the xG model (Step 4)
        retrain: If True (with evaluate), retrain from scratch instead of using pkl
        use_gpu: If True (with evaluate), use GPU for XGBoost training
        tag: Short label for the experiment (included in the results filename)
        shot_population: Shot population preset for export-shots and evaluate steps.
            'default'      - SOG + blocked-shot + goal  (baseline)
            'no-blocked'   - SOG + goal
            'all-attempts' - SOG + blocked-shot + missed-shot + goal
            'sog-missed'   - SOG + missed-shot + goal
        test_seasons: List of season IDs to use as holdout (e.g. ['20232024','20242025','20252026']).
            Defaults to the original three-season split.
        tune: If True, run Optuna hyperparameter search and save best model pkl.
        tune_trials: Number of Optuna trials (default: 40).
        tune_cv_folds: Number of CV folds for tuning (default: 3).
    """
    print("=" * 80)
    print("NHL EXPECTED GOALS PIPELINE")
    print("=" * 80)
    print()


    # ── Step 0: Download raw data if requested ──────────────────────────────
    if download_raw:
        if seasons is None:
            current_year = 2026
            seasons = [f"{year}{year+1}" for year in range(2010, current_year)]

        print("\n[0/3] Downloading raw data from NHL API...")
        print(f"Seasons: {seasons}")
        print(f"Destination: {local_dir}")

        downloader = RawDataDownloader(data_dir=local_dir, rate_limit_delay=0.5)
        downloader.download_teams()
        downloader.download_seasons(seasons, force=force_download)
        downloader.download_all_player_data(force=force_download)

        print("\nRaw data download complete!")

        # Automatically enable use_local after download
        use_local = True

    # ── Step 1: Initialize database ─────────────────────────────────────────
    print("\n[1/3] Initializing database...")
    init_database()

    # ── Step 2: Fetch / load data ────────────────────────────────────────────
    if fetch_data:
        if use_local:
            print("\n[2/2] Loading NHL data from local files...")
        else:
            print("\n[2/2] Fetching NHL data from API...")

        if seasons is None:
            current_year = 2026
            seasons = [f"{year}{year+1}" for year in range(2010, current_year)]

        print(f"Seasons: {seasons}")
        fetcher = NHLDataFetcher(
            rate_limit_delay=0.5,
            use_local=use_local,
            local_data_dir=local_dir
        )
        fetcher.fetch_teams()
        fetcher.fetch_seasons(seasons)
        fetcher.close()
    else:
        print("\n[2/3] Skipping data fetch (using existing data)")

    print("\n" + "=" * 80)
    print("Data preparation complete!")
    print("=" * 80)

    # ── Step 2: Export shots CSV ─────────────────────────────────────────────
    if export_shots:
        from src.data.export_shots import SHOT_POPULATIONS
        _shot_types, _csv_filename = SHOT_POPULATIONS[shot_population]
        _out_path = Path(local_dir).parent / 'shots' / _csv_filename
        print(f"\n[3a] Exporting shots CSV (population: {shot_population!r} → {_out_path.name})...")
        from src.data.export_shots import main as _export_shots_main
        _export_shots_main(shot_types=_shot_types, out_path=_out_path)

    # ── Step 3: Build shift lookup ───────────────────────────────────────────
    if build_shifts:
        print("\n[3b] Building shift lookup table (data/shots/shift_lookup.parquet)...")
        from src.models.build_shift_lookup import build as _build_shifts
        _build_shifts()

    # ── Step 3b: Optuna hyperparameter tuning ───────────────────────────────
    if tune:
        print(f"\n[3c] Running Optuna hyperparameter search ({tune_trials} trials, {tune_cv_folds}-fold CV)...")
        from src.data.export_shots import SHOT_POPULATIONS
        _, _csv_filename = SHOT_POPULATIONS[shot_population]
        _data_path = Path(local_dir).parent / 'shots' / _csv_filename
        _argv_backup = sys.argv[:]
        sys.argv = [sys.argv[0], f'--trials={tune_trials}', f'--cv-folds={tune_cv_folds}']
        if use_gpu:
            sys.argv.append('--gpu')
        if _data_path.exists():
            sys.argv.extend(['--data', str(_data_path)])
        try:
            from src.models.tune_hyperparams import main as _tune_main
            _tune_main()
        finally:
            sys.argv = _argv_backup

    # ── Step 4: Train / evaluate model ──────────────────────────────────────
    if evaluate:
        print("\n[4] Training / evaluating xG model...")
        from src.data.export_shots import SHOT_POPULATIONS
        _, _csv_filename = SHOT_POPULATIONS[shot_population]
        _data_path = Path(local_dir).parent / 'shots' / _csv_filename
        _argv_backup = sys.argv[:]
        sys.argv = [sys.argv[0]]
        if retrain:
            sys.argv.append('--retrain')
        if use_gpu:
            sys.argv.append('--gpu')
        if tag:
            sys.argv.extend(['--tag', tag])
        if test_seasons:
            sys.argv.extend(['--test-seasons', ','.join(test_seasons)])
        if _data_path.exists():
            sys.argv.extend(['--data-path', str(_data_path)])
        try:
            from src.models.evaluate import main as _evaluate_main
            _evaluate_main()
        finally:
            sys.argv = _argv_backup

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)
    print()


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description='NHL Expected Goals Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from scratch (download + DB + shots + shifts + train)
  python main.py --full-pipeline --download --seasons 20222023 20232024 20242025

  # Full pipeline using existing local data, retraining the model
  python main.py --full-pipeline --use-local --retrain

  # Full pipeline using existing local data, only evaluate (no retrain)
  python main.py --full-pipeline --use-local

  # Only download and populate the database
  python main.py --download --seasons 20242025

  # Only export shots CSV from already-downloaded raw files
  python main.py --export-shots --skip-fetch

  # Only build shift lookup parquet
  python main.py --build-shifts --skip-fetch

  # Only evaluate the model (data already prepared, uses pre-trained pkl)
  python main.py --evaluate --skip-fetch

  # Evaluate after retraining from scratch
  python main.py --evaluate --retrain --skip-fetch
        """
    )

    parser.add_argument('--seasons', nargs='+',
                       help='Seasons to fetch (e.g., 20222023 20232024)')
    parser.add_argument('--full-history', action='store_true',
                       help='Fetch full historical data (2010-2026)')
    parser.add_argument('--download', action='store_true',
                       help='Download raw JSON data from API before loading')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download of existing files (use with --download)')
    parser.add_argument('--use-local', action='store_true',
                       help='Load data from local JSON files instead of API')
    parser.add_argument('--local-dir', default='data/raw',
                       help='Directory containing local JSON files (default: data/raw)')
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip the DB fetch / load step')

    # Individual pipeline step flags
    parser.add_argument('--export-shots', action='store_true',
                       help='Parse raw JSON files → data/shots/xg_table.csv.gz')
    parser.add_argument('--build-shifts', action='store_true',
                       help='Build shift lookup parquet (data/shots/shift_lookup.parquet)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the xG model (uses pre-trained pkl unless --retrain is set)')
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain model from scratch (use with --evaluate or --full-pipeline)')
    parser.add_argument('--tune', action='store_true',
                       help='Run Optuna hyperparameter search and save best model pkl')
    parser.add_argument('--tune-trials', type=int, default=40,
                       help='Number of Optuna trials (default: 40)')
    parser.add_argument('--tune-cv-folds', type=int, default=3,
                       help='CV folds for tuning (default: 3; use 4 for more accuracy)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for XGBoost training (use with --evaluate or --full-pipeline)')
    parser.add_argument('--tag', default='',
                       help='Short label for this experiment run (included in the results filename)')
    parser.add_argument('--test-seasons', default=None,
                       help='Comma-separated holdout season IDs, e.g. 20232024,20242025,20252026 '
                            '(default: 20222023,20232024,20242025)')
    parser.add_argument('--shot-population',
                       choices=['default', 'no-blocked', 'all-attempts', 'sog-missed'],
                       default='default',
                       help=(
                           'Shot population used for --export-shots and --evaluate: '
                           'default (SOG+blocked+goal), '
                           'no-blocked (SOG+goal), '
                           'all-attempts (SOG+blocked+missed+goal), '
                           'sog-missed (SOG+missed+goal)'
                       ))

    # Full-pipeline shortcut
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run all pipeline steps: fetch DB → export shots → '
                            'build shifts → train model')

    args = parser.parse_args()

    # --full-pipeline enables all steps
    if args.full_pipeline:
        args.export_shots = True
        args.build_shifts = True
        args.evaluate = True

    # Determine seasons
    seasons = args.seasons
    if args.full_history:
        seasons = [f"{year}{year+1}" for year in range(2010, 2026)]

    # Run pipeline
    run_pipeline(
        seasons=seasons,
        fetch_data=not args.skip_fetch,
        use_local=args.use_local,
        local_dir=args.local_dir,
        download_raw=args.download,
        force_download=args.force,
        export_shots=args.export_shots,
        build_shifts=args.build_shifts,
        evaluate=args.evaluate,
        retrain=args.retrain,
        use_gpu=args.gpu,
        tag=args.tag,
        shot_population=args.shot_population,
        test_seasons=[s.strip() for s in args.test_seasons.split(',')] if args.test_seasons else None,
        tune=args.tune,
        tune_trials=args.tune_trials,
        tune_cv_folds=args.tune_cv_folds,
    )


if __name__ == "__main__":
    main()
