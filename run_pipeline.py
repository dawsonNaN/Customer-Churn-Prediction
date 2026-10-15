"""Unified command-line entry point for the complete customer churn project."""
from pathlib import Path
import argparse
import sys
import os

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'baseline', 'advanced', 'interpret', 'segment', 'all'])
    parser.add_argument('--trials', type=int, default=10, help='Optuna trials per advanced model; default 10.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=Path, help='Alternative model-ready CSV for baseline/advanced/interpret.')
    parser.add_argument('--model', type=Path, help='Alternative CatBoost bundle for interpretation.')
    parser.add_argument('--scorecard', type=Path, help='Alternative scorecard for segmentation.')
    parser.add_argument('--no-lime', action='store_true', help='Skip the optional local LIME computation.')
    parser.add_argument('--output-dir', type=Path, help='Separate output root for an independent run.')
    args = parser.parse_args()
    if args.trials < 1: parser.error('--trials must be positive.')
    if args.output_dir:
        os.environ['CHURN_OUTPUT_DIR'] = str(args.output_dir.resolve())
    if args.stage in ['prepare', 'all']:
        from prepare_data import run
        prepared = run(seed=args.seed)
    data = args.data or (prepared if args.stage == 'all' else None)
    if args.stage in ['baseline', 'all']:
        from train_baselines import run
        run(data)
    if args.stage in ['advanced', 'all']:
        from train_advanced import run
        advanced = run(data, trials=args.trials, seed=args.seed)
    if args.stage in ['interpret', 'all']:
        from interpret_model import run
        # Week 7 explains CatBoost specifically. Keep that choice explicit even
        # if a different model/ensemble leads a newly tuned AUC ranking.
        model = args.model or (advanced['catboost_bundle'] if args.stage == 'all' else None)
        scorecard = run(data, model, seed=args.seed, include_lime=not args.no_lime)
    if args.stage in ['segment', 'all']:
        from segment_customers import run
        run(args.scorecard or (scorecard if args.stage == 'all' else None), seed=args.seed)
    print('Completed:', args.stage)


if __name__ == '__main__':
    main()
