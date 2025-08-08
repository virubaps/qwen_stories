import argparse
import json
import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from ml.training.pipeline import TrainingPipeline, TrainingConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost + LSTM ensemble")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols like NSE_SBIN or file names like NSE_SBIN.csv")
    p.add_argument("--data_dir", default=PROJECT_ROOT, help="Directory containing CSV files")
    p.add_argument("--lookback", type=int, default=60, help="LSTM sequence length")
    p.add_argument("--horizon", type=int, default=5, help="Future horizon in bars for labeling")
    p.add_argument("--return_threshold", type=float, default=0.003, help="Return threshold for classification")
    p.add_argument("--model_dir", default=os.path.join(PROJECT_ROOT, "models"), help="Output directory for model artifacts")
    p.add_argument("--use_optuna", action="store_true", help="Enable Optuna tuning for XGBoost")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainingConfig(
        symbols=args.symbols,
        data_dir=args.data_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        return_threshold=args.return_threshold,
        model_dir=args.model_dir,
        use_optuna=args.use_optuna,
    )
    pipeline = TrainingPipeline(cfg)
    artifacts = pipeline.run()
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()