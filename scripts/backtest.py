import argparse
import json
import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from ml.backtesting.engine import BacktestEngine, BacktestConfig


def parse_args():
    p = argparse.ArgumentParser(description="Backtest trained ensemble")
    p.add_argument("--symbols", nargs="+", required=True, help="Symbols like NSE_SBIN or file names like NSE_SBIN.csv")
    p.add_argument("--data_dir", default=PROJECT_ROOT, help="Directory containing CSV files")
    p.add_argument("--start", default=None, help="Start datetime (YYYY-MM-DD or full timestamp)")
    p.add_argument("--end", default=None, help="End datetime (YYYY-MM-DD or full timestamp)")
    p.add_argument("--model_dir", default=os.path.join(PROJECT_ROOT, "models"), help="Directory containing model artifacts")
    p.add_argument("--transaction_cost_bps", type=float, default=1.0, help="Transaction cost in basis points")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = BacktestConfig(
        symbols=args.symbols,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        model_dir=args.model_dir,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    engine = BacktestEngine(cfg)
    results = engine.run()
    print(json.dumps(results["performance_metrics"], indent=2))


if __name__ == "__main__":
    main()