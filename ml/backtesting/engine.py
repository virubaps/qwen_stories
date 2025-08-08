import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from joblib import load
import torch

from ml.features.feature_engineering import compute_feature_frame
from ml.training.lstm_trainer import build_lstm_input, LSTMClassifier, predict_proba as lstm_predict_proba
from ml.evaluation.metrics import compute_performance_metrics

@dataclass
class BacktestConfig:
    symbols: List[str]
    data_dir: str
    start: Optional[str]
    end: Optional[str]
    model_dir: str
    transaction_cost_bps: float = 1.0

class BacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.xgb = load(os.path.join(self.config.model_dir, "xgb_model.joblib"))
        self.scaler = load(os.path.join(self.config.model_dir, "scaler.joblib"))
        self.blender = load(os.path.join(self.config.model_dir, "ensemble_blender.joblib"))
        with open(os.path.join(self.config.model_dir, "feature_cols.json"), "r") as f:
            self.feature_cols = json.load(f)
        with open(os.path.join(self.config.model_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
        self.lookback = int(meta.get("lookback", 60))
        # Initialize LSTM model
        self.lstm = None
        self._init_lstm(num_features=len(self.feature_cols))

    def _init_lstm(self, num_features: int):
        model = LSTMClassifier(num_features=num_features)
        state_path = os.path.join(self.config.model_dir, "lstm_model.pt")
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
        model.eval()
        self.lstm = model

    def _load_range(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.start:
            df = df[df["datetime"] >= pd.to_datetime(self.config.start)]
        if self.config.end:
            df = df[df["datetime"] <= pd.to_datetime(self.config.end)]
        return df

    def _decision_from_prob(self, probs: np.ndarray) -> int:
        # 0 SELL, 1 HOLD, 2 BUY
        return int(np.argmax(probs))

    def run(self) -> Dict:
        portfolio_value = [1.0]
        daily_returns = []
        trades = []
        for symbol in self.config.symbols:
            path = os.path.join(self.config.data_dir, f"{symbol}.csv") if symbol.endswith(".csv") else os.path.join(self.config.data_dir, f"{symbol}.csv")
            if not os.path.exists(path):
                path = os.path.join(self.config.data_dir, symbol)
            df = pd.read_csv(path)
            df["datetime"] = pd.to_datetime(df["datetime"]) 
            df = df.sort_values("datetime").reset_index(drop=True)
            df = self._load_range(df)

            feats = compute_feature_frame(df)
            X = feats[self.feature_cols].values
            Xs = self.scaler.transform(X)

            # Build sequences
            X_lstm, _ = build_lstm_input(Xs, np.zeros(len(Xs)), lookback=self.lookback)
            X_eff = X[self.lookback:]
            closes = feats["close"].values[self.lookback:]

            xgb_probs = self.xgb.predict_proba(X_eff)
            lstm_probs = lstm_predict_proba(self.lstm, X_lstm)
            probs = self.blender.predict(xgb_probs, lstm_probs)

            pos = 0  # -1,0,1
            prev_price = closes[0]
            for t in range(1, len(closes)):
                dec = self._decision_from_prob(probs[t-1])
                desired_pos = 1 if dec == 2 else (-1 if dec == 0 else 0)
                if desired_pos != pos:
                    portfolio_value[-1] *= (1 - self.config.transaction_cost_bps / 10000.0)
                    pos = desired_pos
                ret = (closes[t] - prev_price) / prev_price
                pnl = pos * ret
                portfolio_value.append(portfolio_value[-1] * (1 + pnl))
                daily_returns.append(pnl)
                prev_price = closes[t]

        results = {
            "trades": trades,
            "portfolio_value": [{"timestamp": i, "value": v} for i, v in enumerate(portfolio_value)],
            "daily_returns": daily_returns,
        }
        results["performance_metrics"] = compute_performance_metrics(results)
        return results