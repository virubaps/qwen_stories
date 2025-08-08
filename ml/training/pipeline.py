import os
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import torch

from ml.features.feature_engineering import compute_feature_frame
from ml.training.xgb_trainer import train_xgboost_classifier
from ml.training.lstm_trainer import train_lstm_model, build_lstm_input
from ml.training.ensemble import EnsembleBlender

@dataclass
class TrainingConfig:
    symbols: List[str]
    data_dir: str
    lookback: int
    horizon: int
    return_threshold: float
    model_dir: str
    use_optuna: bool = False

class TrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.config = config
        os.makedirs(self.config.model_dir, exist_ok=True)

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        # Expecting files like EXCHANGE_SYMBOL.csv e.g. NSE_SBIN.csv
        if symbol.endswith(".csv"):
            path = os.path.join(self.config.data_dir, symbol)
        else:
            path = os.path.join(self.config.data_dir, f"{symbol}.csv")
        if not os.path.exists(path):
            # Try using provided symbol as-is relative to data_dir
            path = os.path.join(self.config.data_dir, symbol)
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"]) 
        else:
            raise ValueError("CSV must contain 'datetime' column")
        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    def _label_from_future_return(self, close: pd.Series) -> np.ndarray:
        future = close.shift(-self.config.horizon)
        ret = (future - close) / close
        labels = np.where(ret > self.config.return_threshold, 2, np.where(ret < -self.config.return_threshold, 0, 1))
        return labels.astype(int)

    def run(self) -> Dict:
        frames: List[pd.DataFrame] = []
        for symbol in self.config.symbols:
            df = self._load_symbol(symbol)
            feats = compute_feature_frame(df)
            feats["symbol"] = symbol
            # Labels aligned to horizon
            feats["label"] = self._label_from_future_return(df["close"]).reindex(feats.index)
            frames.append(feats)
        full = pd.concat(frames, axis=0, ignore_index=True)
        full = full.dropna(subset=["label"])  # drop rows where label is nan from horizon shift
        # Feature columns exclude non-features
        non_feature_cols = {"datetime", "symbol", "label", "close"}
        feature_cols = [c for c in full.columns if c not in non_feature_cols]

        X = full[feature_cols].values
        y = full["label"].values

        # Scale for LSTM; XGBoost can use raw values
        scaler = StandardScaler()
        scaler.fit(X)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        Xs_train, Xs_val = scaler.transform(X_train), scaler.transform(X_val)

        # Train XGBoost
        xgb_model, xgb_metrics = train_xgboost_classifier(X_train, y_train, X_val, y_val, use_optuna=self.config.use_optuna)
        dump(xgb_model, os.path.join(self.config.model_dir, "xgb_model.joblib"))

        # Prepare LSTM sequences
        X_lstm_train, y_lstm_train = build_lstm_input(Xs_train, y_train, lookback=self.config.lookback)
        X_lstm_val, y_lstm_val = build_lstm_input(Xs_val, y_val, lookback=self.config.lookback)
        lstm_model, lstm_history = train_lstm_model(X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val)
        torch.save(lstm_model.state_dict(), os.path.join(self.config.model_dir, "lstm_model.pt"))

        # Save scaler and feature columns and metadata
        dump(scaler, os.path.join(self.config.model_dir, "scaler.joblib"))
        with open(os.path.join(self.config.model_dir, "feature_cols.json"), "w") as f:
            json.dump(feature_cols, f)
        with open(os.path.join(self.config.model_dir, "metadata.json"), "w") as f:
            json.dump({"lookback": self.config.lookback}, f)

        # Fit ensemble blender on validation set
        from ml.training.xgb_trainer import predict_proba as xgb_predict_proba
        from ml.training.lstm_trainer import predict_proba as lstm_predict_proba, LSTMClassifier

        xgb_val_prob = xgb_predict_proba(xgb_model, X_val)
        # build model for inference to get probabilities on val sequences
        model_inf = LSTMClassifier(num_features=X_lstm_val.shape[2])
        model_inf.load_state_dict(torch.load(os.path.join(self.config.model_dir, "lstm_model.pt"), map_location="cpu"))
        model_inf.eval()
        lstm_val_prob = lstm_predict_proba(model_inf, X_lstm_val)

        blender = EnsembleBlender()
        blender.fit(xgb_val_prob[self.config.lookback:], lstm_val_prob, y_val[self.config.lookback:])
        dump(blender, os.path.join(self.config.model_dir, "ensemble_blender.joblib"))

        return {
            "xgb_metrics": xgb_metrics,
            "lstm_history": lstm_history,
            "model_dir": self.config.model_dir,
            "num_samples": int(len(full))
        }