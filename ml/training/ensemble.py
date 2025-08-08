from typing import Tuple
import numpy as np
from scipy.optimize import minimize

class EnsembleBlender:
    def __init__(self):
        self.weights = np.array([0.5, 0.5], dtype=float)

    def fit(self, xgb_probs: np.ndarray, lstm_probs: np.ndarray, y_true: np.ndarray) -> None:
        def objective(w):
            w = np.clip(w, 0.0, 1.0)
            w = w / (w.sum() + 1e-12)
            blended = w[0] * xgb_probs + w[1] * lstm_probs
            # negative log likelihood
            eps = 1e-12
            ll = -np.mean(np.log(blended[np.arange(len(y_true)), y_true] + eps))
            return ll
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(np.clip(w,0,1)) - 1.0},)
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        res = minimize(objective, x0=self.weights, bounds=bounds, constraints=cons)
        if res.success:
            w = np.clip(res.x, 0.0, 1.0)
            self.weights = w / (w.sum() + 1e-12)

    def predict(self, xgb_probs: np.ndarray, lstm_probs: np.ndarray) -> np.ndarray:
        w = self.weights
        return w[0] * xgb_probs + w[1] * lstm_probs