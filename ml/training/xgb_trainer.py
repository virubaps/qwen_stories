from typing import Tuple, Optional
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None


def _build_default_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )


def train_xgboost_classifier(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, use_optuna: bool = False) -> Tuple[XGBClassifier, dict]:
    if use_optuna and optuna is not None:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=3,
                tree_method="hist",
                random_state=42,
                n_jobs=4,
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)
            return -log_loss(y_val, proba)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        best_params = study.best_params
        model = XGBClassifier(
            **best_params,
            objective="multi:softprob",
            num_class=3,
            tree_method="hist",
            random_state=42,
            n_jobs=4,
        )
    else:
        model = _build_default_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)
    metrics = {
        "val_accuracy": float(accuracy_score(y_val, preds)),
        "val_log_loss": float(log_loss(y_val, proba))
    }
    return model, metrics


def predict_proba(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)