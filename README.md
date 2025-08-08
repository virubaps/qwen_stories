# Quant ML Trader (Qwen Stories Data)

- Backend: FastAPI (under `backend/`)
- ML: XGBoost + PyTorch LSTM + Ensemble (under `ml/`)
- Data: CSV per symbol `EXCHANGE_SYMBOL.csv` in repo root

## Setup

```
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install -r requirements.txt
```

## Run API

```
cd /workspace/qwen_stories
source /workspace/venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Train

POST /api/train
```
{
  "symbols": ["NSE_SBIN"],
  "data_dir": ".",
  "lookback": 60,
  "horizon": 5,
  "return_threshold": 0.003,
  "model_dir": "models",
  "use_optuna": false
}
```

## Backtest

POST /api/backtest
```
{
  "symbols": ["NSE_SBIN"],
  "data_dir": ".",
  "start": null,
  "end": null,
  "model_dir": "models",
  "transaction_cost_bps": 1.0
}
```