from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from ml.backtesting.engine import BacktestEngine, BacktestConfig

router = APIRouter()

class BacktestRequest(BaseModel):
    symbols: List[str]
    data_dir: str = "."
    start: Optional[str] = None
    end: Optional[str] = None
    model_dir: str = "models"
    transaction_cost_bps: float = 1.0

class BacktestResponse(BaseModel):
    ok: bool
    message: str
    metrics: dict

@router.post("")
def backtest(req: BacktestRequest) -> BacktestResponse:
    config = BacktestConfig(
        symbols=req.symbols,
        data_dir=req.data_dir,
        start=req.start,
        end=req.end,
        model_dir=req.model_dir,
        transaction_cost_bps=req.transaction_cost_bps,
    )
    engine = BacktestEngine(config)
    results = engine.run()
    return BacktestResponse(ok=True, message="Backtest completed", metrics=results["performance_metrics"])