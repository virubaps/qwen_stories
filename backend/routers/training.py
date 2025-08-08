from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from ml.training.pipeline import TrainingPipeline, TrainingConfig

router = APIRouter()

class TrainRequest(BaseModel):
    symbols: List[str]
    data_dir: str = "."
    lookback: int = 60
    horizon: int = 5
    return_threshold: float = 0.003
    model_dir: str = "models"
    use_optuna: bool = False

class TrainResponse(BaseModel):
    ok: bool
    message: str
    artifacts: dict

@router.post("")
def train(req: TrainRequest) -> TrainResponse:
    config = TrainingConfig(
        symbols=req.symbols,
        data_dir=req.data_dir,
        lookback=req.lookback,
        horizon=req.horizon,
        return_threshold=req.return_threshold,
        model_dir=req.model_dir,
        use_optuna=req.use_optuna,
    )
    pipeline = TrainingPipeline(config)
    artifacts = pipeline.run()
    return TrainResponse(ok=True, message="Training completed", artifacts=artifacts)