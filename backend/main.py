from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os

from backend.routers import training, backtest

app = FastAPI(title="Quant ML Trader", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

app.include_router(training.router, prefix="/api/train", tags=["training"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])