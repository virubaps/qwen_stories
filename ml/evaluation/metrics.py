from typing import Dict, List
import numpy as np


def compute_performance_metrics(results: Dict) -> Dict:
    daily_returns = np.array(results.get("daily_returns", []), dtype=float)
    pv = [x["value"] for x in results.get("portfolio_value", [])]
    if len(pv) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "total_trades": 0
        }

    total_return = pv[-1] / pv[0] - 1.0
    # assume minute bars converted to daily? For simplicity, annualize using 252 days and 390 minutes per day
    if len(daily_returns) == 0:
        volatility = 0.0
        ann_return = 0.0
        sharpe = 0.0
    else:
        per_min_vol = np.std(daily_returns)
        volatility = per_min_vol * np.sqrt(252 * 390)
        per_min_ret = np.mean(daily_returns)
        ann_return = per_min_ret * 252 * 390
        sharpe = ann_return / (volatility + 1e-12)

    # drawdown
    pv_arr = np.array(pv)
    running_max = np.maximum.accumulate(pv_arr)
    drawdowns = (pv_arr - running_max) / (running_max + 1e-12)
    max_drawdown = float(drawdowns.min())

    # trade stats placeholders (strategy above didn't record discrete trades)
    win_rate = float((daily_returns > 0).mean()) if len(daily_returns) > 0 else 0.0
    gains = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    avg_win = float(gains.mean()) if len(gains) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    profit_factor = float(gains.sum() / (-losses.sum())) if len(losses) else float('inf')

    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_trades": 0
    }