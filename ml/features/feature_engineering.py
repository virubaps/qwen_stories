import numpy as np
import pandas as pd
from typing import Tuple

try:
    import ta
except Exception:  # pragma: no cover
    ta = None

WINDOW_SHORT = 12
WINDOW_LONG = 26
WINDOW_SIGNAL = 9


def _safe(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_short = _ema(series, WINDOW_SHORT)
    ema_long = _ema(series, WINDOW_LONG)
    macd = ema_short - ema_long
    signal = _ema(macd, WINDOW_SIGNAL)
    hist = macd - signal
    return macd, signal, hist


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).fillna(0).cumsum()


def _vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    return (typical * df["volume"]).rolling(window).sum() / (df["volume"].rolling(window).sum() + 1e-12)


def compute_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["datetime"] = df["datetime"]

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    # 1 RSI(14)
    out["rsi_14"] = _safe(_rsi(close, 14))

    # 2 MACD Signal
    macd_line, macd_signal, macd_hist = _macd(close)
    out["macd_signal"] = _safe(macd_signal)

    # 3 Price vs EMA20 deviation %
    ema20 = _ema(close, 20)
    out["price_ema20_dev_pct"] = _safe((close - ema20) / (ema20 + 1e-12))

    # 4 Breakout strength (Close vs 20-high)
    hh20 = high.rolling(20).max()
    out["breakout_strength"] = _safe((close - hh20) / (hh20 + 1e-12))

    # 5 RSI inverted
    out["rsi_14_inv"] = _safe(100 - out["rsi_14"])

    # 6 MACD Histogram
    out["macd_hist"] = _safe(macd_hist)

    # 7 Price vs EMA20 negative reuses 3 (model learns sign)

    # 8 Breakdown strength (Close vs 20-low)
    ll20 = low.rolling(20).min()
    out["breakdown_strength"] = _safe((close - ll20) / (ll20 + 1e-12))

    # 9 Volume ROC(5)
    out["vol_roc_5"] = _safe(volume.pct_change(5))

    # 10 OBV slope (rolling 20 slope via diff)
    obv = _obv(close, volume)
    out["obv_slope_20"] = _safe(obv.diff(20) / 20.0)

    # 11 VWAP deviation % (rolling vwap)
    vwap20 = _vwap(df, 20)
    out["vwap_dev_pct"] = _safe((close - vwap20) / (vwap20 + 1e-12))

    # 12 Volume vs 20-period average ratio
    out["vol_vs_avg20"] = _safe(volume / (volume.rolling(20).mean() + 1e-12))

    # 13 Money Flow Index (14)
    typical = (high + low + close) / 3.0
    raw_money_flow = typical * volume
    pos_flow = raw_money_flow.where(typical.diff() > 0, 0.0)
    neg_flow = raw_money_flow.where(typical.diff() < 0, 0.0)
    pmf = pos_flow.rolling(14).sum()
    nmf = neg_flow.rolling(14).sum()
    mfr = pmf / (nmf + 1e-12)
    out["mfi_14"] = _safe(100 - (100 / (1 + mfr)))

    # 14 Accum/Dist slope (20)
    clv = ((close - low) - (high - close)) / ((high - low) + 1e-12)
    adl = (clv * volume).cumsum()
    out["adl_slope_20"] = _safe(adl.diff(20) / 20.0)

    # 15 ATR percentile (lookback 100)
    atr14 = _atr(df, 14)
    atr100 = atr14.rolling(100)
    out["atr_pctile"] = _safe(atr100.apply(lambda s: (s.rank(pct=True).iloc[-1]) if len(s.dropna()) > 0 else np.nan, raw=False))

    # 16 Bollinger Band position (20,2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_dn = bb_mid - 2 * bb_std
    out["bb_pos"] = _safe((close - bb_dn) / ((bb_up - bb_dn) + 1e-12))

    # 17 VIX vs 30d avg (proxy: use symbol's realized vol vs 30d avg)
    rv_10 = close.pct_change().rolling(10).std() * np.sqrt(252)
    rv_30 = close.pct_change().rolling(30).std() * np.sqrt(252)
    out["vix_vs_30d_proxy"] = _safe(rv_10 / (rv_30 + 1e-12))

    # 18 Realized vs Implied vol ratio (implied not available -> NaN)
    out["rv_iv_ratio"] = np.nan

    # 19 Bid-Ask spread percentile (not available -> NaN)
    out["spread_pctile"] = np.nan

    # 20 Order flow imbalance (proxy using signed volume)
    signed_volume = np.sign(close.diff().fillna(0)) * volume
    out["ofi_proxy"] = _safe(signed_volume.rolling(10).mean())

    # 21 Time of day factor (minute of day normalized)
    minute_of_day = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    out["tod_factor"] = (minute_of_day - minute_of_day.min()) / (minute_of_day.max() - minute_of_day.min() + 1e-12)

    # 22 Sector relative strength (not available -> NaN)
    out["sector_rel_strength"] = np.nan

    # 23 Trend-Volume Confluence
    out["trend_vol_confluence"] = _safe(out["rsi_14" ] / 100.0 * out["vol_vs_avg20"] * (out["macd_signal"]))

    # 24 Momentum Quality = (Price_change × Volume_change) / ATR
    price_chg = close.pct_change()
    vol_chg = volume.pct_change()
    out["momentum_quality"] = _safe((price_chg * vol_chg) / (atr14 / (close + 1e-12) + 1e-12))

    # 25 Market Regime Score = Volatility_percentile × Volume_ratio × Spread_tightness (proxy 1/spread)
    spread_tightness = 1.0  # unknown -> assume neutral
    out["market_regime_score"] = _safe(out["atr_pctile"] * out["vol_vs_avg20"] * spread_tightness)

    # Carry base price for reference (not a feature)
    out["close"] = close

    return out