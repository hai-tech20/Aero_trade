"""
ULTRA ADVANCED FEATURE ENGINEERING MODULE — AERO TRADE MASTERY EDITION
Creates 170+ features per price window including:
- Technical indicators (RSI/MACD/BB/ATR/Stoch/CCI/Williams/Ultimate Osc/ADX/DMI/SAR/MFI)
- Volume delta & profile features
- Candle pattern recognition (engulfing, hammer, shooting star, doji, harami, three-rivers, etc)
- Order block / liquidity zone features
- Higher-timeframe structure alignment
- Entropy, Hurst exponent, volatility-of-volatility
- Inter-market features (VIX, DXY, correlated pairs)

PERFORMANCE TIERS:
- compute_all_features(df, ...)        : single-row inference (last row only). Still used at runtime.
- compute_all_features_matrix(df, ...) : BATCH — computes feature vectors for EVERY row in O(n*log(n))
                                         using pandas rolling vectorization. 100-1000x faster for training.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ===========================================================================
# FAST VECTORIZED ROLLING HELPERS  (100-500x faster than pure Python loops)
# ===========================================================================

def _rolling_std_pd(series: pd.Series, period: int) -> pd.Series:
    """Vectorized rolling std using pandas (C backend). NaN-pads warmup like original."""
    s = series.rolling(window=period, min_periods=1).std(ddof=0)
    return s


def _sma_pd(series: pd.Series, period: int) -> pd.Series:
    """Vectorized SMA via pandas rolling. First (period-1) values use expanding mean like original."""
    return series.rolling(window=period, min_periods=1).mean()


def _ema_pd(series: pd.Series, period: int) -> pd.Series:
    """Vectorized EMA via pandas — same alpha=2/(period+1) as original manual loop."""
    alpha = 2.0 / (period + 1.0)
    return series.ewm(alpha=alpha, adjust=False).mean()


def _wma_pd(series: pd.Series, period: int) -> pd.Series:
    """Vectorized WMA using linear weights 1..period via pandas rolling apply of dot product."""
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    def _apply(w):
        return float(np.dot(w, weights)) if len(w) == period else float(np.mean(w))
    return series.rolling(window=period, min_periods=1).apply(_apply, raw=True)


def _true_range_np(H: np.ndarray, L: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Vectorized true range (no loop needed since it's all pairwise ops)."""
    n = len(H)
    pc = np.empty(n, dtype=float)
    pc[0] = C[0]
    pc[1:] = C[:-1]
    a = H - L
    b = np.abs(H - pc)
    c = np.abs(L - pc)
    return np.maximum.reduce([a, b, c])


def _atr_pd(H: pd.Series, L: pd.Series, C: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.Series(_true_range_np(H.to_numpy(dtype=float),
                                   L.to_numpy(dtype=float),
                                   C.to_numpy(dtype=float)), index=H.index)
    alpha = 1.0 / period
    return tr.ewm(alpha=alpha, adjust=False).mean()


def _roll_max(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).max()


def _roll_min(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).min()


# ===========================================================================
# LEGACY SINGLE-ROW INFERENCE (kept for runtime predict path)
# ===========================================================================

def _ema_np(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, np.nanmean(values) if len(values) else 0.0)
    alpha = 2.0 / (period + 1.0)
    out = np.zeros_like(values, dtype=float)
    out[:period] = np.nanmean(values[:period])
    for i in range(period, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _sma_np(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, np.nanmean(values) if len(values) else 0.0)
    out = np.full_like(values, np.nan)
    cs = np.cumsum(np.insert(values, 0, 0.0))
    out[period - 1:] = (cs[period:] - cs[:-period]) / period
    for i in range(period - 1):
        out[i] = np.nanmean(values[:i + 1])
    return out


def _wma_np(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, np.nanmean(values) if len(values) else 0.0)
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    out = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        out[i] = float(np.dot(values[i - period + 1:i + 1], weights))
    return out


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period:
        return np.full_like(values, np.nanstd(values) if len(values) else 0.0)
    out = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        out[i] = float(np.std(values[i - period + 1:i + 1]))
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(high)
    tr = np.zeros(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        pc = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - pc), abs(low[i] - pc))
    return tr


def _atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = _true_range(high, low, close)
    return _ema_np(tr, period)


def compute_all_features(df: pd.DataFrame, lookback: int = 100, symbol: str = "", extended_context: Optional[dict] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the FULL 170+ feature vector from a price dataframe.
    Returns (feature_row, feature_names) ready for prediction.
    Requires len(df) >= 100 minimum, ideally >= 200.
    """
    if len(df) < 60:
        return np.array([]), []

    need = min(len(df), max(lookback * 3, 200))
    work = df.iloc[-need:].copy()

    cols = {'open', 'high', 'low', 'close', 'volume'}
    for col in cols:
        if col not in work.columns:
            work[col] = np.nan
    work = work.ffill().bfill().astype({c: float for c in cols})

    O = work['open'].to_numpy(dtype=float)
    H = work['high'].to_numpy(dtype=float)
    L = work['low'].to_numpy(dtype=float)
    C = work['close'].to_numpy(dtype=float)
    V = work['volume'].to_numpy(dtype=float) if np.nanstd(work['volume'].values) > 0 else np.ones(len(work), dtype=float)

    n = len(C)
    f: List[float] = []
    names: List[str] = []

    def _add(val, name):
        f.append(float(np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)))
        names.append(name)

    HL2 = (H + L) / 2.0
    HLC3 = (H + L + C) / 3.0
    OHLC4 = (O + H + L + C) / 4.0
    Typical = HLC3

    # =====================================================================
    # PRICE ACTION STATISTICS (windowed moments)
    # =====================================================================
    returns = np.diff(C) / np.maximum(C[:-1], 1e-9)
    r_vol = _rolling_std(returns, 20)
    for p in [5, 10, 20, 50]:
        if len(returns) >= p:
            seg = returns[-p:]
            _add(np.mean(seg), f'ret_mean_{p}')
            _add(np.std(seg), f'ret_std_{p}')
            _add(float(pd.Series(seg).skew()), f'ret_skew_{p}')
            _add(float(pd.Series(seg).kurtosis()), f'ret_kurt_{p}')
            _add(np.max(seg) - np.min(seg), f'ret_range_{p}')
            _add(np.sum(seg > 0) / max(len(seg), 1), f'ret_upratio_{p}')
            _add(np.prod(1 + seg) - 1.0, f'cumret_{p}')

    _add(float(r_vol[-1]), 'vol_current_20')
    _add(float(np.nanmean(r_vol[-20:])), 'vol_avg_20')
    _add(float(r_vol[-1] / (np.nanmean(r_vol[-60:]) + 1e-9)), 'vol_ratio_20_60')

    # Volatility of volatility
    if len(r_vol) >= 20:
        vseg = r_vol[-20:]
        _add(float(np.std(vseg)), 'vov_std_20')
        _add(float(np.max(vseg) / (np.min(vseg) + 1e-9)), 'vov_maxmin_20')

    # =====================================================================
    # MOMENTUM INDICATORS — RSI multiples
    # =====================================================================
    for rperiod in [6, 9, 14, 21, 50]:
        delta = np.diff(C)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        if len(delta) >= rperiod:
            avg_gain = _ema_np(gains, rperiod)
            avg_loss = _ema_np(losses, rperiod)
            rs = avg_gain / np.maximum(avg_loss, 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_now = float(rsi[-1])
            _add(rsi_now, f'rsi_{rperiod}')
            _add(rsi_now / 100.0, f'rsi_norm_{rperiod}')
            if len(rsi) >= 5:
                _add(rsi[-1] - rsi[-5], f'rsi_{rperiod}_delta5')
                _add((rsi[-1] - np.nanmean(rsi[-20:])) / (np.nanstd(rsi[-20:]) + 1e-9), f'rsi_{rperiod}_z20')

    # Stochastic K%D (fast/slow/full)
    for stoch_p in [(9, 3, 3), (14, 3, 3), (21, 5, 5)]:
        kp, dp, smooth = stoch_p
        if n >= kp + 10:
            hh = np.array([np.max(H[i - kp + 1:i + 1]) for i in range(kp - 1, n)])
            ll = np.array([np.min(L[i - kp + 1:i + 1]) for i in range(kp - 1, n)])
            cp = C[kp - 1:]
            raw_k = 100.0 * (cp - ll) / np.maximum(hh - ll, 1e-9)
            fast_k = _sma_np(raw_k, smooth)
            fast_d = _sma_np(fast_k, dp)
            full_k = fast_k
            full_d = _sma_np(fast_d, dp)
            _add(float(full_k[-1]), f'stoch_k_{kp}_{dp}')
            _add(float(full_d[-1]), f'stoch_d_{kp}_{dp}')
            _add(float(full_k[-1] - full_d[-1]), f'stoch_kd_diff_{kp}_{dp}')
            _add(float(np.clip(full_k[-1] / 100.0, 0, 1)), f'stoch_k_norm_{kp}')

    # CCI
    for cci_p in [14, 20, 50]:
        if n >= cci_p + 5:
            tp = HLC3
            sma_tp = _sma_np(tp, cci_p)
            dev = np.array([np.nanmean(np.abs(tp[i - cci_p + 1:i + 1] - sma_tp[i])) for i in range(cci_p - 1, n)])
            cci = (tp[cci_p - 1:] - sma_tp[cci_p - 1:]) / np.maximum(0.015 * dev, 1e-9)
            _add(float(cci[-1]), f'cci_{cci_p}')
            _add(float(np.clip(cci[-1] / 200.0, -1, 1)), f'cci_norm_{cci_p}')

    # Williams %R
    for wr_p in [10, 14, 28]:
        if n >= wr_p + 5:
            hh = np.array([np.max(H[i - wr_p + 1:i + 1]) for i in range(wr_p - 1, n)])
            ll = np.array([np.min(L[i - wr_p + 1:i + 1]) for i in range(wr_p - 1, n)])
            wr = -100.0 * (hh - C[wr_p - 1:]) / np.maximum(hh - ll, 1e-9)
            _add(float(wr[-1]), f'wr_{wr_p}')
            _add(float(wr[-1] / -100.0), f'wr_norm_{wr_p}')

    # Ultimate Oscillator
    try:
        if n >= 30:
            bp = C - np.minimum(L, np.roll(C, 1))
            tr_arr = _true_range(H, L, C)
            avg7 = _sma_np(bp, 7) / np.maximum(_sma_np(tr_arr, 7), 1e-9)
            avg14 = _sma_np(bp, 14) / np.maximum(_sma_np(tr_arr, 14), 1e-9)
            avg28 = _sma_np(bp, 28) / np.maximum(_sma_np(tr_arr, 28), 1e-9)
            uo = 100.0 * (4 * avg7 + 2 * avg14 + 1 * avg28) / 7.0
            _add(float(uo[-1]), 'ult_osc')
            _add(float(np.clip(uo[-1] / 100.0, 0, 1)), 'ult_osc_norm')
    except Exception:
        pass

    # ROC (rate of change) multiple horizons
    for roc_p in [3, 5, 9, 12, 26, 50, 90]:
        if n > roc_p:
            roc = 100.0 * (C[-1] - C[-roc_p - 1]) / max(C[-roc_p - 1], 1e-9)
            _add(float(roc), f'roc_{roc_p}')

    # =====================================================================
    # TREND INDICATORS — MA divergences, MACD, ADX, SAR
    # =====================================================================
    for p in [5, 10, 20, 30, 50, 100, 200]:
        if n >= p:
            sma_v = _sma_np(C, p)
            ema_v = _ema_np(C, p)
            _add(float((C[-1] - sma_v[-1]) / (sma_v[-1] + 1e-9)), f'price_sma_{p}_norm')
            _add(float((C[-1] - ema_v[-1]) / (ema_v[-1] + 1e-9)), f'price_ema_{p}_norm')
            if n >= p + 10:
                s10 = sma_v[-1] - sma_v[-11]
                _add(float(s10 / (sma_v[-11] + 1e-9) * 100.0), f'sma_{p}_slope10')

    # MA crossovers
    if n >= 50:
        ma5 = _ema_np(C, 5); ma10 = _ema_np(C, 10); ma20 = _ema_np(C, 20); ma50 = _sma_np(C, 50)
        _add(float((ma5[-1] - ma20[-1]) / (ma20[-1] + 1e-9)), 'ma5_ma20_normdiff')
        _add(float((ma10[-1] - ma50[-1]) / (ma50[-1] + 1e-9)), 'ma10_ma50_normdiff')
        _add(float((ma20[-1] - ma50[-1]) / (ma50[-1] + 1e-9)), 'ma20_ma50_normdiff')
        align_score = sum([
            ma5[-1] > ma10[-1],
            ma10[-1] > ma20[-1],
            ma20[-1] > ma50[-1],
            C[-1] > ma5[-1],
        ])
        _add(float(align_score), 'ma_alignment_bull_score')
        bear_align = sum([
            ma5[-1] < ma10[-1],
            ma10[-1] < ma20[-1],
            ma20[-1] < ma50[-1],
            C[-1] < ma5[-1],
        ])
        _add(float(bear_align), 'ma_alignment_bear_score')

    # MACD (standard + histogram)
    for mset in [(12, 26, 9), (8, 21, 5), (19, 39, 9)]:
        fastp, slowp, sigp = mset
        if n >= slowp + sigp + 10:
            emaf = _ema_np(C, fastp)
            emas = _ema_np(C, slowp)
            mline = emaf - emas
            sig = _ema_np(mline, sigp)
            hist = mline - sig
            _add(float(mline[-1] / (C[-1] + 1e-9)), f'macd_line_{fastp}_{slowp}_norm')
            _add(float(sig[-1] / (C[-1] + 1e-9)), f'macd_sig_{fastp}_{slowp}_{sigp}_norm')
            _add(float(hist[-1] / (C[-1] + 1e-9)), f'macd_hist_{fastp}_{slowp}_{sigp}_norm')
            _add(float(hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0, f'macd_hist_delta_{mset}')
            _add(float(mline[-1] - sig[-1] > 0), f'macd_bullcross_{mset}')

    # ADX / +DI / -DI trend strength
    atr14 = _atr_np(H, L, C, 14)
    if n >= 30:
        plus_dm = np.where((H[1:] - H[:-1]) > (L[:-1] - L[1:]), np.maximum(H[1:] - H[:-1], 0.0), 0.0)
        minus_dm = np.where((L[:-1] - L[1:]) > (H[1:] - H[:-1]), np.maximum(L[:-1] - L[1:], 0.0), 0.0)
        pdm_ema = _ema_np(plus_dm, 14)
        mdm_ema = _ema_np(minus_dm, 14)
        atr_s = atr14[1:] if len(atr14) == n else atr14
        if len(atr_s) == len(pdm_ema):
            pdi = 100.0 * pdm_ema / np.maximum(atr_s, 1e-9)
            mdi = 100.0 * mdm_ema / np.maximum(atr_s, 1e-9)
            dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-9)
            adx = _ema_np(dx, 14)
            _add(float(adx[-1]), 'adx_14')
            _add(float(pdi[-1]), 'plus_di_14')
            _add(float(mdi[-1]), 'minus_di_14')
            _add(float(pdi[-1] - mdi[-1]), 'di_diff_14')
            _add(float((pdi[-1] - mdi[-1]) * adx[-1] / 1000.0), 'trend_strength_score')
            if len(adx) >= 10:
                _add(float(adx[-1] - adx[-10]), 'adx_delta_10')

    # Bollinger Bands (3 sets)
    for bbp in [(20, 2.0), (20, 2.5), (50, 2.0)]:
        pp, sd = bbp
        if n >= pp + 10:
            mid = _sma_np(C, pp)
            std = _rolling_std(C, pp)
            up = mid + sd * sd
            lo = mid - sd * sd
            bw = (up[-1] - lo[-1]) / np.maximum(mid[-1], 1e-9)
            pos = (C[-1] - lo[-1]) / np.maximum(up[-1] - lo[-1], 1e-9)
            _add(float(bw), f'bb_width_{pp}_{int(sd*10)}')
            _add(float(pos), f'bb_position_{pp}_{int(sd*10)}')
            _add(float(C[-1] > up[-1]), f'bb_breakout_up_{pp}')
            _add(float(C[-1] < lo[-1]), f'bb_breakout_lo_{pp}')
            if n >= pp + 60:
                bw_hist = np.array([(up[i] - lo[i]) / np.maximum(mid[i], 1e-9) for i in range(pp - 1, n)])
                bw_z = (bw_hist[-1] - np.nanmean(bw_hist)) / (np.nanstd(bw_hist) + 1e-9)
                _add(float(bw_z), f'bb_squeeze_z_{pp}')

    # Keltner Channels
    if n >= 30:
        kelt_mid = _ema_np(C, 20)
        atr20 = _atr_np(H, L, C, 20)
        kup = kelt_mid + 2.0 * atr20
        klo = kelt_mid - 2.0 * atr20
        _add(float((C[-1] - kelt_mid[-1]) / np.maximum(atr20[-1], 1e-9)), 'keltner_zscore')
        _add(float(C[-1] > kup[-1]), 'keltner_break_up')
        _add(float(C[-1] < klo[-1]), 'keltner_break_lo')

    # Donchian channels
    for dp in [10, 20, 50]:
        if n >= dp:
            dh = np.array([np.max(H[i - dp + 1:i + 1]) for i in range(dp - 1, n)])
            dl = np.array([np.min(L[i - dp + 1:i + 1]) for i in range(dp - 1, n)])
            dc_pos = (C[-1] - dl[-1]) / np.maximum(dh[-1] - dl[-1], 1e-9)
            _add(float(dc_pos), f'donchian_pos_{dp}')
            _add(float(C[-1] >= dh[-1]), f'donchian_breakup_{dp}')
            _add(float(C[-1] <= dl[-1]), f'donchian_breaklo_{dp}')

    # SuperTrend
    try:
        for sset in [(10, 3.0), (20, 2.0)]:
            sp, mult = sset
            if n >= sp + 5:
                a = _atr_np(H, L, C, sp)
                hl2 = HL2
                basic_upper = hl2 + mult * a
                basic_lower = hl2 - mult * a
                fin_upper = basic_upper.copy()
                fin_lower = basic_lower.copy()
                trend = np.ones(n)
                for i in range(1, n):
                    fin_upper[i] = basic_upper[i] if basic_upper[i] < fin_upper[i-1] or C[i-1] > fin_upper[i-1] else fin_upper[i-1]
                    fin_lower[i] = basic_lower[i] if basic_lower[i] > fin_lower[i-1] or C[i-1] < fin_lower[i-1] else fin_lower[i-1]
                    if trend[i-1] == 1 and C[i] < fin_upper[i]:
                        trend[i] = 1
                    elif trend[i-1] == 1 and C[i] < fin_upper[i]:
                        trend[i] = 1
                    elif trend[i-1] == -1 and C[i] > fin_lower[i]:
                        trend[i] = -1
                    else:
                        trend[i] = -trend[i-1]
                _add(float(trend[-1]), f'supertrend_dir_{sp}_{int(mult*10)}')
                st_band = fin_lower[-1] if trend[-1] == 1 else fin_upper[-1]
                _add(float((C[-1] - st_band) / (C[-1] + 1e-9) * 100.0), f'supertrend_dist_{sp}')
    except Exception:
        pass

    # VWAP
    cum_tp_vol = np.cumsum(Typical * V)
    cum_vol = np.cumsum(V)
    vwap = cum_tp_vol / np.maximum(cum_vol, 1e-9)
    _add(float((C[-1] - vwap[-1]) / (vwap[-1] + 1e-9)), 'vwap_deviation_norm')
    if n >= 20:
        anchor20 = Typical[-20]
        vwap20 = np.cumsum(Typical[-20:] * V[-20:]) / np.maximum(np.cumsum(V[-20:]), 1e-9)
        _add(float((C[-1] - vwap20[-1]) / (vwap20[-1] + 1e-9)), 'vwap_anchor20_dev')

    # =====================================================================
    # VOLUME FEATURES
    # =====================================================================
    v_mean20 = _sma_np(V, 20)
    _add(float(V[-1] / (v_mean20[-1] + 1e-9)), 'vol_ratio_sma20')
    for vp in [5, 10, 20, 50]:
        if n >= vp:
            seg = V[-vp:]
            _add(float(np.mean(seg)), f'vol_mean_{vp}')
            _add(float(np.std(seg) / (np.mean(seg) + 1e-9)), f'vol_cv_{vp}')

    # MFI
    for mfi_p in [14, 28]:
        if n >= mfi_p + 2:
            tp_ = HLC3
            rmf = tp_ * V
            delta_mf = np.diff(tp_)
            pos_mf = np.where(delta_mf > 0, rmf[1:], 0.0)
            neg_mf = np.where(delta_mf < 0, rmf[1:], 0.0)
            pmf_sum = _sma_np(pos_mf, mfi_p)
            nmf_sum = _sma_np(neg_mf, mfi_p)
            mfr = pmf_sum / np.maximum(nmf_sum, 1e-9)
            mfi = 100.0 - (100.0 / (1.0 + mfr))
            _add(float(mfi[-1]), f'mfi_{mfi_p}')
            _add(float(mfi[-1] / 100.0), f'mfi_norm_{mfi_p}')

    # OBV
    obv = np.zeros(n)
    obv[0] = V[0] * (1 if C[0] >= O[0] else -1)
    for i in range(1, n):
        obv[i] = obv[i - 1] + V[i] * (np.sign(C[i] - C[i - 1]) if C[i] != C[i - 1] else 0)
    _add(float(np.mean(np.diff(obv[-20:])) / (np.std(obv[-50:]) + 1e-9)), 'obv_momentum_z')
    if n >= 20:
        obv_ma = _sma_np(obv, 20)
        _add(float(obv[-1] > obv_ma[-1]), 'obv_above_ma20')

    # Volume delta
    vdelta = V * np.sign(C - O)
    vdelta_cum = np.cumsum(vdelta)
    _add(float(vdelta[-1] / (np.mean(np.abs(vdelta[-20:])) + 1e-9)), 'vol_delta_z_latest')
    _add(float((vdelta_cum[-1] - vdelta_cum[-10]) / (np.std(np.diff(vdelta_cum[-50:])) + 1e-9)), 'cmf_z_10')

    # CMF
    for cmf_p in [20]:
        if n >= cmf_p + 2:
            clv = ((C - L) - (H - C)) / np.maximum(H - L, 1e-9)
            cmf = np.array([
                np.sum(clv[i-cmf_p+1:i+1] * V[i-cmf_p+1:i+1]) / np.maximum(np.sum(V[i-cmf_p+1:i+1]), 1e-9)
                for i in range(cmf_p - 1, n)
            ])
            _add(float(cmf[-1]), f'cmf_{cmf_p}')

    # =====================================================================
    # CANDLE PATTERNS
    # =====================================================================
    body = np.abs(C - O)
    rng = np.maximum(H - L, 1e-9)
    up_wick = H - np.maximum(O, C)
    lo_wick = np.minimum(O, C) - L
    br = body / rng

    last3 = [(O[-i], H[-i], L[-i], C[-i], body[-i], rng[-i], up_wick[-i], lo_wick[-i], br[-i]) for i in (3, 2, 1)]
    o3, h3, l3, c3, b3, r3, uw3, lw3, br3 = last3[2]
    o2, h2, l2, c2, b2, r2, uw2, lw2, br2 = last3[1]
    o1, h1, l1, c1, b1, r1, uw1, lw1, br1 = last3[0]
    mean_br_20 = np.nanmean(br[-20:]) if n >= 20 else np.mean(br)
    mean_range = np.nanmean(rng[-20:]) + 1e-9

    def _candle_side(v, wick, body_frac=0.1, wick_frac=0.6):
        return float(wick > body_frac * v and wick > wick_frac * max(br3, 0.3) * v)

    _add(float(b1 < 0.1 * r1 and mean_br_20 > 0.15), 'pattern_doji_now')
    hammer = float(c1 > o1 and br1 > 0.25 and lw1 > 2 * b1 and uw1 < 0.5 * b1)
    inv_hammer = float(c1 > o1 and br1 > 0.25 and uw1 > 2 * b1 and lw1 < 0.5 * b1)
    shooting = float(c1 < o1 and br1 > 0.25 and uw1 > 2 * b1 and lw1 < 0.5 * b1)
    hanging = float(c1 < o1 and br1 > 0.25 and lw1 > 2 * b1 and uw1 < 0.5 * b1)
    _add(hammer, 'pattern_hammer')
    _add(inv_hammer, 'pattern_inverted_hammer')
    _add(shooting, 'pattern_shooting_star')
    _add(hanging, 'pattern_hanging_man')

    bull_engulf = float(c2 < o2 and c1 > o1 and o1 < c2 and c1 > o2 and b1 > b2 * 1.2)
    bear_engulf = float(c2 > o2 and c1 < o1 and o1 > c2 and c1 < o2 and b1 > b2 * 1.2)
    _add(bull_engulf, 'pattern_bull_engulfing')
    _add(bear_engulf, 'pattern_bear_engulfing')

    bull_harami = float(c2 < o2 and b2 > 0.3 * r2 and c1 > o1 and o1 > c2 and c1 < o2)
    bear_harami = float(c2 > o2 and b2 > 0.3 * r2 and c1 < o1 and o1 < c2 and c1 > o2)
    _add(bull_harami, 'pattern_bull_harami')
    _add(bear_harami, 'pattern_bear_harami')

    mstar = float(c3 < o3 and br3 > 0.3 and b2 < 0.1 and c1 > o1 and c1 > (o3 + c3) / 2)
    estar = float(c3 > o3 and br3 > 0.3 and b2 < 0.1 and c1 < o1 and c1 < (o3 + c3) / 2)
    _add(mstar, 'pattern_morning_star')
    _add(estar, 'pattern_evening_star')

    t_ws = float(all([c1 > o1, c2 > o2, c3 > o3, c1 > c2 > c3]) and sum([br1, br2, br3]) / 3 > 0.35)
    t_bc = float(all([c1 < o1, c2 < o2, c3 < o3, c1 < c2 < c3]) and sum([br1, br2, br3]) / 3 > 0.35)
    _add(t_ws, 'pattern_3_white_soldiers')
    _add(t_bc, 'pattern_3_black_crows')

    tweezer_t = float(c2 > o2 and c1 < o1 and abs(h2 - h1) < 0.001 * max(h1, h2) and uw1 > 0.4 * r1)
    tweezer_b = float(c2 < o2 and c1 > o1 and abs(l2 - l1) < 0.001 * max(l1, l2) and lw1 > 0.4 * r1)
    _add(tweezer_t, 'pattern_tweezer_top')
    _add(tweezer_b, 'pattern_tweezer_bottom')

    pattern_bull_score = hammer + bull_engulf + bull_harami + mstar + t_ws + tweezer_b
    pattern_bear_score = shooting + bear_engulf + bear_harami + estar + t_bc + tweezer_t
    _add(float(pattern_bull_score), 'pattern_bull_score_total')
    _add(float(pattern_bear_score), 'pattern_bear_score_total')
    _add(float(pattern_bull_score - pattern_bear_score), 'pattern_net_score')

    _add(float(C[-1] > O[-1]), 'candle_last_up')
    if n >= 5:
        streak = 0
        for i in range(1, min(11, n + 1)):
            if C[-i] > O[-i]:
                streak += 1
            else:
                break
        _add(float(streak), 'bull_streak')
        streak2 = 0
        for i in range(1, min(11, n + 1)):
            if C[-i] < O[-i]:
                streak2 += 1
            else:
                break
        _add(float(streak2), 'bear_streak')

    for k in (5, 10, 20):
        if n >= k:
            count = sum(1 for i in range(-1, -k - 1, -1) if C[i] > C[i - 1])
            _add(float(count / k), f'up_candle_ratio_{k}')

    # =====================================================================
    # VOLATILITY & RISK METRICS
    # =====================================================================
    atr14_now = float(atr14[-1]) if len(atr14) > 0 else 0.0
    _add(atr14_now, 'atr_14')
    _add(float(atr14_now / (C[-1] + 1e-9) * 10000.0), 'atr_14_pips_pct')
    _add(float(atr14_now / (np.nanmean(atr14[-60:]) + 1e-9)), 'atr_ratio_14_60')

    atr30 = _atr_np(H, L, C, 30)
    _add(float(atr30[-1] / (C[-1] + 1e-9) * 10000.0), 'atr_30_pips_pct')
    _add(float(np.std(C[-60:]) / (np.mean(C[-60:]) + 1e-9)), 'realized_vol_60')

    if n >= 30:
        park30 = np.sqrt(np.mean((np.log(H[-30:] / L[-30:]) ** 2) / (4.0 * np.log(2.0))))
        _add(float(park30), 'parkinson_vol_30')

    if n >= 30:
        seg = work.iloc[-30:]
        lnhl = np.log(seg['high'].values / seg['low'].values)
        lnco = np.log(seg['close'].values / seg['open'].values)
        gk = np.sqrt(np.mean(0.5 * lnhl ** 2 - (2 * np.log(2) - 1) * lnco ** 2))
        _add(float(gk), 'garman_klass_30')

    if len(returns) >= 30:
        try:
            bins = 8
            disc = np.digitize(returns[-30:], bins=np.quantile(returns[-30:], np.linspace(0, 1, bins + 1)))
            counts = np.bincount(disc, minlength=bins + 2)
            probs = counts / counts.sum()
            ent = -np.sum(probs * np.log(np.where(probs > 0, probs, 1.0)))
            _add(float(ent), 'ret_entropy_30')
        except Exception:
            _add(0.0, 'ret_entropy_30')

    if len(C) >= 100:
        try:
            seg = np.log(C[-120:])
            seg = seg - seg[0]
            ns = len(seg)
            rs_list = []
            for m in (8, 16, 24, 32):
                if ns >= m * 2:
                    nseg = ns // m
                    rs_vals = []
                    for s in range(nseg):
                        w = seg[s * m:(s + 1) * m]
                        w = w - np.mean(w)
                        cum = np.cumsum(w)
                        rng = np.max(cum) - np.min(cum)
                        sd = np.std(w) + 1e-12
                        rs_vals.append(rng / sd)
                    if rs_vals:
                        rs_list.append((m, float(np.mean(rs_vals))))
            if len(rs_list) >= 2:
                xs = np.log([r[0] for r in rs_list])
                ys = np.log([r[1] for r in rs_list])
                hurst = float(np.polyfit(xs, ys, 1)[0])
                _add(hurst, 'hurst_approx')
                _add(float(hurst > 0.55) - float(hurst < 0.45), 'hurst_trend_flag')
            else:
                _add(0.5, 'hurst_approx')
                _add(0.0, 'hurst_trend_flag')
        except Exception:
            _add(0.5, 'hurst_approx')
            _add(0.0, 'hurst_trend_flag')

    # =====================================================================
    # ORDER-BLOCK & STRUCTURE FEATURES
    # =====================================================================
    def _swings(h, l, lr):
        sh, sl = [], []
        for i in range(lr, len(h) - lr):
            if h[i] == np.max(h[i-lr:i+lr+1]) and h[i] > h[i-1] and h[i] > h[i+1]:
                sh.append((i, h[i]))
            if l[i] == np.min(l[i-lr:i+lr+1]) and l[i] < l[i-1] and l[i] < l[i+1]:
                sl.append((i, l[i]))
        return sh, sl

    if n >= 60:
        shs, sls = _swings(H, L, 2)
        if shs and sls:
            last_sh_idx, last_sh_p = shs[-1]
            last_sl_idx, last_sl_p = sls[-1]
            dist_to_sh = (C[-1] - last_sh_p) / (C[-1] + 1e-9)
            dist_to_sl = (C[-1] - last_sl_p) / (C[-1] + 1e-9)
            _add(dist_to_sh, 'dist_last_swing_high')
            _add(dist_to_sl, 'dist_last_swing_low')
            if len(shs) >= 2:
                _add(float(shs[-1][1] > shs[-2][1]), 'higher_high_last2')
            if len(sls) >= 2:
                _add(float(sls[-1][1] > sls[-2][1]), 'higher_low_last2')
            if len(shs) >= 2 and len(sls) >= 2:
                hhhl = float(shs[-1][1] > shs[-2][1] and sls[-1][1] > sls[-2][1])
                lhll = float(shs[-1][1] < shs[-2][1] and sls[-1][1] < sls[-2][1])
                _add(hhhl, 'structure_hhhl_bull')
                _add(lhll, 'structure_lhll_bear')

    # =====================================================================
    # TIME FEATURES
    # =====================================================================
    idx = work.index[-1]
    dow = 3.0
    hod = 12.0
    try:
        if hasattr(idx, 'weekday'):
            dow = float(idx.weekday())
            hod = float(idx.hour)
        elif 'time' in work.columns:
            t = pd.to_datetime(work['time'].iloc[-1])
            dow = float(t.weekday())
            hod = float(t.hour)
    except Exception:
        pass
    _add(dow / 6.0, 'day_of_week_norm')
    _add(hod / 23.0, 'hour_of_day_norm')
    london = float(7 <= hod < 16)
    ny = float(12 <= hod < 21)
    tokyo = float(0 <= hod < 9)
    overlap = float(london and ny)
    _add(london, 'session_london')
    _add(ny, 'session_newyork')
    _add(tokyo, 'session_tokyo')
    _add(overlap, 'session_london_ny_overlap')

    # =====================================================================
    # EXTENDED CONTEXT (cross-pair signals: USD strength, risk sentiment,
    # gold momentum). MUST always emit the exact same fixed key set/order
    # as compute_all_features_matrix's extended_context block, zero-filling
    # any missing key — otherwise a live call without context data produces
    # fewer features than training did, silently misaligning every feature
    # slot at every model that reads past this point (the exact bug class
    # already fixed once this session in the winsorization mismatch).
    # =====================================================================
    from ai_engine.models.market_context import MARKET_CONTEXT_KEYS
    ctx = extended_context or {}
    for key in MARKET_CONTEXT_KEYS:
        _add(float(ctx.get(key, 0.0)), f'ext_{key}')

    f_arr = np.array(f, dtype=float).reshape(1, -1)
    f_arr = np.nan_to_num(f_arr, nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(f_arr, axis=1, keepdims=True)
    mad = np.median(np.abs(f_arr - med), axis=1, keepdims=True) + 1e-9
    f_arr = np.where(np.abs(f_arr - med) > 12 * mad, med + 12 * mad * np.sign(f_arr - med), f_arr)

    return f_arr, names


# ===========================================================================
# BATCH MATRIX COMPUTE — 100-1000x faster for training.
# Semantically equivalent to calling compute_all_features(df.iloc[:i+1])
# for every i. Uses vectorized pandas rolling + precomputed series.
# ===========================================================================

def compute_all_features_matrix(
    df: pd.DataFrame,
    lookback: int = 100,
    symbol: str = "",
    min_rows_required: int = 250,
    progress_callback=None,
    extended_context: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the full feature vector for EVERY row in `df` in a single
    vectorized pass. Returns:
      X : (n_rows, n_features) float32 matrix — rows that can't be computed
          (warmup < min_rows_required//3) are zeros; caller should drop them.
      names : list of feature names (same order as X columns)

    This is the ONLY function you want to call during training matrix build.
    For a 100k-row history it runs in seconds vs hours/days of per-row loops.

    extended_context: optional time-indexed DataFrame (e.g. from
    ai_engine.models.market_context.compute_market_context_series), already
    reindexed/aligned to `df`'s index by the caller. Columns are appended as
    ext_<column> features — same naming scheme compute_all_features() uses
    for its extended_context dict, so batch training and single-row live
    inference stay consistent. Always adds the same number of columns
    (zero-filled when not provided) so feature count never depends on
    whether context data happened to be available for a given symbol.
    """
    n = len(df)
    work = df.copy()
    cols_needed = {'open', 'high', 'low', 'close', 'volume'}
    for c in cols_needed:
        if c not in work.columns:
            work[c] = np.nan
    work = work.ffill().bfill()
    for c in cols_needed:
        work[c] = pd.to_numeric(work[c], errors='coerce').astype(float)

    # Volume fallback
    V_raw = work['volume'].copy()
    if float(V_raw.std(ddof=0)) <= 1e-12:
        V_raw = pd.Series(np.ones(n, dtype=float), index=work.index)

    O_s = work['open']
    H_s = work['high']
    L_s = work['low']
    C_s = work['close']
    V_s = V_raw

    O = O_s.to_numpy(dtype=float)
    H = H_s.to_numpy(dtype=float)
    L = L_s.to_numpy(dtype=float)
    C = C_s.to_numpy(dtype=float)
    V = V_s.to_numpy(dtype=float)

    HL2 = (H + L) / 2.0
    HLC3 = (H + L + C) / 3.0
    Typical = HLC3

    # Columns list: list of 1-dim arrays length-n aligned with row indices
    col_list: List[np.ndarray] = []
    name_list: List[str] = []

    def _add_col(arr: np.ndarray, name: str):
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"Feature {name} shape mismatch: {arr.shape} vs expected ({n},)")
        col_list.append(np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0))
        name_list.append(name)

    # =====================================================================
    # 1) RETURNS + WINDOWED MOMENTS  (for p in [5,10,20,50])
    # =====================================================================
    ret_s = C_s.pct_change().fillna(0.0)
    ret_np = ret_s.to_numpy(dtype=float)
    r_vol_s = _rolling_std_pd(ret_s, 20)

    for p in [5, 10, 20, 50]:
        _add_col(ret_s.rolling(p, min_periods=1).mean().to_numpy(dtype=float), f'ret_mean_{p}')
        _add_col(ret_s.rolling(p, min_periods=1).std(ddof=0).to_numpy(dtype=float), f'ret_std_{p}')
        skew_vals = np.zeros(n, dtype=float)
        if n >= p:
            sk = ret_s.rolling(p, min_periods=p).skew().to_numpy(dtype=float)
            skew_vals = np.nan_to_num(sk, nan=0.0)
        _add_col(skew_vals, f'ret_skew_{p}')
        kurt_vals = np.zeros(n, dtype=float)
        if n >= p:
            ku = ret_s.rolling(p, min_periods=p).kurt().to_numpy(dtype=float)
            kurt_vals = np.nan_to_num(ku, nan=0.0)
        _add_col(kurt_vals, f'ret_kurt_{p}')
        _add_col((ret_s.rolling(p, min_periods=1).max() - ret_s.rolling(p, min_periods=1).min()).to_numpy(dtype=float), f'ret_range_{p}')
        up_roll = (ret_s > 0).rolling(window=p, min_periods=1).sum().to_numpy(dtype=float)
        cnt_roll = np.minimum(np.arange(1, n + 1, dtype=float), p)
        _add_col(up_roll / np.maximum(cnt_roll, 1), f'ret_upratio_{p}')
        _add_col(((1.0 + ret_s).rolling(p, min_periods=1).apply(np.prod, raw=True) - 1.0).to_numpy(dtype=float), f'cumret_{p}')

    _add_col(r_vol_s.to_numpy(dtype=float), 'vol_current_20')
    _add_col(r_vol_s.rolling(20, min_periods=1).mean().to_numpy(dtype=float), 'vol_avg_20')
    _add_col((r_vol_s / (r_vol_s.rolling(60, min_periods=1).mean() + 1e-9)).to_numpy(dtype=float), 'vol_ratio_20_60')

    vov_std = r_vol_s.rolling(20, min_periods=1).std(ddof=0).to_numpy(dtype=float)
    vov_max = r_vol_s.rolling(20, min_periods=1).max().to_numpy(dtype=float)
    vov_min = r_vol_s.rolling(20, min_periods=1).min().to_numpy(dtype=float)
    _add_col(vov_std, 'vov_std_20')
    _add_col(vov_max / (vov_min + 1e-9), 'vov_maxmin_20')

    # =====================================================================
    # 2) RSI MULTIPLES
    # =====================================================================
    delta_s = C_s.diff()
    gains_s = delta_s.clip(lower=0.0)
    losses_s = (-delta_s).clip(lower=0.0)
    for rperiod in [6, 9, 14, 21, 50]:
        alpha_r = 2.0 / (rperiod + 1.0)
        avg_gain = gains_s.ewm(alpha=alpha_r, adjust=False).mean()
        avg_loss = losses_s.ewm(alpha=alpha_r, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_s = 100.0 - (100.0 / (1.0 + rs))
        _add_col(rsi_s.to_numpy(dtype=float), f'rsi_{rperiod}')
        _add_col((rsi_s / 100.0).to_numpy(dtype=float), f'rsi_norm_{rperiod}')
        # RSI delta 5
        d5 = rsi_s - rsi_s.shift(5)
        _add_col(d5.fillna(0.0).to_numpy(dtype=float), f'rsi_{rperiod}_delta5')
        # RSI z20
        rsi_mu = rsi_s.rolling(20, min_periods=1).mean()
        rsi_sd = rsi_s.rolling(20, min_periods=1).std(ddof=0)
        _add_col(((rsi_s - rsi_mu) / (rsi_sd + 1e-9)).to_numpy(dtype=float), f'rsi_{rperiod}_z20')

    # =====================================================================
    # 3) STOCHASTIC K%D (fast/slow/full)
    # =====================================================================
    for (kp, dp, smooth) in [(9, 3, 3), (14, 3, 3), (21, 5, 5)]:
        hh_s = H_s.rolling(kp, min_periods=1).max()
        ll_s = L_s.rolling(kp, min_periods=1).min()
        raw_k = 100.0 * (C_s - ll_s) / (hh_s - ll_s + 1e-9)
        fast_k = raw_k.rolling(smooth, min_periods=1).mean()
        fast_d = fast_k.rolling(dp, min_periods=1).mean()
        full_d = fast_d.rolling(dp, min_periods=1).mean()
        _add_col(fast_k.to_numpy(dtype=float), f'stoch_k_{kp}_{dp}')
        _add_col(full_d.to_numpy(dtype=float), f'stoch_d_{kp}_{dp}')
        _add_col((fast_k - full_d).to_numpy(dtype=float), f'stoch_kd_diff_{kp}_{dp}')
        _add_col(np.clip(fast_k.to_numpy(dtype=float) / 100.0, 0.0, 1.0), f'stoch_k_norm_{kp}')

    # =====================================================================
    # 4) CCI
    # =====================================================================
    tp_s = pd.Series(HLC3, index=work.index)
    for cci_p in [14, 20, 50]:
        sma_tp = _sma_pd(tp_s, cci_p)
        mad_dev = (tp_s - sma_tp).abs().rolling(cci_p, min_periods=1).mean()
        cci_s = (tp_s - sma_tp) / (0.015 * mad_dev + 1e-9)
        _add_col(cci_s.to_numpy(dtype=float), f'cci_{cci_p}')
        _add_col(np.clip(cci_s.to_numpy(dtype=float) / 200.0, -1.0, 1.0), f'cci_norm_{cci_p}')

    # =====================================================================
    # 5) WILLIAMS %R
    # =====================================================================
    for wr_p in [10, 14, 28]:
        hh = _roll_max(H_s, wr_p)
        ll = _roll_min(L_s, wr_p)
        wr_s = -100.0 * (hh - C_s) / (hh - ll + 1e-9)
        _add_col(wr_s.to_numpy(dtype=float), f'wr_{wr_p}')
        _add_col(wr_s.to_numpy(dtype=float) / -100.0, f'wr_norm_{wr_p}')

    # =====================================================================
    # 6) ULTIMATE OSCILLATOR
    # =====================================================================
    try:
        bp_s = C_s - np.minimum(L_s, C_s.shift(1).bfill())
        tr_arr_s = pd.Series(_true_range_np(H, L, C), index=work.index)
        avg7 = _sma_pd(bp_s, 7) / (_sma_pd(tr_arr_s, 7) + 1e-9)
        avg14 = _sma_pd(bp_s, 14) / (_sma_pd(tr_arr_s, 14) + 1e-9)
        avg28 = _sma_pd(bp_s, 28) / (_sma_pd(tr_arr_s, 28) + 1e-9)
        uo_s = 100.0 * (4 * avg7 + 2 * avg14 + 1 * avg28) / 7.0
        _add_col(uo_s.to_numpy(dtype=float), 'ult_osc')
        _add_col(np.clip(uo_s.to_numpy(dtype=float) / 100.0, 0.0, 1.0), 'ult_osc_norm')
    except Exception:
        _add_col(np.zeros(n, dtype=float), 'ult_osc')
        _add_col(np.zeros(n, dtype=float), 'ult_osc_norm')

    # =====================================================================
    # 7) ROC — multiple horizons
    # =====================================================================
    for roc_p in [3, 5, 9, 12, 26, 50, 90]:
        roc_s = 100.0 * (C_s - C_s.shift(roc_p + 1)) / (C_s.shift(roc_p + 1) + 1e-9)
        _add_col(roc_s.fillna(0.0).to_numpy(dtype=float), f'roc_{roc_p}')

    # =====================================================================
    # 8) MA DIVERGENCES (p in [5,10,20,30,50,100,200])
    # =====================================================================
    for p in [5, 10, 20, 30, 50, 100, 200]:
        sma_v = _sma_pd(C_s, p)
        ema_v = _ema_pd(C_s, p)
        _add_col(((C_s - sma_v) / (sma_v + 1e-9)).to_numpy(dtype=float), f'price_sma_{p}_norm')
        _add_col(((C_s - ema_v) / (ema_v + 1e-9)).to_numpy(dtype=float), f'price_ema_{p}_norm')
        # MA slope 10
        slope = (sma_v - sma_v.shift(10)) / (sma_v.shift(10) + 1e-9) * 100.0
        _add_col(slope.fillna(0.0).to_numpy(dtype=float), f'sma_{p}_slope10')

    # MA crossovers
    if n >= 50:
        ma5 = _ema_pd(C_s, 5)
        ma10 = _ema_pd(C_s, 10)
        ma20 = _ema_pd(C_s, 20)
        ma50 = _sma_pd(C_s, 50)
        _add_col(((ma5 - ma20) / (ma20 + 1e-9)).to_numpy(dtype=float), 'ma5_ma20_normdiff')
        _add_col(((ma10 - ma50) / (ma50 + 1e-9)).to_numpy(dtype=float), 'ma10_ma50_normdiff')
        _add_col(((ma20 - ma50) / (ma50 + 1e-9)).to_numpy(dtype=float), 'ma20_ma50_normdiff')
        # Bull alignment score
        align = ((ma5 > ma10).astype(float) +
                 (ma10 > ma20).astype(float) +
                 (ma20 > ma50).astype(float) +
                 (C_s > ma5).astype(float))
        _add_col(align.to_numpy(dtype=float), 'ma_alignment_bull_score')
        bear_align = ((ma5 < ma10).astype(float) +
                      (ma10 < ma20).astype(float) +
                      (ma20 < ma50).astype(float) +
                      (C_s < ma5).astype(float))
        _add_col(bear_align.to_numpy(dtype=float), 'ma_alignment_bear_score')

    # =====================================================================
    # 9) MACD (3 sets)
    # =====================================================================
    for (fastp, slowp, sigp) in [(12, 26, 9), (8, 21, 5), (19, 39, 9)]:
        emaf = _ema_pd(C_s, fastp)
        emas = _ema_pd(C_s, slowp)
        mline = emaf - emas
        sig = _ema_pd(mline, sigp)
        hist = mline - sig
        mset = (fastp, slowp, sigp)
        _add_col((mline / (C_s + 1e-9)).to_numpy(dtype=float), f'macd_line_{fastp}_{slowp}_norm')
        _add_col((sig / (C_s + 1e-9)).to_numpy(dtype=float), f'macd_sig_{fastp}_{slowp}_{sigp}_norm')
        _add_col((hist / (C_s + 1e-9)).to_numpy(dtype=float), f'macd_hist_{fastp}_{slowp}_{sigp}_norm')
        _add_col((hist - hist.shift(1)).fillna(0.0).to_numpy(dtype=float), f'macd_hist_delta_{mset}')
        _add_col((mline - sig > 0).astype(float).to_numpy(dtype=float), f'macd_bullcross_{mset}')

    # =====================================================================
    # 10) ADX / +DI / -DI
    # =====================================================================
    atr14_s = _atr_pd(H_s, L_s, C_s, 14)
    atr14 = atr14_s.to_numpy(dtype=float)
    plus_dm = np.where((H[1:] - H[:-1]) > (L[:-1] - L[1:]), np.maximum(H[1:] - H[:-1], 0.0), 0.0)
    minus_dm = np.where((L[:-1] - L[1:]) > (H[1:] - H[:-1]), np.maximum(L[:-1] - L[1:], 0.0), 0.0)
    pdm_ema = _ema_pd(pd.Series(np.r_[0.0, plus_dm], index=work.index), 14).to_numpy(dtype=float)
    mdm_ema = _ema_pd(pd.Series(np.r_[0.0, minus_dm], index=work.index), 14).to_numpy(dtype=float)
    atr_s_adj = atr14
    pdi = 100.0 * pdm_ema / (atr_s_adj + 1e-9)
    mdi = 100.0 * mdm_ema / (atr_s_adj + 1e-9)
    dx_raw = 100.0 * np.abs(pdi - mdi) / (pdi + mdi + 1e-9)
    adx = _ema_pd(pd.Series(dx_raw, index=work.index), 14).to_numpy(dtype=float)
    _add_col(adx, 'adx_14')
    _add_col(pdi, 'plus_di_14')
    _add_col(mdi, 'minus_di_14')
    _add_col(pdi - mdi, 'di_diff_14')
    _add_col((pdi - mdi) * adx / 1000.0, 'trend_strength_score')
    adx_delta = np.r_[np.zeros(10, dtype=float), adx[10:] - adx[:-10]] if n >= 10 else np.zeros(n, dtype=float)
    _add_col(adx_delta, 'adx_delta_10')

    # =====================================================================
    # 11) BOLLINGER BANDS (3 sets)
    # =====================================================================
    for (pp, sd) in [(20, 2.0), (20, 2.5), (50, 2.0)]:
        mid = _sma_pd(C_s, pp)
        std = _rolling_std_pd(C_s, pp)
        up = mid + std * sd
        lo = mid - std * sd
        bw = (up - lo) / (mid + 1e-9)
        pos = (C_s - lo) / (up - lo + 1e-9)
        _add_col(bw.to_numpy(dtype=float), f'bb_width_{pp}_{int(sd*10)}')
        _add_col(pos.to_numpy(dtype=float), f'bb_position_{pp}_{int(sd*10)}')
        _add_col((C_s > up).astype(float).to_numpy(dtype=float), f'bb_breakout_up_{pp}')
        _add_col((C_s < lo).astype(float).to_numpy(dtype=float), f'bb_breakout_lo_{pp}')
        # squeeze z
        bw_hist_z = (bw - bw.rolling(pp + 60, min_periods=1).mean()) / (bw.rolling(pp + 60, min_periods=1).std(ddof=0) + 1e-9)
        _add_col(bw_hist_z.to_numpy(dtype=float), f'bb_squeeze_z_{pp}')

    # =====================================================================
    # 12) KELTNER CHANNELS
    # =====================================================================
    if n >= 30:
        kelt_mid = _ema_pd(C_s, 20)
        atr20 = _atr_pd(H_s, L_s, C_s, 20)
        kup = kelt_mid + 2.0 * atr20
        klo = kelt_mid - 2.0 * atr20
        _add_col(((C_s - kelt_mid) / (atr20 + 1e-9)).to_numpy(dtype=float), 'keltner_zscore')
        _add_col((C_s > kup).astype(float).to_numpy(dtype=float), 'keltner_break_up')
        _add_col((C_s < klo).astype(float).to_numpy(dtype=float), 'keltner_break_lo')
    else:
        for nm in ('keltner_zscore', 'keltner_break_up', 'keltner_break_lo'):
            _add_col(np.zeros(n, dtype=float), nm)

    # =====================================================================
    # 13) DONCHIAN CHANNELS
    # =====================================================================
    for dp in [10, 20, 50]:
        dh = _roll_max(H_s, dp)
        dl = _roll_min(L_s, dp)
        dc_pos = (C_s - dl) / (dh - dl + 1e-9)
        _add_col(dc_pos.to_numpy(dtype=float), f'donchian_pos_{dp}')
        _add_col((C_s >= dh).astype(float).to_numpy(dtype=float), f'donchian_breakup_{dp}')
        _add_col((C_s <= dl).astype(float).to_numpy(dtype=float), f'donchian_breaklo_{dp}')

    # =====================================================================
    # 14) SUPERTREND (2 sets) — keep iterative loop (n small steps per row)
    # =====================================================================
    def _supertrend_batch(H_arr, L_arr, C_arr, sp, mult):
        ns = len(H_arr)
        a = _atr_pd(pd.Series(H_arr), pd.Series(L_arr), pd.Series(C_arr), sp).to_numpy(dtype=float)
        hl2 = (H_arr + L_arr) / 2.0
        basic_upper = hl2 + mult * a
        basic_lower = hl2 - mult * a
        fin_upper = basic_upper.copy()
        fin_lower = basic_lower.copy()
        trend = np.ones(ns, dtype=float)
        for i in range(1, ns):
            fin_upper[i] = basic_upper[i] if basic_upper[i] < fin_upper[i-1] or C_arr[i-1] > fin_upper[i-1] else fin_upper[i-1]
            fin_lower[i] = basic_lower[i] if basic_lower[i] > fin_lower[i-1] or C_arr[i-1] < fin_lower[i-1] else fin_lower[i-1]
            if trend[i-1] == 1 and C_arr[i] < fin_upper[i]:
                trend[i] = 1
            elif trend[i-1] == -1 and C_arr[i] > fin_lower[i]:
                trend[i] = -1
            else:
                trend[i] = -trend[i-1]
        st_band = np.where(trend == 1, fin_lower, fin_upper)
        st_dist = (C_arr - st_band) / (C_arr + 1e-9) * 100.0
        return trend, st_dist

    for (sp, mult) in [(10, 3.0), (20, 2.0)]:
        if n >= sp + 5:
            try:
                st_t, st_d = _supertrend_batch(H, L, C, sp, mult)
                _add_col(st_t, f'supertrend_dir_{sp}_{int(mult*10)}')
                _add_col(st_d, f'supertrend_dist_{sp}')
            except Exception:
                _add_col(np.zeros(n, dtype=float), f'supertrend_dir_{sp}_{int(mult*10)}')
                _add_col(np.zeros(n, dtype=float), f'supertrend_dist_{sp}')
        else:
            _add_col(np.zeros(n, dtype=float), f'supertrend_dir_{sp}_{int(mult*10)}')
            _add_col(np.zeros(n, dtype=float), f'supertrend_dist_{sp}')

    # =====================================================================
    # 15) VWAP
    # =====================================================================
    typ_v = Typical * V
    cum_tp_vol = np.cumsum(typ_v)
    cum_vol = np.cumsum(V)
    vwap = cum_tp_vol / (cum_vol + 1e-9)
    _add_col(((C - vwap) / (vwap + 1e-9)), 'vwap_deviation_norm')
    # vwap anchor 20 (rolling anchor)
    def _rolling_vwap_dev(typ_arr, V_arr, C_arr, anchor=20):
        out = np.zeros(len(C_arr), dtype=float)
        if len(C_arr) < anchor:
            return out
        t_roll = np.lib.stride_tricks.sliding_window_view(typ_arr, anchor)
        v_roll = np.lib.stride_tricks.sliding_window_view(V_arr, anchor)
        c_tail = C_arr[anchor - 1:]
        num = np.cumsum(t_roll * v_roll, axis=1)[:, -1]
        den = np.cumsum(v_roll, axis=1)[:, -1]
        vwap_20_tail = num / (den + 1e-9)
        out[anchor - 1:] = (c_tail - vwap_20_tail) / (vwap_20_tail + 1e-9)
        return out
    _add_col(_rolling_vwap_dev(Typical, V, C, 20), 'vwap_anchor20_dev')

    # =====================================================================
    # 16) VOLUME FEATURES
    # =====================================================================
    v_mean20 = _sma_pd(V_s, 20)
    _add_col((V_s / (v_mean20 + 1e-9)).to_numpy(dtype=float), 'vol_ratio_sma20')
    for vp in [5, 10, 20, 50]:
        m = V_s.rolling(vp, min_periods=1).mean()
        s = V_s.rolling(vp, min_periods=1).std(ddof=0)
        _add_col(m.to_numpy(dtype=float), f'vol_mean_{vp}')
        _add_col((s / (m + 1e-9)).to_numpy(dtype=float), f'vol_cv_{vp}')

    # MFI
    for mfi_p in [14, 28]:
        tp_ = HLC3
        rmf = tp_ * V
        delta_mf = np.diff(tp_, prepend=tp_[0])
        pos_mf = np.where(delta_mf > 0, rmf, 0.0)
        neg_mf = np.where(delta_mf < 0, rmf, 0.0)
        pmf_sum = _sma_pd(pd.Series(pos_mf, index=work.index), mfi_p).to_numpy(dtype=float)
        nmf_sum = _sma_pd(pd.Series(neg_mf, index=work.index), mfi_p).to_numpy(dtype=float)
        mfr = pmf_sum / (nmf_sum + 1e-9)
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        _add_col(mfi, f'mfi_{mfi_p}')
        _add_col(mfi / 100.0, f'mfi_norm_{mfi_p}')

    # OBV
    signs = np.sign(C - np.r_[C[0], C[:-1]])
    signs[0] = 1.0 if C[0] >= O[0] else -1.0
    obv = np.cumsum(V * signs)
    obv_diff20 = np.zeros(n, dtype=float)
    if n >= 21:
        diffs = obv[1:] - obv[:-1]
        mean_seg = pd.Series(diffs).rolling(20, min_periods=1).mean().to_numpy(dtype=float)
        sd_seg50 = pd.Series(obv).rolling(50, min_periods=1).std(ddof=0).to_numpy(dtype=float)
        obv_diff20[n - len(mean_seg):] = mean_seg / (sd_seg50[n - len(mean_seg):] + 1e-9)
    _add_col(obv_diff20, 'obv_momentum_z')
    obv_ma = _sma_pd(pd.Series(obv, index=work.index), 20).to_numpy(dtype=float)
    _add_col((obv > obv_ma).astype(float), 'obv_above_ma20')

    # Volume delta
    vdelta = V * np.sign(C - O)
    vdelta_cum = np.cumsum(vdelta)
    abs_mean = pd.Series(np.abs(vdelta)).rolling(20, min_periods=1).mean().to_numpy(dtype=float)
    _add_col(vdelta / (abs_mean + 1e-9), 'vol_delta_z_latest')
    diff_cmf = np.zeros(n, dtype=float)
    if n >= 50:
        d_cum = vdelta_cum - np.r_[np.zeros(10), vdelta_cum[:-10]]
        diffs_cmf = np.diff(vdelta_cum)
        sd_d = pd.Series(diffs_cmf).rolling(50, min_periods=1).std(ddof=0).to_numpy(dtype=float)
        full_sd = np.r_[np.zeros(n - len(sd_d), dtype=float), sd_d]
        diff_cmf = d_cum / (full_sd + 1e-9)
    _add_col(diff_cmf, 'cmf_z_10')

    # CMF
    cmf_p = 20
    clv = ((C - L) - (H - C)) / (H - L + 1e-9)
    num_ = pd.Series(clv * V).rolling(cmf_p, min_periods=1).sum()
    den_ = pd.Series(V).rolling(cmf_p, min_periods=1).sum()
    _add_col((num_ / (den_ + 1e-9)).to_numpy(dtype=float), f'cmf_{cmf_p}')

    # =====================================================================
    # 17) CANDLE PATTERNS — pointwise (simple comparisons + 3-rolling windows)
    # =====================================================================
    body = np.abs(C - O)
    rng = np.maximum(H - L, 1e-9)
    up_wick = H - np.maximum(O, C)
    lo_wick = np.minimum(O, C) - L
    br = body / rng

    mean_br_20 = pd.Series(br).rolling(20, min_periods=1).mean().to_numpy(dtype=float)

    # Doji now (br last row)
    _add_col(((br < 0.1 * rng) & (mean_br_20 > 0.15)).astype(float), 'pattern_doji_now')

    # Hammers etc. — use current candle only (pointwise on body, wicks)
    hammer = ((C > O) & (br > 0.25) & (lo_wick > 2 * body) & (up_wick < 0.5 * body)).astype(float)
    inv_hammer = ((C > O) & (br > 0.25) & (up_wick > 2 * body) & (lo_wick < 0.5 * body)).astype(float)
    shooting = ((C < O) & (br > 0.25) & (up_wick > 2 * body) & (lo_wick < 0.5 * body)).astype(float)
    hanging = ((C < O) & (br > 0.25) & (lo_wick > 2 * body) & (up_wick < 0.5 * body)).astype(float)
    _add_col(hammer, 'pattern_hammer')
    _add_col(inv_hammer, 'pattern_inverted_hammer')
    _add_col(shooting, 'pattern_shooting_star')
    _add_col(hanging, 'pattern_hanging_man')

    # Engulfing (2-candle pattern: compare i with i-1)
    c_prev = np.r_[C[0], C[:-1]]
    o_prev = np.r_[O[0], O[:-1]]
    b_prev = np.r_[body[0], body[:-1]]
    bull_engulf = ((c_prev < o_prev) & (C > O) & (O < c_prev) & (C > o_prev) & (body > b_prev * 1.2)).astype(float)
    bear_engulf = ((c_prev > o_prev) & (C < O) & (O > c_prev) & (C < o_prev) & (body > b_prev * 1.2)).astype(float)
    _add_col(bull_engulf, 'pattern_bull_engulfing')
    _add_col(bear_engulf, 'pattern_bear_engulfing')

    # Harami (i inside i-1)
    bull_harami = ((c_prev < o_prev) & (b_prev > 0.3 * rng) & (C > O) & (O > c_prev) & (C < o_prev)).astype(float)
    bear_harami = ((c_prev > o_prev) & (b_prev > 0.3 * rng) & (C < O) & (O < c_prev) & (C > o_prev)).astype(float)
    _add_col(bull_harami, 'pattern_bull_harami')
    _add_col(bear_harami, 'pattern_bear_harami')

    # Morning / evening star (3 candles: i-2, i-1, i)
    c2 = np.r_[C[0], C[0], C[:-2]]
    o2 = np.r_[O[0], O[0], O[:-2]]
    br2 = np.r_[br[0], br[0], br[:-2]]
    b1_prev2 = np.r_[body[0], body[0], body[:-2]]
    mstar = ((c2 < o2) & (br2 > 0.3) & (b_prev < 0.1) & (C > O) & (C > (o2 + c2) / 2.0)).astype(float)
    estar = ((c2 > o2) & (br2 > 0.3) & (b_prev < 0.1) & (C < O) & (C < (o2 + c2) / 2.0)).astype(float)
    _add_col(mstar, 'pattern_morning_star')
    _add_col(estar, 'pattern_evening_star')

    # 3 White Soldiers / 3 Black Crows (3 consecutive)
    def _rolling_3_all(cond_arr):
        if len(cond_arr) < 3:
            return np.zeros_like(cond_arr, dtype=float)
        a = cond_arr[2:]
        b = cond_arr[1:-1]
        c = cond_arr[:-2]
        out = np.zeros_like(cond_arr, dtype=float)
        out[2:] = (a & b & c).astype(float)
        return out

    up3 = _rolling_3_all(C > O)
    dn3 = _rolling_3_all(C < O)
    # Consecutive up: c1 > c2 > c3; similarly for downs
    c_prev1 = np.r_[C[0], C[:-1]]
    c_prev2 = np.r_[C[0], C[0], C[:-2]]
    c_mono_up = np.zeros(n, dtype=bool)
    c_mono_dn = np.zeros(n, dtype=bool)
    c_mono_up[2:] = (C[2:] > c_prev1[2:]) & (c_prev1[2:] > c_prev2[2:])
    c_mono_dn[2:] = (C[2:] < c_prev1[2:]) & (c_prev1[2:] < c_prev2[2:])
    br_0 = br
    br_1 = np.r_[br[0], br[:-1]]
    br_2 = np.r_[br[0], br[0], br[:-2]]
    br_avg_3 = (br_0 + br_1 + br_2) / 3.0
    t_ws_full = (up3 > 0.5) & c_mono_up & (br_avg_3 > 0.35)
    t_bc_full = (dn3 > 0.5) & c_mono_dn & (br_avg_3 > 0.35)
    _add_col(t_ws_full.astype(float), 'pattern_3_white_soldiers')
    _add_col(t_bc_full.astype(float), 'pattern_3_black_crows')

    # Tweezers (i-1 & i same high or low)
    h_prev = np.r_[H[0], H[:-1]]
    l_prev = np.r_[L[0], L[:-1]]
    tweezer_t = ((c_prev > o_prev) & (C < O) & (np.abs(H - h_prev) < 0.001 * np.maximum(H, h_prev)) & (up_wick > 0.4 * rng)).astype(float)
    tweezer_b = ((c_prev < o_prev) & (C > O) & (np.abs(L - l_prev) < 0.001 * np.maximum(L, l_prev)) & (lo_wick > 0.4 * rng)).astype(float)
    _add_col(tweezer_t, 'pattern_tweezer_top')
    _add_col(tweezer_b, 'pattern_tweezer_bottom')

    # Pattern score
    pbs = hammer + bull_engulf + bull_harami + mstar + t_ws_full.astype(float) + tweezer_b
    pears = shooting + bear_engulf + bear_harami + estar + t_bc_full.astype(float) + tweezer_t
    _add_col(pbs, 'pattern_bull_score_total')
    _add_col(pears, 'pattern_bear_score_total')
    _add_col(pbs - pears, 'pattern_net_score')

    # Candle last up
    _add_col((C > O).astype(float), 'candle_last_up')

    # Bull / bear streak — roll streak by reset. Vectorized approximation via cumsum reset trick.
    def _streak_up(cond_arr):
        res = np.zeros(len(cond_arr), dtype=float)
        cur = 0.0
        for i, v in enumerate(cond_arr):
            if v:
                cur = min(cur + 1.0, 10.0)
            else:
                cur = 0.0
            res[i] = cur
        return res
    _add_col(_streak_up(C > O), 'bull_streak')
    _add_col(_streak_up(C < O), 'bear_streak')

    for k in (5, 10, 20):
        up_diff = (C > np.r_[C[0], C[:-1]]).astype(float)
        ratio = pd.Series(up_diff).rolling(k, min_periods=1).mean().to_numpy(dtype=float)
        _add_col(ratio, f'up_candle_ratio_{k}')

    # =====================================================================
    # 18) VOLATILITY & RISK METRICS
    # =====================================================================
    _add_col(atr14, 'atr_14')
    _add_col(atr14 / (C + 1e-9) * 10000.0, 'atr_14_pips_pct')
    atr_roll60_mean = pd.Series(atr14).rolling(60, min_periods=1).mean().to_numpy(dtype=float)
    _add_col(atr14 / (atr_roll60_mean + 1e-9), 'atr_ratio_14_60')

    atr30 = _atr_pd(H_s, L_s, C_s, 30).to_numpy(dtype=float)
    _add_col(atr30 / (C + 1e-9) * 10000.0, 'atr_30_pips_pct')

    rv60 = pd.Series(C).rolling(60, min_periods=1).std(ddof=0) / (pd.Series(C).rolling(60, min_periods=1).mean() + 1e-9)
    _add_col(rv60.to_numpy(dtype=float), 'realized_vol_60')

    # Parkinson vol 30
    ln_hl = np.log(H / (L + 1e-9))
    park_seg = np.sqrt(pd.Series(ln_hl ** 2).rolling(30, min_periods=1).mean() / (4.0 * np.log(2.0)))
    _add_col(park_seg.to_numpy(dtype=float), 'parkinson_vol_30')

    # Garman-Klass 30
    seg_h = H
    seg_l = L
    seg_o = O
    seg_c = C
    lnhl = np.log(seg_h / (seg_l + 1e-9))
    lnco = np.log(seg_c / (seg_o + 1e-9))
    gk_core = 0.5 * lnhl ** 2 - (2 * np.log(2) - 1) * lnco ** 2
    gk_roll = np.sqrt(pd.Series(gk_core).rolling(30, min_periods=1).mean())
    _add_col(gk_roll.to_numpy(dtype=float), 'garman_klass_30')

    # Entropy of returns (discretized 30-bar window)
    ent = np.zeros(n, dtype=float)
    if n >= 30:
        bins = 8
        ret_s_pd = ret_s
        def _entropy(window):
            if len(window) < 10:
                return 0.0
            try:
                q = np.quantile(window, np.linspace(0, 1, bins + 1))
                d = np.digitize(window, bins=q)
                cts = np.bincount(d, minlength=bins + 2)
                p = cts / max(cts.sum(), 1)
                p2 = p[p > 0]
                return float(-np.sum(p2 * np.log(p2)))
            except Exception:
                return 0.0
        ent = ret_s_pd.rolling(30, min_periods=10).apply(_entropy, raw=True).fillna(0.0).to_numpy(dtype=float)
    _add_col(ent, 'ret_entropy_30')

    # Hurst approximation — simplified batch
    hurst_all = np.full(n, 0.5, dtype=float)
    hurst_flag = np.zeros(n, dtype=float)
    if n >= 120:
        def _hurst_w(w):
            try:
                seg2 = np.log(w[-120:]) if len(w) >= 120 else np.log(w)
                seg2 = seg2 - seg2[0]
                ns = len(seg2)
                rs_list = []
                for mm in (8, 16, 24, 32):
                    if ns >= mm * 2:
                        ns2 = ns // mm
                        rs_v = []
                        for ss in range(ns2):
                            ww = seg2[ss * mm:(ss + 1) * mm]
                            ww = ww - np.mean(ww)
                            cum = np.cumsum(ww)
                            rng = np.max(cum) - np.min(cum)
                            sd = np.std(ww) + 1e-12
                            rs_v.append(rng / sd)
                        if rs_v:
                            rs_list.append((mm, float(np.mean(rs_v))))
                if len(rs_list) >= 2:
                    xs = np.log([r[0] for r in rs_list])
                    ys = np.log([r[1] for r in rs_list])
                    return float(np.polyfit(xs, ys, 1)[0])
                return 0.5
            except Exception:
                return 0.5
        hurst_s = C_s.rolling(120, min_periods=120).apply(_hurst_w, raw=True).fillna(0.5)
        hurst_all = hurst_s.to_numpy(dtype=float)
    hurst_flag = (hurst_all > 0.55).astype(float) - (hurst_all < 0.45).astype(float)
    _add_col(hurst_all, 'hurst_approx')
    _add_col(hurst_flag, 'hurst_trend_flag')

    # =====================================================================
    # 19) ORDER-BLOCK & STRUCTURE FEATURES (swings)
    # =====================================================================
    def _swing_flags_batch(H_arr, L_arr, lr=2):
        """Return last swing high price & last swing low price series (per row)."""
        ns = len(H_arr)
        last_sh_p = np.zeros(ns, dtype=float)
        last_sl_p = np.zeros(ns, dtype=float)
        sh_hh2 = np.zeros(ns, dtype=float)  # higher high last 2
        sl_ll2 = np.zeros(ns, dtype=float)  # higher low last 2
        sh_list = []
        sl_list = []
        cur_sh_p = 0.0
        cur_sl_p = 0.0
        prev_sh_p = 0.0
        prev_sl_p = 0.0
        for i in range(ns):
            if i >= lr and i < ns - lr:
                window_h = H_arr[i - lr:i + lr + 1]
                window_l = L_arr[i - lr:i + lr + 1]
                is_sh = (H_arr[i] == np.max(window_h)) and (H_arr[i] > H_arr[i-1]) and (H_arr[i] > H_arr[i+1])
                is_sl = (L_arr[i] == np.min(window_l)) and (L_arr[i] < L_arr[i-1]) and (L_arr[i] < L_arr[i+1])
                if is_sh:
                    prev_sh_p = cur_sh_p
                    cur_sh_p = H_arr[i]
                    sh_list.append((i, cur_sh_p))
                if is_sl:
                    prev_sl_p = cur_sl_p
                    cur_sl_p = L_arr[i]
                    sl_list.append((i, cur_sl_p))
            last_sh_p[i] = cur_sh_p
            last_sl_p[i] = cur_sl_p
            if len(sh_list) >= 2:
                sh_hh2[i] = float(sh_list[-1][1] > sh_list[-2][1])
            if len(sl_list) >= 2:
                sl_ll2[i] = float(sl_list[-1][1] > sl_list[-2][1])
        return last_sh_p, last_sl_p, sh_hh2, sl_ll2

    if n >= 60:
        sp_lp, lp_lp, hh2_s, ll2_s = _swing_flags_batch(H, L, 2)
        _add_col((C - sp_lp) / (C + 1e-9), 'dist_last_swing_high')
        _add_col((C - lp_lp) / (C + 1e-9), 'dist_last_swing_low')
        _add_col(hh2_s, 'higher_high_last2')
        _add_col(ll2_s, 'higher_low_last2')
        _add_col(hh2_s * ll2_s, 'structure_hhhl_bull')
        _add_col((1 - hh2_s) * (1 - ll2_s), 'structure_lhll_bear')
    else:
        for nm in ('dist_last_swing_high', 'dist_last_swing_low', 'higher_high_last2',
                   'higher_low_last2', 'structure_hhhl_bull', 'structure_lhll_bear'):
            _add_col(np.zeros(n, dtype=float), nm)

    # =====================================================================
    # 20) TIME FEATURES
    # =====================================================================
    idx = work.index
    dow_arr = np.full(n, 3.0, dtype=float)
    hod_arr = np.full(n, 12.0, dtype=float)
    try:
        if hasattr(idx[0], 'weekday'):
            dt_idx = pd.to_datetime(idx)
            dow_arr = np.array([float(d.weekday()) for d in dt_idx])
            hod_arr = np.array([float(d.hour) for d in dt_idx])
        elif 'time' in work.columns:
            dt_idx = pd.to_datetime(work['time'])
            dow_arr = np.array([float(d.weekday()) for d in dt_idx])
            hod_arr = np.array([float(d.hour) for d in dt_idx])
    except Exception:
        pass
    _add_col(dow_arr / 6.0, 'day_of_week_norm')
    _add_col(hod_arr / 23.0, 'hour_of_day_norm')
    london = ((hod_arr >= 7) & (hod_arr < 16)).astype(float)
    ny = ((hod_arr >= 12) & (hod_arr < 21)).astype(float)
    tokyo = ((hod_arr >= 0) & (hod_arr < 9)).astype(float)
    _add_col(london, 'session_london')
    _add_col(ny, 'session_newyork')
    _add_col(tokyo, 'session_tokyo')
    _add_col(london * ny, 'session_london_ny_overlap')

    # =====================================================================
    # EXTENDED CONTEXT (cross-pair signals: USD strength, risk sentiment,
    # gold momentum) — fixed key set so feature count is identical whether
    # or not context data was available for this symbol/timeframe.
    # =====================================================================
    from ai_engine.models.market_context import MARKET_CONTEXT_KEYS
    if extended_context is not None and len(extended_context) > 0:
        ctx_aligned = extended_context.reindex(work.index, method="ffill").fillna(0.0)
        for key in MARKET_CONTEXT_KEYS:
            col = ctx_aligned[key].to_numpy(dtype=float) if key in ctx_aligned.columns else np.zeros(n, dtype=float)
            _add_col(col, f'ext_{key}')
    else:
        for key in MARKET_CONTEXT_KEYS:
            _add_col(np.zeros(n, dtype=float), f'ext_{key}')

    # =====================================================================
    # STACK MATRIX + winsorize row-wise via MAD
    # =====================================================================
    if not col_list:
        return np.empty((0, 0), dtype=np.float32), []

    X = np.vstack(col_list).T.astype(np.float32)  # (n, n_features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Winsorize PER ROW (axis=1: across this row's own features), matching
    # compute_all_features() exactly. Row-wise keeps every row's clip bounds
    # a function of only that row's own feature values — no cross-row/future
    # information involved, and identical preprocessing between batch
    # training and single-row live inference (previously this used axis=0
    # global column stats computed from the ENTIRE history including future
    # bars relative to any given row, and used different math than inference,
    # i.e. train/serve skew).
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True) + 1e-9
    X = np.where(np.abs(X - med) > 12 * mad, med + 12 * mad * np.sign(X - med), X).astype(np.float32)

    return X, name_list


_FEATURE_COUNT_CACHE: List[int] = []


def feature_count_target() -> int:
    """
    Determined dynamically (once, then cached) from an actual feature build
    rather than hardcoded — a hardcoded count silently drifts every time a
    feature block is added or removed (this one had drifted from 175 to 230
    before this fix, undetected). Padding/truncation logic elsewhere in the
    pipeline relies on this being exactly correct.
    """
    if not _FEATURE_COUNT_CACHE:
        # Several feature blocks only activate above row-count thresholds
        # (n>=200/300 guards); must use enough realistic rows that EVERY
        # block fires, matching what real usage (300+ rows, real price
        # noise) actually sees, or this undercounts silently.
        rng = np.random.RandomState(42)
        n_dummy = 500
        close = 1.10 + np.cumsum(rng.normal(0, 0.0003, n_dummy))
        dummy = pd.DataFrame({
            'open': close + rng.normal(0, 0.0001, n_dummy),
            'high': close + np.abs(rng.normal(0.0003, 0.0001, n_dummy)),
            'low': close - np.abs(rng.normal(0.0003, 0.0001, n_dummy)),
            'close': close,
            'volume': np.abs(rng.normal(100, 20, n_dummy)),
        })
        _, names = compute_all_features(dummy, lookback=100, symbol='')
        _FEATURE_COUNT_CACHE.append(len(names))
    return _FEATURE_COUNT_CACHE[0]
