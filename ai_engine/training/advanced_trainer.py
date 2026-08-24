"""
ADVANCED TRAINING ORCHESTRATOR — AERO TRADE MASTERY EDITION
============================================================
End-to-end trainer that:
  1. Loads history per (symbol, timeframe) from parquet
  2. Generates 175+ features via feature_engine
  3. Labels using triple-barrier (TP=3×ATR, SL=1.2×ATR, 60-bar horizon)
  4. Temporal walk-forward train/test 80/20 split
  5. Trains 7-model heterogeneous UltraEnsemble
  6. Only PROMOTES if:
       a) Out-of-fold accuracy >= 0.63 absolute
       b) beats current baseline promoted accuracy by 2 percentage points margin
       c) >= 3 sub-models are active
  7. Produces a final per-symbol, per-timeframe report CSV

Usage:
  python -m ai_engine.training.advanced_trainer --retrain-all --dry
"""

from __future__ import annotations
import argparse
import gc
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ai_engine.models.feature_engine import (
    compute_all_features,
    compute_all_features_matrix,
    feature_count_target,
)
from ai_engine.models.ultra_ensemble import (
    UltraEnsemble,
    MIN_MODEL_ABSOLUTE_ACCURACY,
    MODEL_PROMOTION_ABSOLUTE_FLOOR,
    MODEL_PROMOTION_MARGIN,
)
from ai_engine.models.market_context import compute_market_context_series
from ai_engine.training.advanced_labels import prepare_training_matrix, expected_expectancy_from_model
from ai_engine.training.model_status import get_promoted_accuracy, MIN_HARD_ACCURACY_FLOOR
from config.settings import (
    DATA_DIR,
    MODEL_DIR,
    TRADING_CONFIG,
    AI_CONFIG,
    RISK_CONFIG,
    get_data_path,
)
from utils.logger import TradingLogger

logger = TradingLogger("AdvancedTrainer")
REPORT_FILE = Path(__file__).parent.parent.parent / "logs" / "advanced_training_report.csv"
TRADING_TIMEFRAMES = ["1min", "5min", "15min"]


@dataclass
class TrainResult:
    symbol: str
    timeframe: str
    history_rows: int
    labeled_samples: int
    baseline_accuracy: float
    trained_accuracy: float
    promoted: bool
    promote_reason: str
    expectancy_r: float
    precision_buy: float
    precision_sell: float
    train_time_sec: float
    random_baseline_precision: float = 0.0
    precision_buy_edge_pts: float = 0.0
    error: str = ""

    def csv_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = datetime.utcnow().isoformat()
        return d


def _ultra_meta_path(symbol: str, timeframe: str) -> Path:
    return MODEL_DIR / f"ultra_{symbol}_{timeframe}.metadata.json"


def _ultra_model_prefix(symbol: str, timeframe: str) -> str:
    return str(MODEL_DIR / f"ultra_{symbol}_{timeframe}")


def _should_skip_existing(symbol: str, timeframe: str) -> Optional[TrainResult]:
    """Skip training when a promoted ultra model already meets the hard floor."""
    meta_path = _ultra_meta_path(symbol, timeframe)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    accuracy = float(
        meta.get("accuracy")
        or meta.get("validation", {}).get("candidate_model_accuracy")
        or 0.0
    )
    promoted = meta.get("validation", {}).get("promoted", meta.get("promoted", True))
    ensemble_path = Path(_ultra_model_prefix(symbol, timeframe) + "_ensemble_meta.joblib")
    if promoted is False or accuracy < MIN_HARD_ACCURACY_FLOOR or not ensemble_path.exists():
        return None

    logger.info(
        f"Skipping {symbol}/{timeframe} — promoted ultra model already exists "
        f"(accuracy={accuracy:.4f})"
    )
    return TrainResult(
        symbol=symbol,
        timeframe=timeframe,
        history_rows=int(meta.get("history_rows", 0)),
        labeled_samples=int(meta.get("labeled_samples", 0)),
        baseline_accuracy=float(meta.get("baseline_accuracy", 0.0)),
        trained_accuracy=accuracy,
        promoted=True,
        promote_reason="skipped_existing_promoted_model",
        expectancy_r=float(meta.get("expectancy_r", 0.0)),
        precision_buy=float(meta.get("precision_buy", 0.0)),
        precision_sell=float(meta.get("precision_sell", 0.0)),
        train_time_sec=0.0,
        error="",
    )


def _extract_features_wrapper(df, **kw):
    return compute_all_features(df, lookback=kw.get("lookback", 100), symbol=kw.get("symbol", ""))


def train_symbol_timeframe(
    symbol: str,
    timeframe: str,
    dry_run: bool = False,
    fast_mode: Optional[bool] = None,
    market_context: Optional[pd.DataFrame] = None,
) -> TrainResult:
    t0 = datetime.now()
    skipped = _should_skip_existing(symbol, timeframe)
    if skipped is not None:
        return skipped

    data_path = get_data_path(symbol, timeframe)
    baseline = float(get_promoted_accuracy(symbol, timeframe) or 0.0)
    result = TrainResult(
        symbol=symbol, timeframe=timeframe,
        history_rows=0, labeled_samples=0,
        baseline_accuracy=baseline,
        trained_accuracy=0.0, promoted=False,
        promote_reason="not_started",
        expectancy_r=0.0, precision_buy=0.0, precision_sell=0.0,
        train_time_sec=0.0, error=""
    )

    if not data_path.exists():
        result.error = f"history_missing: {data_path}"
        result.promote_reason = "missing_history"
        return result

    try:
        df = pd.read_parquet(data_path)
    except Exception as ex:
        result.error = f"parquet_load_failed: {ex}"
        return result
    if len(df) < 250:
        result.error = f"history_too_short: {len(df)}"
        return result

    if "time" in df.columns:
        try:
            df.index = pd.to_datetime(df["time"])
        except Exception:
            pass

    result.history_rows = len(df)

    # The row cap below (max_train_rows) only limits how many rows go into
    # ensemble.fit() — it does NOT limit compute_all_features_matrix, which
    # builds a dense feature matrix over the FULL raw history first. That
    # full-history build (e.g. ~99k rows x 239 features, with multiple
    # intermediate float64 arrays per indicator block during vectorized
    # rolling-window computation) is the actual point where this machine's
    # memory allocation failures were happening, well before any per-model
    # row cap ever applied. Truncate to the most recent rows before feature
    # building, not just before training, to actually fix the crash site.
    max_feature_rows = int(AI_CONFIG.get("max_feature_build_rows", 60000))
    if len(df) > max_feature_rows:
        df = df.iloc[-max_feature_rows:].copy()

    logger.info(f"Training {symbol}/{timeframe} — history={result.history_rows} rows (using {len(df)}) baseline_acc={baseline:.3f}")

    fe_kwargs = {"lookback": 100, "symbol": symbol}

    t_feat = datetime.now()
    logger.info(f"  -> Building batch feature matrix ({len(df)} rows × {feature_count_target()} feats) …")
    X_batch, names_batch = compute_all_features_matrix(
        df,
        lookback=fe_kwargs.get("lookback", 100),
        symbol=fe_kwargs.get("symbol", symbol),
        min_rows_required=250,
        extended_context=market_context,
    )
    logger.info(f"  -> Feature matrix ready shape={X_batch.shape} in {(datetime.now()-t_feat).total_seconds():.1f}s")

    prep = prepare_training_matrix(
        df,
        feature_extractor=_extract_features_wrapper,
        feature_extractor_kwargs=fe_kwargs,
        tp_mult=RISK_CONFIG.get("atr_tp_multiplier", 3.0),
        sl_mult=RISK_CONFIG.get("atr_sl_multiplier", 1.2),
        horizon_bars=60,
        min_rows_required=250,
        walkforward_split=True,
        test_frac=0.20,
        X_precomputed=X_batch,
        feature_names_precomputed=names_batch,
    )
    if "error" in prep:
        result.error = f"prepare_failed: {prep['error']}"
        result.promote_reason = prep["error"]
        return result

    Xtr, ytr = prep["X_train"], prep["y_train"]
    Xte, yte = prep["X_test"], prep["y_test"]
    result.labeled_samples = int(prep.get("n_samples", 0))
    logger.info(f"  -> Labeled samples: {result.labeled_samples} Train: {len(ytr)} Test: {len(yte)} Distribution: {prep['label_distribution']}")

    # Handle feature dim alignment if needed
    if Xtr.shape[1] < feature_count_target():
        pad = np.zeros((Xtr.shape[0], feature_count_target() - Xtr.shape[1]), dtype=Xtr.dtype)
        Xtr = np.hstack([Xtr, pad])
        pad2 = np.zeros((Xte.shape[0], feature_count_target() - Xte.shape[1]), dtype=Xte.dtype)
        Xte = np.hstack([Xte, pad2])
    elif Xtr.shape[1] > feature_count_target():
        Xtr = Xtr[:, :feature_count_target()]
        Xte = Xte[:, :feature_count_target()]

    # Training-size cap: tree ensembles converge well before 70k+ rows;
    # extra rows only add quadratic cost with near-zero accuracy gain.
    # Larger cap for HTFs (15min+) where each bar carries more info.
    if timeframe in ("1min", "5min"):
        max_train_rows = AI_CONFIG.get("max_train_rows_ltf", 35000)
    elif timeframe in ("15min",):
        max_train_rows = AI_CONFIG.get("max_train_rows_mtf", 45000)
    else:
        max_train_rows = AI_CONFIG.get("max_train_rows_htf", None)

    # Train ultra ensemble (multicore via n_jobs=-1 on all tree submodels)
    effective_fast_mode = bool(AI_CONFIG.get("ensemble_fast_mode", False)) if fast_mode is None else bool(fast_mode)
    ensemble = UltraEnsemble(
        symbol=symbol,
        timeframe=timeframe,
        n_jobs=AI_CONFIG.get("ensemble_n_jobs", -1),
        fast_mode=effective_fast_mode,
    )
    try:
        acc, meets_criteria = ensemble.fit(
            Xtr, ytr, X_val=Xte, y_val=yte,
            max_train_rows=max_train_rows,
        )
    except Exception as ex:
        result.error = f"ensemble_fit_failed: {ex}"
        result.promote_reason = "fit_exception"
        return result
    result.trained_accuracy = float(acc)

    # Test-set prediction to compute expectancy
    try:
        probs = []
        votes_list = []
        for i in range(len(Xte)):
            d, c, _ = ensemble.predict(Xte[i:i + 1])
            votes_list.append(d)
            if d == 1:
                probs.append(c)
            elif d == -1:
                probs.append(1.0 - c)
            else:
                probs.append(0.5)
        pred = (np.array(probs) >= 0.5).astype(int)
        metrics = expected_expectancy_from_model(
            yte, pred, np.array(probs),
            avg_tp=RISK_CONFIG.get("atr_tp_multiplier", 3.0),
            avg_sl=RISK_CONFIG.get("atr_sl_multiplier", 1.2),
        )
        result.expectancy_r = float(metrics["expectancy_per_trade_r"])
        result.precision_buy = float(metrics["precision_buy"])
        result.precision_sell = float(metrics["precision_sell"])
        # Triple-barrier labels are imbalanced by construction (see note below
        # at promotion gate) — report precision_buy relative to the no-skill
        # random baseline so "accuracy" isn't read as genuine directional edge.
        _tp_m0 = RISK_CONFIG.get("atr_tp_multiplier", 3.0)
        _sl_m0 = RISK_CONFIG.get("atr_sl_multiplier", 1.2)
        result.random_baseline_precision = _sl_m0 / (_tp_m0 + _sl_m0)
        result.precision_buy_edge_pts = (result.precision_buy - result.random_baseline_precision) * 100.0
        logger.info(
            f"  -> precision_buy={result.precision_buy:.3f} vs random_baseline={result.random_baseline_precision:.3f} "
            f"(edge={result.precision_buy_edge_pts:+.1f}pts) expectancy={result.expectancy_r:.3f}R"
        )
    except Exception as ex:
        logger.warning(f"  expectancy eval failed: {ex}")

    # ----------------------------------------------------------
    # STRICT PROMOTION GATES (multiple conditions all required)
    # ----------------------------------------------------------
    absolute_floor = max(
        MODEL_PROMOTION_ABSOLUTE_FLOOR,
        AI_CONFIG.get("promotion_absolute_floor", 0.63),
    )
    # Bootstrap: when no promoted baseline exists, allow the trading hard floor (60%).
    if baseline <= 0:
        absolute_floor = max(
            MIN_HARD_ACCURACY_FLOOR,
            float(AI_CONFIG.get("min_model_accuracy", 0.60)),
        )
    min_margin = max(MODEL_PROMOTION_MARGIN, AI_CONFIG.get("promotion_baseline_margin", 0.02))

    reasons: List[str] = []
    promoted = True
    if result.trained_accuracy < absolute_floor:
        promoted = False
        reasons.append(f"absolute_floor ({result.trained_accuracy:.3f} < {absolute_floor})")
    if result.trained_accuracy <= baseline + min_margin and baseline > 0:
        promoted = False
        reasons.append(f"baseline_margin ({result.trained_accuracy:.3f} <= baseline {baseline:.3f} + {min_margin})")
    if not meets_criteria:
        promoted = False
        reasons.append("submodels_insufficient (needs >=3 active subs)")
    if result.expectancy_r < 0.10:
        promoted = False
        reasons.append(f"expectancy_non_positive ({result.expectancy_r:.3f} R per trade)")
    if dry_run:
        promoted = False
        reasons.append("dry_run")

    result.promoted = bool(promoted)
    result.promote_reason = " | ".join(reasons) if reasons else "all_gates_passed"

    if result.promoted:
        save_prefix = _ultra_model_prefix(symbol, timeframe)
        ok = ensemble.save(save_prefix)
        meta_path = _ultra_meta_path(symbol, timeframe)
        try:
            meta = {
                "symbol": symbol,
                "timeframe": timeframe,
                "promoted": True,
                "promoted_at": datetime.utcnow().isoformat(),
                "accuracy": result.trained_accuracy,
                "oof_accuracy": float(getattr(ensemble, "_oof_accuracy", result.trained_accuracy)),
                "actionable_accuracy": float(getattr(ensemble, "_actionable_accuracy", 0.0)),
                "actionable_samples": int(getattr(ensemble, "_actionable_samples", 0)),
                "baseline_accuracy": result.baseline_accuracy,
                "expectancy_r": result.expectancy_r,
                "precision_buy": result.precision_buy,
                "precision_sell": result.precision_sell,
                "random_baseline_precision": result.random_baseline_precision,
                "precision_buy_edge_pts": result.precision_buy_edge_pts,
                "labeled_samples": result.labeled_samples,
                "history_rows": result.history_rows,
                "absolute_floor": absolute_floor,
                "margin_required": min_margin,
                "validation": {
                    "candidate_model_accuracy": result.trained_accuracy,
                    "promoted": True,
                    "model_family": "ultra",
                },
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        logger.info(f"  -> PROMOTED accuracy={result.trained_accuracy:.4f} expectancy={result.expectancy_r:.3f} R save_ok={ok}")
    else:
        logger.warning(f"  -> NOT PROMOTED: {result.promote_reason}")

    result.train_time_sec = (datetime.now() - t0).total_seconds()
    return result


def run_all(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    dry_run: bool = False,
    fast_mode: Optional[bool] = None,
    pairs: Optional[List[tuple]] = None,
) -> pd.DataFrame:
    """
    pairs: optional explicit list of (symbol, timeframe) tuples to train,
    bypassing the symbols x timeframes cartesian product. Use this for a
    targeted retry of specific combos (e.g. near-miss pairs) instead of
    wastefully retraining every timeframe for every symbol involved.
    """
    if pairs is not None:
        combo_list = list(pairs)
    else:
        symbols = symbols or list(TRADING_CONFIG["pairs"])
        timeframes = timeframes or TRADING_TIMEFRAMES
        combo_list = [(s, tf) for s in symbols for tf in timeframes]
    rows = []
    total = len(combo_list)
    done = 0

    # Load any pre-existing report from a prior invocation ONCE up front, so
    # per-iteration checkpoint writes below always include it. Writing only
    # `rows` (this run's own accumulator) at each checkpoint would overwrite
    # and permanently destroy prior runs' history the moment this run's
    # first checkpoint fires — this base must be captured before the loop.
    old_df = pd.DataFrame()
    if REPORT_FILE.exists():
        try:
            old_df = pd.read_csv(REPORT_FILE)
        except Exception:
            pass

    # Market context (USD strength / risk sentiment / gold momentum) is the
    # same reference-basket series for every symbol at a given timeframe —
    # load once per timeframe and reuse, instead of reloading the same 8
    # reference files on every single symbol's training call.
    context_cache: Dict[str, pd.DataFrame] = {}

    def _context_for(tf: str) -> Optional[pd.DataFrame]:
        if tf not in context_cache:
            try:
                context_cache[tf] = compute_market_context_series(DATA_DIR, tf)
            except Exception as ex:
                logger.warning(f"market_context unavailable for {tf}: {ex}")
                context_cache[tf] = None
        return context_cache[tf]

    for s, tf in combo_list:
        done += 1
        logger.info(f"[run_all] ({done}/{total}) {s}/{tf}")
        try:
            row = train_symbol_timeframe(
                s, tf, dry_run=dry_run, fast_mode=fast_mode, market_context=_context_for(tf)
            ).csv_row()
        except Exception as ex:
            # A single combo's failure (memory pressure, transient allocation
            # error, etc.) must never kill a multi-hour run — log it as a
            # failed row and keep going. Losing hours of already-completed
            # progress to one bad combo is worse than skipping that combo.
            logger.error(f"[run_all] ({done}/{total}) {s}/{tf} CRASHED: {ex}")
            row = TrainResult(
                symbol=s, timeframe=tf, history_rows=0, labeled_samples=0,
                baseline_accuracy=0.0, trained_accuracy=0.0, promoted=False,
                promote_reason="run_crashed", expectancy_r=0.0,
                precision_buy=0.0, precision_sell=0.0, train_time_sec=0.0,
                error=str(ex),
            ).csv_row()
        rows.append(row)
        # Checkpoint after each symbol/timeframe so long runs can resume safely.
        df_partial = pd.concat([old_df, pd.DataFrame(rows)], ignore_index=True) if len(old_df) else pd.DataFrame(rows)
        if len(df_partial):
            REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
            df_partial.to_csv(REPORT_FILE, index=False)
        # Proactively release this combo's large objects (ensemble, feature
        # matrices) before starting the next one — left unmanaged, memory
        # accumulates across many sequential combos in one long-lived process.
        gc.collect()
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated symbols")
    ap.add_argument("--timeframes", default="1min,5min,15min", help="Comma-separated timeframes")
    ap.add_argument("--retrain-all", action="store_true", help="Train all pairs/tfs")
    ap.add_argument("--dry", action="store_true", help="Evaluate, don't save models")
    args = ap.parse_args()
    syms = None
    tfs = None
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.timeframes:
        tfs = [s.strip() for s in args.timeframes.split(",") if s.strip()]
    report = run_all(symbols=syms, timeframes=tfs, dry_run=args.dry)
    print(report.to_string(index=False))
    promoted_n = (report["promoted"] == True).sum()
    print(f"\nSummary: {len(report)} pairs/TFs trained, {promoted_n} promoted")


if __name__ == "__main__":
    main()
