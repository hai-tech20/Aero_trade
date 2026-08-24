"""
run_advanced_retrain.py — CLI wrapper for Ultra-Ensemble + Triple-Barrier training.
=================================================================================
Usage examples:
  # Dry-run report (no training, no writes):
  python scripts/run_advanced_retrain.py --dry

  # Train a specific well-known symbol first:
  python scripts/run_advanced_retrain.py --symbols EURUSD --timeframes 15min 5min 1min

  # Full retrain across all configured symbols + timeframes:
  python scripts/run_advanced_retrain.py --retrain-all

  # Train only symbols that currently have NO promoted model (gaps):
  python scripts/run_advanced_retrain.py --fill-gaps
"""

from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.training.advanced_trainer import run_all, main
from ai_engine.training.model_status import get_best_timeframe_for_symbol
from config.settings import TRADING_CONFIG


def _gap_symbols() -> list[str]:
    """Return configured symbols that have NO active promoted timeframe at all."""
    all_pairs = [str(p) for p in TRADING_CONFIG.get("pairs", ["EURUSD"])]
    gaps = []
    for sym in all_pairs:
        if get_best_timeframe_for_symbol(sym) is None:
            gaps.append(sym)
    return gaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced Ultra-Ensemble retraining entry-point (wraps ai_engine.training.advanced_trainer)",
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbol list (e.g. EURUSD GBPUSD XAUUSD)")
    parser.add_argument("--timeframes", nargs="+", default=None, help="Timeframes list (e.g. 1min 5min 15min 1h)")
    parser.add_argument("--dry", action="store_true", help="Dry run: report only, do not train or write files")
    parser.add_argument("--retrain-all", action="store_true", help="Retrain all configured symbols/timeframes")
    parser.add_argument("--fill-gaps", action="store_true", help="Only train symbols with NO promoted model")
    parser.add_argument("--full-ensemble", action="store_true", help="Use full-size sub-models (more estimators) instead of fast_mode — same TP:SL, same promotion bar, just more modeling effort. Slower per combo.")
    parser.add_argument("--pairs", nargs="+", default=None, help="Explicit SYMBOL:TIMEFRAME combos, e.g. --pairs CADCHF:1min USDCAD:15min. Bypasses the symbols x timeframes cartesian product.")
    args = parser.parse_args()

    start = datetime.utcnow()
    print(f"[run_advanced_retrain] started at {start.isoformat()} | args={vars(args)}")

    symbols = args.symbols
    timeframes = args.timeframes
    dry = bool(args.dry)

    if args.fill_gaps and not symbols:
        symbols = _gap_symbols()
        print(f"[run_advanced_retrain] --fill-gaps selected: {len(symbols)} symbols with no promoted model -> {symbols}")
        if not symbols:
            print("[run_advanced_retrain] no gaps found. Nothing to do.")
            sys.exit(0)

    if args.retrain_all and not symbols:
        symbols = [str(p) for p in TRADING_CONFIG.get("pairs", ["EURUSD"])]
        print(f"[run_advanced_retrain] --retrain-all selected: {len(symbols)} symbols -> {symbols}")

    fast_mode = False if args.full_ensemble else None
    explicit_pairs = None
    if args.pairs:
        explicit_pairs = []
        for p in args.pairs:
            sym, _, tf = p.partition(":")
            if not sym or not tf:
                print(f"[run_advanced_retrain] skipping malformed --pairs entry: {p!r} (expected SYMBOL:TIMEFRAME)")
                continue
            explicit_pairs.append((sym, tf))
        print(f"[run_advanced_retrain] --pairs selected: {len(explicit_pairs)} explicit combos -> {explicit_pairs}")

    report = run_all(symbols=symbols, timeframes=timeframes, dry_run=dry, fast_mode=fast_mode, pairs=explicit_pairs)

    end = datetime.utcnow()
    if report is not None and len(report) > 0:
        promoted = int(report.get("promoted", pd.Series([0])).sum()) if "promoted" in report.columns else 0
        rows = len(report)
        print(f"\n[run_advanced_retrain] DONE in {(end-start).total_seconds():.0f}s.")
        print(f"  Rows:        {rows}")
        print(f"  Promoted:    {promoted}")
        if "trained_accuracy" in report.columns:
            print(f"  Mean acc:    {report['trained_accuracy'].mean():.4f}  (NOTE: inflated by triple-barrier label imbalance — not comparable to a 50/50 baseline, see precision_buy_edge_pts)")
        if "precision_buy_edge_pts" in report.columns:
            print(f"  Mean edge:   {report['precision_buy_edge_pts'].mean():+.2f} pts (precision_buy vs random baseline — this is the real skill signal)")
        if "expectancy_r" in report.columns:
            print(f"  Mean exp(R): {report['expectancy_r'].replace([float('inf'), -float('inf')], float('nan')).mean():.4f}")
    else:
        print(f"\n[run_advanced_retrain] DONE in {(end-start).total_seconds():.0f}s — empty report.")
