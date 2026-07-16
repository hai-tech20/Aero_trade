import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ai_engine.training.retraining_service import MODEL_KEYS
from ai_engine.models.model_keys import make_model_key, base_model_key_for_timeframe
from config.settings import DATA_DIR, MODEL_DIR, TRADING_CONFIG


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_model_status() -> List[Dict]:
    rows: List[Dict] = []
    for symbol in TRADING_CONFIG.get("pairs", []):
        for timeframe in TRADING_CONFIG.get("timeframes", []):
            model_name = MODEL_KEYS.get(timeframe)
            history_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
            history_available = history_path.exists()
            history_rows = 0
            history_range = [None, None]
            if history_available:
                try:
                    df = pd.read_parquet(history_path)
                    history_rows = int(len(df))
                    if len(df) > 0:
                        ts = pd.to_datetime(df["time"]) if "time" in df.columns else pd.to_datetime(df.index)
                        history_range = [str(ts.min()), str(ts.max())]
                except Exception:
                    history_available = False

            metadata = None
            active_model_version = None
            active_accuracy = None
            last_retrain_at = None
            last_retrain_outcome = None

            if model_name:
                family = 'gbm' if timeframe == '15min' else 'rf'
                symbol_model_name = make_model_key(symbol, timeframe, family)
                fallback_model_name = base_model_key_for_timeframe(timeframe)
                for candidate_name in [symbol_model_name, fallback_model_name]:
                    active_metadata_path = MODEL_DIR / f"{candidate_name}.metadata.json"
                    metadata = _read_json(active_metadata_path)
                    if metadata and metadata.get("symbol") == symbol and metadata.get("timeframe") == timeframe:
                        version = metadata.get("version")
                        active_model_version = f"{candidate_name}_v{version}" if version else candidate_name
                        active_accuracy = metadata.get("validation", {}).get("candidate_model_accuracy")
                        last_retrain_at = metadata.get("trained_at")
                        last_retrain_outcome = "promoted" if metadata.get("validation", {}).get("promoted") else "rejected"
                        break

            rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "history_available": history_available,
                "history_rows": history_rows,
                "history_date_range": history_range,
                "active_model_version": active_model_version,
                "active_model_accuracy": active_accuracy,
                "last_retrain_at": last_retrain_at,
                "last_retrain_outcome": last_retrain_outcome,
                "weak_signal": active_accuracy is not None and float(active_accuracy) <= 0.52,
            })

    rows.sort(key=lambda row: (not row["history_available"], row["weak_signal"], row["symbol"], row["timeframe"]))
    return rows
