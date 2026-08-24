import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ai_engine.training.retraining_service import MODEL_KEYS
from ai_engine.models.model_keys import make_model_key, base_model_key_for_timeframe
from config.settings import DATA_DIR, MODEL_DIR, TRADING_CONFIG, AI_CONFIG


MIN_HARD_ACCURACY_FLOOR: float = float(AI_CONFIG.get("min_model_accuracy", 0.60))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _demote_model_metadata(metadata_path: Path, reason: str = "below_hard_floor") -> bool:
    """Demote (invalidate) a model metadata file by marking promoted=False and writing reason."""
    try:
        meta = _read_json(metadata_path)
        if meta is None:
            return False
        if "validation" not in meta:
            meta["validation"] = {}
        meta["validation"]["promoted"] = False
        meta["validation"]["demoted"] = True
        meta["validation"]["demotion_reason"] = reason
        meta["validation"]["demoted_at"] = pd.Timestamp.utcnow().isoformat()
        metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _find_best_metadata(symbol: str, timeframe: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Find the best (most accurate) metadata for this symbol/timeframe, checking:
      1. New ultra_ensemble metadata (ultra_{symbol}_{tf}.metadata.json)
      2. Standard RF/GBM metadata
      3. Versioned variants
    Returns (metadata_or_None, source_family).
    """
    candidates: List[Tuple[float, Dict[str, Any], str]] = []

    # 1) Ultra ensemble
    ultra_path = MODEL_DIR / f"ultra_{symbol}_{timeframe}.metadata.json"
    umeta = _read_json(ultra_path)
    if umeta:
        uacc = float(umeta.get("accuracy") or umeta.get("validation", {}).get("candidate_model_accuracy") or 0.0)
        candidates.append((uacc, umeta, "ultra"))

    # 2) Standard RF/GBM
    family = "gbm" if timeframe == "15min" else "rf"
    symbol_model_name = make_model_key(symbol, timeframe, family)
    fallback_model_name = base_model_key_for_timeframe(timeframe)
    for candidate_name in [symbol_model_name, fallback_model_name]:
        meta = _read_json(MODEL_DIR / f"{candidate_name}.metadata.json")
        if meta and meta.get("symbol") == symbol and meta.get("timeframe") == timeframe:
            acc = float(meta.get("validation", {}).get("candidate_model_accuracy") or 0.0)
            candidates.append((acc, meta, family))
        # Versioned variants
        for p in MODEL_DIR.glob(f"{candidate_name}_v*.metadata.json"):
            meta2 = _read_json(p)
            if meta2 and meta2.get("symbol") == symbol and meta2.get("timeframe") == timeframe:
                acc2 = float(meta2.get("validation", {}).get("candidate_model_accuracy") or 0.0)
                candidates.append((acc2, meta2, family))

    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def get_best_timeframe_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Return best active timeframe/accuracy for this symbol.

    HARD FILTER: a model whose accuracy < MIN_HARD_ACCURACY_FLOOR is treated as INACTIVE
    (counts as None). This prevents trading on weak coin-flip models.
    """
    best: Optional[Dict[str, Any]] = None
    live_timeframes = TRADING_CONFIG.get("live_intraday_timeframes") or TRADING_CONFIG.get("timeframes", [])
    for timeframe in live_timeframes:
        metadata, family = _find_best_metadata(symbol, timeframe)
        if metadata is None:
            continue
        # Extract accuracy (ultra metadata uses "accuracy" key directly)
        accuracy = metadata.get("accuracy") or metadata.get("validation", {}).get("candidate_model_accuracy")
        if accuracy is None:
            continue
        accuracy = float(accuracy)

        # Promoted flag check — ultra metadata uses top-level promoted=True
        promoted = metadata.get("validation", {}).get("promoted")
        if promoted is None:
            promoted = metadata.get("promoted")
        if promoted is False:
            continue

        # HARD ACCURACY FLOOR — silently skip models that can't meet it
        if accuracy < MIN_HARD_ACCURACY_FLOOR:
            continue

        if best is None or accuracy > best["accuracy"]:
            best = {"timeframe": timeframe, "accuracy": accuracy, "family": family or "rf"}
    return best


def get_promoted_accuracy(symbol: str, timeframe: str) -> Optional[float]:
    """Return promoted accuracy for a symbol/timeframe pair (if valid and above hard floor)."""
    metadata, _ = _find_best_metadata(symbol, timeframe)
    if metadata is None:
        return None
    accuracy = metadata.get("accuracy") or metadata.get("validation", {}).get("candidate_model_accuracy")
    if accuracy is None:
        return None
    accuracy = float(accuracy)
    if accuracy < MIN_HARD_ACCURACY_FLOOR:
        return None
    return accuracy


def _latest_versioned_metadata(prefix: str, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
    candidates: list[tuple[str, Dict[str, Any]]] = []
    for path in MODEL_DIR.glob(f"{prefix}_v*.metadata.json"):
        metadata = _read_json(path)
        if metadata and metadata.get("symbol") == symbol and metadata.get("timeframe") == timeframe:
            candidates.append((path.stem, metadata))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def get_model_status() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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

            active_model_version = None
            active_accuracy = None
            last_retrain_at = None
            last_retrain_outcome = None
            demoted = False
            demotion_reason = ""

            metadata, family = _find_best_metadata(symbol, timeframe)
            if metadata is not None:
                # Extract info — handle both old (RF/GBM) and new (ultra) metadata shapes
                accuracy = metadata.get("accuracy") or metadata.get("validation", {}).get("candidate_model_accuracy")
                if accuracy is not None:
                    active_accuracy = float(accuracy)
                version = metadata.get("version") or metadata.get("promoted_at") or metadata.get("trained_at")
                active_model_version = f"ultra_{symbol}_{timeframe}" if family == "ultra" else (
                    f"{make_model_key(symbol, timeframe, family or 'rf')}_v{version}" if version else f"{family}_{symbol}_{timeframe}"
                )
                promoted = (
                    metadata.get("validation", {}).get("promoted")
                    if "validation" in metadata
                    else metadata.get("promoted")
                )
                if promoted is False:
                    last_retrain_outcome = "rejected"
                    demoted = bool(metadata.get("validation", {}).get("demoted"))
                    demotion_reason = str(metadata.get("validation", {}).get("demotion_reason", ""))
                else:
                    last_retrain_outcome = "promoted"
                last_retrain_at = metadata.get("promoted_at") or metadata.get("trained_at")

            # LEGACY fallback path — populate if new metadata lookup returned nothing
            if active_model_version is None and model_name:
                fam2 = "gbm" if timeframe == "15min" else "rf"
                symbol_model_name = make_model_key(symbol, timeframe, fam2)
                fallback_model_name = base_model_key_for_timeframe(timeframe)
                active_metadata: Optional[Dict[str, Any]] = None
                active_name: Optional[str] = None
                for candidate_name in [symbol_model_name, fallback_model_name]:
                    active_metadata_path = MODEL_DIR / f"{candidate_name}.metadata.json"
                    meta2 = _read_json(active_metadata_path)
                    if meta2 and meta2.get("symbol") == symbol and meta2.get("timeframe") == timeframe:
                        active_metadata = meta2
                        active_name = candidate_name
                        break
                if active_metadata is not None and active_name is not None:
                    v = active_metadata.get("version")
                    active_model_version = f"{active_name}_v{v}" if v else active_name
                    active_accuracy = active_metadata.get("validation", {}).get("candidate_model_accuracy")
                latest_metadata = _latest_versioned_metadata(symbol_model_name, symbol, timeframe)
                if latest_metadata is None:
                    latest_metadata = _latest_versioned_metadata(fallback_model_name, symbol, timeframe)
                if latest_metadata is None:
                    latest_metadata = active_metadata
                if latest_metadata is not None:
                    last_retrain_at = latest_metadata.get("trained_at")
                    last_retrain_outcome = "promoted" if latest_metadata.get("validation", {}).get("promoted") else "rejected"

            # Normalize active accuracy — HARD FLOOR: if accuracy < 0.60, treat as NO ACTIVE MODEL
            if active_accuracy is not None:
                active_accuracy = float(active_accuracy)
                below_floor = active_accuracy < MIN_HARD_ACCURACY_FLOOR
                weak_signal = below_floor or active_accuracy < 0.62
                if below_floor:
                    active_model_version = None
            else:
                weak_signal = True

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
                "weak_signal": weak_signal,
                "below_hard_floor": demoted or (active_accuracy is not None and active_accuracy < MIN_HARD_ACCURACY_FLOOR),
                "demotion_reason": demotion_reason,
            })

    rows.sort(key=lambda row: (not row["history_available"], row["weak_signal"], row["symbol"], row["timeframe"]))
    return rows
