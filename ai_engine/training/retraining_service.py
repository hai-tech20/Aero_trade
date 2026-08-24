"""Model retraining orchestration and versioned metadata persistence."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.settings import AI_CONFIG, DATA_DIR, MODEL_DIR, TRADING_CONFIG
from ai_engine.models.model_keys import make_model_key, make_scaler_key, base_model_key_for_timeframe
from core.ledger.trade_ledger import (
    get_recent_closed_trades_for_training,
    get_mistake_rows,
    count_closed_trades_since,
)
from data.yfinance_source import load_supplementary_history
from utils.logger import TradingLogger

logger = TradingLogger("RetrainingService")

MODEL_KEYS: Dict[str, str] = {
    "1min": "rf_1min",
    "5min": "rf_5min",
    "15min": "gbm_15min",
    "1h": "rf_1h",
    "4h": "rf_4h",
    "1d": "rf_1d",
}


@dataclass
class TrainingDataset:
    symbol: str
    timeframe: str
    history_path: str
    history_rows: int
    start_time: str
    end_time: str
    history_samples: int
    trade_samples: int
    source: str


class RetrainingService:
    def __init__(self, predictor: Any):
        self.predictor = predictor
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._trigger_event = asyncio.Event()
        self._running_retrain = False
        self._scheduled_interval_seconds = int(AI_CONFIG.get("retrain_interval_seconds", 24 * 60 * 60))
        self._buffer_threshold = int(AI_CONFIG.get("retrain_buffer_threshold", 50))
        self._last_retrain_at: Optional[str] = None
        self._last_retrain_reason: Optional[str] = None
        self._scheduler_test_events: List[Dict[str, Any]] = []

    def set_test_config(self, buffer_threshold: Optional[int] = None, interval_seconds: Optional[int] = None):
        if buffer_threshold is not None:
            self._buffer_threshold = int(buffer_threshold)
        if interval_seconds is not None:
            self._scheduled_interval_seconds = int(interval_seconds)

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._task is not None and not self._task.done(),
            "buffer_threshold": self._buffer_threshold,
            "scheduled_interval_seconds": self._scheduled_interval_seconds,
            "pending_buffer_size": len(self.predictor._pending_retrain_buffer),
            "last_retrain_at": self._last_retrain_at,
            "last_retrain_reason": self._last_retrain_reason,
            "test_events": self._scheduler_test_events[-10:],
        }

    async def start(self):
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            f"Retraining scheduler started (threshold={self._buffer_threshold}, interval={self._scheduled_interval_seconds}s)"
        )

    async def stop(self):
        if self._task is None:
            return
        self._stop_event.set()
        self._trigger_event.set()
        try:
            await self._task
        finally:
            self._task = None
        logger.info("Retraining scheduler stopped")

    def notify_new_trade_sample(self, sample: Dict[str, Any]):
        self.predictor._pending_retrain_buffer.append(sample)
        size = len(self.predictor._pending_retrain_buffer)
        logger.info(f"Queued retrain sample. Pending buffer size={size}")
        if size >= self._buffer_threshold:
            logger.info(
                f"Retrain trigger reached by buffer threshold ({size}/{self._buffer_threshold})"
            )
            self._scheduler_test_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "trigger": "buffer_threshold",
                    "buffer_size": size,
                }
            )
            self._trigger_event.set()

    async def _scheduler_loop(self):
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._trigger_event.wait(), timeout=self._scheduled_interval_seconds)
                trigger_reason = "buffer_threshold"
            except asyncio.TimeoutError:
                trigger_reason = "interval"
            if self._stop_event.is_set():
                break
            self._trigger_event.clear()
            await self.run_retraining(reason=trigger_reason)

    def _all_symbol_timeframe_combos(self) -> List[Tuple[str, str]]:
        combos: List[Tuple[str, str]] = []
        for symbol in TRADING_CONFIG.get("pairs", ["EURUSD"]):
            for timeframe in TRADING_CONFIG.get("timeframes", ["5min"]):
                if timeframe in MODEL_KEYS:
                    combos.append((symbol, timeframe))
        return combos

    async def run_retraining(
        self,
        reason: str,
        targets: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        if self._running_retrain:
            logger.warning("Retrain request skipped because another retrain is already running")
            return {"status": "skipped", "reason": "already_running"}

        self._running_retrain = True
        self._last_retrain_reason = reason
        try:
            results: Dict[str, Any] = {"status": "completed", "reason": reason, "models": []}
            combinations = targets if targets is not None else self._all_symbol_timeframe_combos()
            for symbol, timeframe in combinations:
                if timeframe not in MODEL_KEYS:
                    continue
                history_df, dataset_info = await self._load_verified_history(symbol, timeframe)
                if history_df is None or dataset_info is None or len(history_df) < 200:
                    logger.warning(f"Skipping retrain for {symbol} {timeframe}: insufficient verified history")
                    continue

                X_hist, y_hist = await self._build_history_training_set(history_df)
                if len(X_hist) < 50:
                    logger.warning(f"Skipping retrain for {symbol} {timeframe}: insufficient historical samples")
                    continue

                X_trade, y_trade = self._build_trade_outcome_training_set(timeframe)
                X, y = self._merge_training_sets(X_hist, y_hist, X_trade, y_trade)
                if len(X) < 50:
                    continue

                model_result = await self._train_and_maybe_promote(
                    symbol=symbol,
                    timeframe=timeframe,
                    X=X,
                    y=y,
                    dataset_info=dataset_info,
                    reason=reason,
                )
                if model_result:
                    results["models"].append(model_result)

            self._last_retrain_at = datetime.now().isoformat()
            if results["models"]:
                self.predictor._pending_retrain_buffer.clear()
            return results
        finally:
            self._running_retrain = False

    async def _load_verified_history(self, symbol: str, timeframe: str) -> Tuple[Optional[pd.DataFrame], Optional[TrainingDataset]]:
        history_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
        if not history_path.exists():
            logger.error(f"Verified broker history missing: {history_path}")
            return None, None

        df = pd.read_parquet(history_path)
        if df is None or len(df) == 0:
            logger.error(f"Verified broker history empty: {history_path}")
            return None, None

        if "time" in df.columns:
            time_series = pd.to_datetime(df["time"])
        else:
            time_series = pd.to_datetime(df.index)
        start_time = str(time_series.min())
        end_time = str(time_series.max())

        source = "verified_real_history"
        yfinance_df = load_supplementary_history(symbol, "1d")
        if yfinance_df is not None and len(yfinance_df) > 0:
            source = "verified_real_history_with_yfinance_supplement"

        dataset = TrainingDataset(
            symbol=symbol,
            timeframe=timeframe,
            history_path=str(history_path),
            history_rows=int(len(df)),
            start_time=start_time,
            end_time=end_time,
            history_samples=0,
            trade_samples=0,
            source=source,
        )
        logger.info(
            f"Verified history for training: {symbol} {timeframe} | rows={len(df)} | range={start_time} -> {end_time}"
        )
        return df, dataset

    async def _build_history_training_set(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X: List[np.ndarray] = []
        y: List[int] = []
        lookback = 20
        horizon = 5
        for i in range(lookback, len(df) - horizon):
            window = df.iloc[i - lookback:i]
            features = await self.predictor.prepare_features(window)
            if features.size == 0:
                continue
            current_price = float(df["close"].iloc[i])
            future_price = float(df["close"].iloc[i + horizon])
            price_diff = future_price - current_price

            atr_window = df["high"].iloc[i - lookback:i].astype(float).to_numpy()
            low_window = df["low"].iloc[i - lookback:i].astype(float).to_numpy()
            close_window = df["close"].iloc[i - lookback:i].astype(float).to_numpy()
            if len(close_window) < 2:
                continue
            tr = np.maximum.reduce([
                atr_window[1:] - low_window[1:],
                np.abs(atr_window[1:] - close_window[:-1]),
                np.abs(low_window[1:] - close_window[:-1]),
            ])
            atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr)) if len(tr) else 0.0
            min_move = max(atr * 0.25, current_price * 0.0002, 0.0003)
            if abs(price_diff) < min_move:
                continue
            label = 1 if price_diff > 0 else 0
            X.append(features.flatten())
            y.append(label)
        return np.array(X), np.array(y)

    def _build_trade_outcome_training_set(self, timeframe: str) -> Tuple[np.ndarray, np.ndarray]:
        rows = get_recent_closed_trades_for_training(limit=500)
        X_trade: List[np.ndarray] = []
        y_trade: List[int] = []
        for row in rows:
            features_json = row.get("features_json") or "[]"
            try:
                features = np.array(json.loads(features_json), dtype=float)
            except Exception:
                continue
            if features.size == 0:
                continue
            label = 1 if row.get("outcome") == "win" else 0
            repeats = 3
            for _ in range(repeats):
                X_trade.append(features)
                y_trade.append(label)
        return np.array(X_trade), np.array(y_trade)

    def _merge_training_sets(
        self,
        X_hist: np.ndarray,
        y_hist: np.ndarray,
        X_trade: np.ndarray,
        y_trade: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(X_trade) == 0:
            return X_hist, y_hist

        target_features = X_hist.shape[1]
        aligned_trade = []
        for vec in X_trade:
            if vec.shape[0] > target_features:
                vec = vec[:target_features]
            elif vec.shape[0] < target_features:
                vec = np.pad(vec, (0, target_features - vec.shape[0]), mode="constant", constant_values=0.0)
            aligned_trade.append(vec)
        X = np.vstack([X_hist, np.array(aligned_trade)])
        y = np.concatenate([y_hist, y_trade])
        return X, y

    async def _train_and_maybe_promote(
        self,
        symbol: str,
        timeframe: str,
        X: np.ndarray,
        y: np.ndarray,
        dataset_info: TrainingDataset,
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        family = 'gbm' if timeframe == '15min' else 'rf'
        active_model_name = make_model_key(symbol, timeframe, family)
        fallback_model_name = base_model_key_for_timeframe(timeframe)
        active_model = cast(Any, self.predictor.models.get(active_model_name) or self.predictor.models.get(fallback_model_name))
        if active_model is None:
            return None

        X_train, X_test, y_train, y_test = cast(
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False),
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        baseline_score = None
        try:
            baseline_model = cast(Any, active_model)
            X_test_scaled_arr = np.asarray(X_test_scaled)
            expected_n = int(getattr(baseline_model, "n_features_in_", X_test_scaled_arr.shape[1]))
            X_eval = self._align_matrix(X_test_scaled_arr, expected_n)
            baseline_pred = baseline_model.predict(X_eval)
            baseline_score = float(accuracy_score(y_test, baseline_pred))
        except Exception as e:
            logger.warning(f"Could not score current model {active_model_name}: {e}")

        candidate_model = cast(Any, clone(active_model))
        candidate_model.fit(X_train_scaled, y_train)
        candidate_pred = candidate_model.predict(X_test_scaled)
        candidate_score = float(accuracy_score(y_test, candidate_pred))

        mistakes_before, mistakes_after = self._score_on_mistakes(
            baseline_model=active_model,
            candidate_model=candidate_model,
            timeframe=timeframe,
        )

        promote = baseline_score is None or candidate_score > baseline_score
        logger.info(
            f"Retrain validation for {active_model_name}: current={baseline_score} candidate={candidate_score} | mistakes current={mistakes_before} candidate={mistakes_after} | promote={promote}"
        )

        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "model_name": active_model_name,
            "version": version,
            "symbol": symbol,
            "timeframe": timeframe,
            "trained_at": datetime.now().isoformat(),
            "training_reason": reason,
            "training_source": dataset_info.source,
            "verified_real_history": True,
            "history_path": dataset_info.history_path,
            "history_rows": dataset_info.history_rows,
            "history_date_range": {
                "start": dataset_info.start_time,
                "end": dataset_info.end_time,
            },
            "historical_samples": int(len(X) - len(get_recent_closed_trades_for_training(limit=500)) * 3),
            "trade_outcome_rows_since_last_run": int(count_closed_trades_since(self._last_retrain_at)),
            "validation": {
                "current_model_accuracy": baseline_score,
                "candidate_model_accuracy": candidate_score,
                "mistake_accuracy_before": mistakes_before,
                "mistake_accuracy_after": mistakes_after,
                "promoted": promote,
            },
        }

        active_scaler_name = make_scaler_key(symbol, timeframe)
        versioned_model_path = MODEL_DIR / f"{active_model_name}_v{version}.joblib"
        versioned_scaler_path = MODEL_DIR / f"{active_scaler_name}_v{version}.joblib"
        metadata_path = MODEL_DIR / f"{active_model_name}_v{version}.metadata.json"
        joblib.dump(candidate_model, versioned_model_path)
        joblib.dump(scaler, versioned_scaler_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if promote:
            active_model_path = MODEL_DIR / f"{active_model_name}.joblib"
            active_scaler_path = MODEL_DIR / f"{active_scaler_name}.joblib"
            active_metadata_path = MODEL_DIR / f"{active_model_name}.metadata.json"
            joblib.dump(candidate_model, active_model_path)
            joblib.dump(scaler, active_scaler_path)
            active_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.predictor.models[active_model_name] = candidate_model
            self.predictor.scalers[active_scaler_name] = scaler
            self.predictor.scalers[timeframe] = scaler

        return {
            "model_name": active_model_name,
            "version": version,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "mistake_accuracy_before": mistakes_before,
            "mistake_accuracy_after": mistakes_after,
            "promoted": promote,
            "metadata_path": str(metadata_path),
        }

    def _score_on_mistakes(
        self,
        baseline_model: Any,
        candidate_model: Any,
        timeframe: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        rows = get_mistake_rows(limit=200)
        if not rows:
            return None, None

        X: List[np.ndarray] = []
        y: List[int] = []
        for row in rows:
            try:
                vec = np.array(json.loads(row.get("features_json") or "[]"), dtype=float)
            except Exception:
                continue
            if vec.size == 0:
                continue
            X.append(vec)
            y.append(0)
        if not X:
            return None, None

        target_features = int(getattr(candidate_model, "n_features_in_", len(X[0])))
        matrix_rows: List[np.ndarray] = []
        for vec in X:
            if vec.shape[0] > target_features:
                vec = vec[:target_features]
            elif vec.shape[0] < target_features:
                vec = np.pad(vec, (0, target_features - vec.shape[0]), mode="constant", constant_values=0.0)
            matrix_rows.append(vec)
        matrix = np.asarray(matrix_rows)
        scaler = self.predictor.scalers.get(timeframe)
        if scaler is not None and hasattr(scaler, "transform"):
            try:
                matrix = scaler.transform(matrix)
            except Exception:
                pass
        baseline_score = None
        try:
            baseline_pred = baseline_model.predict(self._align_matrix(matrix, int(getattr(baseline_model, "n_features_in_", matrix.shape[1]))))
            baseline_score = float(accuracy_score(y, baseline_pred))
        except Exception:
            baseline_score = None
        try:
            candidate_pred = candidate_model.predict(matrix)
            candidate_score = float(accuracy_score(y, candidate_pred))
        except Exception:
            candidate_score = None
        return baseline_score, candidate_score

    def _align_matrix(self, X: np.ndarray, expected_n: int) -> np.ndarray:
        if X.shape[1] == expected_n:
            return X
        if X.shape[1] > expected_n:
            return X[:, :expected_n]
        pad = expected_n - X.shape[1]
        return np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
