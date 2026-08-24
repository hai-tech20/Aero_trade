"""
Hybrid AI Predictor - Combines ML models for forex predictions
"""

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ai_engine.models.model_keys import make_model_key, make_scaler_key, base_model_key_for_timeframe
from ai_engine.models.feature_engine import compute_all_features, feature_count_target
from ai_engine.models.ultra_ensemble import UltraEnsemble
from ai_engine.training.retraining_service import RetrainingService
from ai_engine.training.model_status import get_best_timeframe_for_symbol
from config.settings import AI_CONFIG, MODEL_DIR
from utils.logger import TradingLogger

warnings.filterwarnings('ignore')
logger = TradingLogger("AI-Predictor")


@dataclass
class Prediction:
    """AI prediction result"""
    direction: int  # 1 = BUY, -1 = SELL, 0 = HOLD
    confidence: float  # 0.0 to 1.0
    price_target: float
    timeframe: str
    features: List[str]
    timestamp: float
    model_used: str

class HybridAIPredictor:
    """
    Hybrid AI system combining multiple ML models
    Fast inference for real-time trading
    """
    
    def __init__(self, autoload_models: bool = True):
        logger.info("AI-Predictor initialized")
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.ultra_ensembles: Dict[str, UltraEnsemble] = {}
        self.feature_names: List[str] = []
        self.is_trained: bool = False
        self._pending_retrain_buffer: List[Dict[str, Any]] = []
        self.training_metadata: Dict[str, Dict[str, Any]] = {}
        self.retraining_service: Optional[RetrainingService] = None
        
        # Initialize models
        self._initialize_models()
        
        # Load pre-trained models if available
        if autoload_models:
            self._load_saved_models()
            self._load_ultra_ensembles()

        self.retraining_service = RetrainingService(self)
    
    def _initialize_models(self):
        """Initialize ML models with optimized parameters"""
        logger.info("Initializing AI models...")
        
        # Base prototypes used to materialize per-symbol models on demand
        self.models['rf_5min'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,  # Safer on Windows/locked-down environments
            verbose=0
        )
        
        # Gradient Boosting (15-minute predictions)
        self.models['gbm_15min'] = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.85,
            random_state=42,
            verbose=0
        )
        
        # Online-adjustment model (supports partial_fit)
        self.models['sgd_online'] = SGDClassifier(
            loss='log_loss',
            max_iter=1000,
            random_state=42,
            warm_start=True,
            class_weight='balanced',
        )
        self._sgd_classes = np.array([0, 1])

        # Lightweight model for 1-minute (if data available)
        self.models['rf_1min'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )

        # Hourly, multi-hour and daily models for longer horizons
        self.models['rf_1h'] = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        self.models['rf_4h'] = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        self.models['rf_1d'] = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
            verbose=0,
        )
        
        # Initialize default scalers for each timeframe; per-symbol scalers are loaded/trained on demand
        for timeframe in ['1min', '5min', '15min', '1h', '4h', '1d']:
            self.scalers[timeframe] = StandardScaler()
        
        logger.success(f"Initialized {len(self.models)} AI models")
    
    def _load_saved_models(self):
        """Load pre-trained models from disk"""
        try:
            if MODEL_DIR.exists():
                model_files = list(MODEL_DIR.glob("*.joblib"))
                if model_files:
                    for model_file in model_files:
                        stem = model_file.stem
                        if stem.startswith("ultra_") and (
                            "_sub_" in stem or stem.endswith("_ensemble_meta")
                        ):
                            continue

                        loaded = joblib.load(model_file)

                        # Keep scalers out of the models dict
                        if stem.startswith("scaler_"):
                            scaler_key = stem
                            self.scalers[scaler_key] = loaded
                            parts = stem.split('_')
                            if len(parts) == 2:
                                self.scalers[parts[-1]] = loaded
                            logger.success(f"Loaded scaler: {scaler_key}")
                            continue

                        self.models[stem] = loaded
                        metadata_path = model_file.with_suffix('.metadata.json')
                        if metadata_path.exists():
                            try:
                                self.training_metadata[stem] = json.loads(metadata_path.read_text(encoding='utf-8'))
                            except Exception as e:
                                logger.warning(f"Could not load metadata for {stem}: {e}")
                        # Avoid joblib parallel backends causing permission issues on some Windows setups
                        try:
                            if hasattr(loaded, "set_params") and hasattr(loaded, "n_jobs"):
                                loaded.set_params(n_jobs=1)
                        except Exception:
                            pass
                        logger.success(f"Loaded model: {stem}")
                    self.is_trained = True
        except Exception as e:
            logger.warning(f"Could not load saved models: {e}")

    @staticmethod
    def _ultra_key(symbol: str, timeframe: str) -> str:
        return f"ultra_{symbol}_{timeframe}"

    @staticmethod
    def _parse_ultra_stem(stem: str) -> Optional[tuple[str, str]]:
        if not stem.startswith("ultra_"):
            return None
        body = stem[len("ultra_"):]
        for tf in ("1min", "5min", "15min", "1h", "4h", "1d"):
            suffix = f"_{tf}"
            if body.endswith(suffix):
                return body[: -len(suffix)], tf
        return None

    def _load_ultra_ensembles(self) -> None:
        if not AI_CONFIG.get("use_ultra_ensemble", True) or not MODEL_DIR.exists():
            return
        min_acc = float(AI_CONFIG.get("min_model_accuracy", 0.60))
        for meta_path in MODEL_DIR.glob("ultra_*_*.metadata.json"):
            # NOTE: meta_path.stem only strips the LAST suffix, leaving
            # "ultra_SYMBOL_TF.metadata" (not "ultra_SYMBOL_TF") for these
            # double-extension filenames — strip the full ".metadata.json"
            # suffix explicitly instead, or _parse_ultra_stem never matches
            # and every promoted ultra model silently fails to load.
            name = meta_path.name
            if not name.endswith(".metadata.json"):
                continue
            parsed = self._parse_ultra_stem(name[: -len(".metadata.json")])
            if parsed is None:
                continue
            symbol, timeframe = parsed
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            accuracy = float(
                meta.get("accuracy")
                or meta.get("validation", {}).get("candidate_model_accuracy")
                or 0.0
            )
            promoted = meta.get("validation", {}).get("promoted", meta.get("promoted", True))
            if promoted is False or accuracy < min_acc:
                continue
            prefix = str(MODEL_DIR / f"ultra_{symbol}_{timeframe}")
            ensemble = UltraEnsemble(symbol=symbol, timeframe=timeframe, n_jobs=1)
            if ensemble.load(prefix):
                key = self._ultra_key(symbol, timeframe)
                self.ultra_ensembles[key] = ensemble
                logger.success(
                    f"Loaded ultra ensemble: {key} (accuracy={accuracy:.4f}, oof={ensemble._oof_accuracy:.4f})"
                )

    def _parse_model_name(self, model_name: str) -> tuple[str, Optional[str], str]:
        parts = model_name.split('_')
        if len(parts) == 2:
            return parts[0], None, parts[1]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) >= 4:
            return parts[0], '_'.join(parts[1:-1]), parts[-1]
        return parts[0], None, parts[-1] if len(parts) else ''

    def _resolve_model_for_training(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]
        family, _, timeframe = self._parse_model_name(model_name)
        prototype_name = f"{family}_{timeframe}"
        if prototype_name in self.models:
            model = clone(self.models[prototype_name])
            self.models[model_name] = model
            return model
        raise KeyError(f"No base model prototype found for {model_name}")

    async def prepare_features(self, df: pd.DataFrame, 
                             lookback: int = 20) -> np.ndarray:
        """
        Prepare features from price data
        Returns: Feature array for prediction
        """
        if len(df) < lookback:
            return np.array([])
        
        features = []
        feature_names = []
        
        recent = df.iloc[-lookback:].copy()
        recent = recent.astype({col: float for col in ['open', 'high', 'low', 'close', 'volume'] if col in recent.columns})
        close = recent['close'].to_numpy(dtype=float)
        high = recent['high'].to_numpy(dtype=float)
        low = recent['low'].to_numpy(dtype=float)
        open_ = recent['open'].to_numpy(dtype=float)
        volume = recent['volume'].to_numpy(dtype=float) if 'volume' in recent.columns else np.zeros(len(recent), dtype=float)

        returns = np.diff(close) / np.maximum(close[:-1], 1e-9)
        features.append(float(np.nan_to_num(np.mean(returns), nan=0.0)))
        feature_names.append('mean_return')
        features.append(float(np.nan_to_num(np.std(returns), nan=0.0)))
        feature_names.append('volatility')
        features.append(float(np.nan_to_num(pd.Series(returns).skew(), nan=0.0)))
        feature_names.append('skewness')
        features.append(float(np.nan_to_num(pd.Series(returns).kurtosis(), nan=0.0)))
        feature_names.append('kurtosis')

        delta = close[1:] - close[:-1]
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else (np.mean(gains) if len(gains) > 0 else 0.0)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else (np.mean(losses) if len(losses) > 0 else 0.0)
        rsi = 50.0
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        features.append(float(np.nan_to_num(rsi, nan=50.0)))
        feature_names.append('rsi')

        def ema(values: np.ndarray, period: int) -> np.ndarray:
            if len(values) < period:
                return np.full_like(values, values.mean() if len(values) > 0 else 0.0)
            alpha = 2.0 / (period + 1)
            ema_values = np.zeros_like(values)
            ema_values[:period] = np.mean(values[:period])
            for i in range(period, len(values)):
                ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i - 1]
            return ema_values

        ma5 = np.mean(close[-5:]) if len(close) >= 5 else float(close[-1])
        ma10 = np.mean(close[-10:]) if len(close) >= 10 else float(close[-1])
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else float(close[-1])
        ma50 = np.mean(close[-50:]) if len(close) >= 50 else float(close[-1])
        features.extend([
            float((close[-1] - ma5) / np.maximum(ma5, 1e-9)),
            float((ma5 - ma10) / np.maximum(ma10, 1e-9)),
            float((ma10 - ma20) / np.maximum(ma20, 1e-9)),
            float((ma20 - ma50) / np.maximum(ma50, 1e-9)),
        ])
        feature_names.extend(['ma5_diff', 'ma5_10_diff', 'ma10_20_diff', 'ma20_50_diff'])

        atr = 0.0
        if len(close) >= 2:
            tr = np.maximum.reduce([
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ])
            atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        features.append(float(np.nan_to_num(atr, nan=0.0)))
        feature_names.append('atr')

        if len(close) >= 20:
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            upper_band = sma20 + 2 * std20
            lower_band = sma20 - 2 * std20
            bb_width = float((upper_band - lower_band) / np.maximum(close[-1], 1e-9))
            bb_position = float((close[-1] - lower_band) / np.maximum((upper_band - lower_band), 1e-9))
        else:
            bb_width = 0.0
            bb_position = 0.5
        features.extend([bb_width, bb_position])
        feature_names.extend(['bb_width', 'bb_position'])

        macd_line = ema(close, 12)
        signal_line = ema(macd_line, 9)
        macd = float(macd_line[-1] - signal_line[-1]) if len(macd_line) > 0 else 0.0
        features.append(float(np.nan_to_num(macd, nan=0.0)))
        feature_names.append('macd_diff')

        body = np.abs(close - open_)
        bodies = np.abs(high - low)
        body_ratio = float(np.nan_to_num(np.mean(body / np.maximum(bodies, 1e-9)), nan=0.0))
        features.append(body_ratio)
        feature_names.append('body_ratio')

        high_low_ratio = float(np.nan_to_num(np.mean((high - low) / np.maximum(close, 1e-9)), nan=0.0))
        features.append(high_low_ratio)
        feature_names.append('high_low_ratio')

        features.append(float(np.nan_to_num((close[-1] - open_[-1]) / np.maximum(open_[-1], 1e-9), nan=0.0)))
        feature_names.append('close_open_pct')

        volume_pct = 0.0
        if len(volume) > 0 and np.mean(volume) > 0:
            volume_pct = float((volume[-1] - np.mean(volume[-5:])) / np.maximum(np.mean(volume[-5:]), 1e-9))
        features.append(float(np.nan_to_num(volume_pct, nan=0.0)))
        feature_names.append('volume_pct')

        momentum_5 = float((close[-1] - close[-5]) / np.maximum(close[-5], 1e-9)) if len(close) >= 5 else 0.0
        momentum_10 = float((close[-1] - close[-10]) / np.maximum(close[-10], 1e-9)) if len(close) >= 10 else 0.0
        features.extend([momentum_5, momentum_10])
        feature_names.extend(['momentum_5', 'momentum_10'])

        time_index = recent.index[-1]
        if hasattr(time_index, 'weekday'):
            features.append(float(time_index.weekday() / 6.0))
            feature_names.append('day_of_week')
            features.append(float(time_index.hour / 23.0))
            feature_names.append('hour_of_day')
        else:
            features.extend([0.0, 0.0])
            feature_names.extend(['day_of_week', 'hour_of_day'])

        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)

    def _align_feature_matrix(self, features: np.ndarray, expected_n: int) -> np.ndarray:
        X = features.reshape(1, -1) if features.ndim == 1 else features
        if X.shape[1] == expected_n:
            return X
        if X.shape[1] > expected_n:
            return X[:, :expected_n]
        pad = expected_n - X.shape[1]
        return np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    async def _predict_ultra(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        extended_context: Optional[dict] = None,
    ) -> Prediction:
        key = self._ultra_key(symbol, timeframe)
        ensemble = self.ultra_ensembles.get(key)
        if ensemble is None:
            return self._get_neutral_prediction(timeframe)

        features, feature_names = compute_all_features(
            df, lookback=100, symbol=symbol, extended_context=extended_context
        )
        if features.size == 0:
            logger.warning(f"Ultra ensemble feature build failed for {symbol}/{timeframe}")
            return self._get_neutral_prediction(timeframe)

        X = self._align_feature_matrix(features, ensemble.n_features)
        direction, confidence, details = ensemble.predict(X)
        min_confidence = float(AI_CONFIG.get("min_confidence", 0.62))
        if direction == 0 or confidence < min_confidence:
            logger.info(
                f"Ultra ensemble hold for {symbol}/{timeframe} "
                f"(dir={direction}, conf={confidence:.3f} < {min_confidence:.3f}, "
                f"weighted_p={details.get('weighted_prob')}, agree={details.get('submodel_agreement')})"
            )
            return self._get_neutral_prediction(timeframe)

        current_price = float(df["close"].iloc[-1])
        move_pct = min(0.05, max(0.005, confidence * 0.02))
        price_target = (
            current_price * (1 + move_pct)
            if direction == 1
            else current_price * (1 - move_pct)
        )
        self.feature_names = feature_names
        return Prediction(
            direction=direction,
            confidence=confidence,
            price_target=float(price_target),
            timeframe=timeframe,
            features=feature_names,
            timestamp=datetime.now().timestamp(),
            model_used=key,
        )
    
    async def predict(self, df: pd.DataFrame,
                     timeframe: str = '5min',
                     model_type: str = 'auto',
                     symbol: str = 'EURUSD',
                     extended_context: Optional[dict] = None) -> Prediction:
        """
        Make AI prediction on market data
        """
        try:
            ultra_key = self._ultra_key(symbol, timeframe)
            if AI_CONFIG.get("use_ultra_ensemble", True) and ultra_key in self.ultra_ensembles:
                return await self._predict_ultra(df, symbol, timeframe, extended_context=extended_context)

            # Check if we have any models
            if not self.models or len(self.models) == 0:
                logger.warning("No models available, returning neutral prediction")
                return self._get_neutral_prediction(timeframe)
            
            # Validate input data
            if df is None or len(df) < 20:
                logger.warning("Insufficient data for prediction")
                return self._get_neutral_prediction(timeframe)
            
            # Prepare features
            features = await self.prepare_features(df)
            
            if features.size == 0 or features.shape[1] == 0:
                logger.warning("Insufficient features for prediction")
                return self._get_neutral_prediction(timeframe)
            
            # Select model: use symbol-specific model when available, otherwise fallback to the timeframe prototype
            if model_type == 'auto':
                preferred_family = 'gbm' if timeframe == '15min' else 'rf'
                model_key = make_model_key(symbol, timeframe, preferred_family)
                if model_key not in self.models:
                    fallback_model_key = base_model_key_for_timeframe(timeframe)
                    if fallback_model_key in self.models:
                        logger.warning(
                            f"No trained symbol-specific model for {symbol} {timeframe}; using fallback {fallback_model_key}"
                        )
                        model_key = fallback_model_key
                    else:
                        logger.warning(f"No trained model available for {symbol} {timeframe}; returning neutral")
                        return self._get_neutral_prediction(timeframe)
            else:
                model_key = f'{model_type}_{timeframe}'
                if model_key not in self.models:
                    fallback_model_key = base_model_key_for_timeframe(timeframe)
                    if fallback_model_key in self.models:
                        logger.warning(
                            f"Model {model_key} not found; using fallback {fallback_model_key}"
                        )
                        model_key = fallback_model_key
                    else:
                        logger.warning(f"Model {model_key} not found for {symbol} {timeframe}")
                        return self._get_neutral_prediction(timeframe)
            
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not found for {symbol} {timeframe}")
                return self._get_neutral_prediction(timeframe)
            
            model = self.models[model_key]
            
            # Check if model is trained
            if not hasattr(model, 'classes_') or not hasattr(model, 'predict'):
                logger.warning(f"Model {model_key} not trained, returning neutral")
                return self._get_neutral_prediction(timeframe)

            # Align feature vector length to what the model expects (prevents sklearn shape errors)
            expected_n = int(getattr(model, "n_features_in_", features.shape[1]))
            X = features
            if X.shape[1] != expected_n:
                logger.warning(
                    f"Feature length mismatch for {model_key}: got {X.shape[1]}, expected {expected_n}. "
                    "Padding/truncating to match."
                )
                if X.shape[1] > expected_n:
                    X = X[:, :expected_n]
                    self.feature_names = self.feature_names[:expected_n]
                else:
                    pad = expected_n - X.shape[1]
                    X = np.pad(X, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
                    self.feature_names = self.feature_names + [f"pad_{i}" for i in range(pad)]

            # Scale features (only if scaler shape matches)
            features_scaled = X
            scaler_key = make_scaler_key(symbol, timeframe)
            fallback_scaler_key = timeframe
            if scaler_key in self.scalers or fallback_scaler_key in self.scalers:
                try:
                    scaler = self.scalers.get(scaler_key, self.scalers[fallback_scaler_key])
                    if not hasattr(scaler, 'mean_'):
                        scaler.fit(X)
                    elif hasattr(scaler, "n_features_in_") and int(scaler.n_features_in_) != int(X.shape[1]):
                        raise ValueError(
                            f"Scaler expects {int(scaler.n_features_in_)} features but got {int(X.shape[1])}"
                        )
                    features_scaled = scaler.transform(X)
                except Exception as e:
                    logger.warning(f"Scaler error for {timeframe}, using raw features: {e}")
                    features_scaled = X
            
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                    class_index = int(np.argmax(probabilities))
                    predicted_label = int(model.classes_[class_index]) if hasattr(model, 'classes_') else int(class_index)
                    confidence = float(np.max(probabilities))
                elif hasattr(model, 'decision_function'):
                    score = model.decision_function(features_scaled)[0]
                    probabilities = 1.0 / (1.0 + np.exp(-score))
                    predicted_label = 1 if score > 0 else 0
                    confidence = float(np.abs(probabilities - 0.5) * 2.0)
                else:
                    predicted_label = int(model.predict(features_scaled)[0])
                    confidence = 0.5

                direction = 1 if predicted_label == 1 else -1
                min_confidence = float(AI_CONFIG.get('min_confidence', 0.45))
                if confidence < min_confidence:
                    logger.info(f"Prediction confidence below threshold ({confidence:.3f} < {min_confidence:.3f})")
                    return self._get_neutral_prediction(timeframe)

                current_price = float(df['close'].iloc[-1])
                move_pct = min(0.05, max(0.005, confidence * 0.02))
                if direction == 1:
                    price_target = current_price * (1 + move_pct)
                else:
                    price_target = current_price * (1 - move_pct)

                return Prediction(
                    direction=direction,
                    confidence=confidence,
                    price_target=float(price_target),
                    timeframe=timeframe,
                    features=self.feature_names,
                    timestamp=datetime.now().timestamp(),
                    model_used=model_key
                )
            except Exception as e:
                logger.error(f"Model prediction error: {e}")
                return self._get_neutral_prediction(timeframe)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._get_neutral_prediction(timeframe)
    
    def _get_neutral_prediction(self, timeframe: str) -> Prediction:
        """Return neutral prediction on error"""
        return Prediction(
            direction=0,
            confidence=0.0,
            price_target=0.0,
            timeframe=timeframe,
            features=[],
            timestamp=datetime.now().timestamp(),
            model_used='error_fallback'
        )
    
    async def train(self, X_train: np.ndarray, y_train: np.ndarray,
                   model_name: str = 'rf_5min', save: bool = True,
                   metadata: Optional[Dict[str, Any]] = None,
                   version_suffix: Optional[str] = None,
                   promote: bool = True,
                   active_name: Optional[str] = None,
                   scaler_name: Optional[str] = None):
        """
        Train a model
        """
        try:
            if model_name not in self.models:
                try:
                    self.models[model_name] = self._resolve_model_for_training(model_name)
                except KeyError as e:
                    logger.error(f"Training error: {e}")
                    return

            logger.info(f"Training {model_name}...")
            parts = model_name.split('_')
            timeframe = parts[-1]
            resolved_scaler_name = scaler_name or f"scaler_{timeframe}"
            if scaler_name is None and len(parts) == 3:
                resolved_scaler_name = make_scaler_key(parts[1], timeframe)

            active_scaler = self.scalers.get(resolved_scaler_name, self.scalers.get(timeframe))
            if active_scaler is None:
                active_scaler = StandardScaler()
                self.scalers[resolved_scaler_name] = active_scaler
                self.scalers[timeframe] = active_scaler

            if X_train is not None and len(X_train):
                try:
                    active_scaler.fit(X_train)
                    X_train_scaled = active_scaler.transform(X_train)
                except Exception as scaler_error:
                    logger.warning(f"Scaler training failed for {resolved_scaler_name}: {scaler_error}")
                    X_train_scaled = X_train
            else:
                X_train_scaled = X_train

            self.models[model_name].fit(X_train_scaled, y_train)
            self.is_trained = True

            if save:
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                suffix = f"_v{version_suffix}" if version_suffix else ""
                model_path = MODEL_DIR / f"{model_name}{suffix}.joblib"
                scaler_path = MODEL_DIR / f"{resolved_scaler_name}{suffix}.joblib"
                joblib.dump(self.models[model_name], model_path)
                joblib.dump(active_scaler, scaler_path)
                logger.success(f"Model saved: {model_path}")
                if metadata is not None:
                    metadata_path = MODEL_DIR / f"{model_name}{suffix}.metadata.json"
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                    self.training_metadata[model_name] = metadata
                if promote and suffix:
                    active_model_key = active_name or model_name
                    active_model_path = MODEL_DIR / f"{active_model_key}.joblib"
                    active_scaler_path = MODEL_DIR / f"{resolved_scaler_name}.joblib"
                    joblib.dump(self.models[model_name], active_model_path)
                    joblib.dump(active_scaler, active_scaler_path)
                    if metadata is not None:
                        active_metadata_path = MODEL_DIR / f"{active_model_key}.metadata.json"
                        active_metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    async def update_with_new_data(self, df: pd.DataFrame,
                                 actual_direction: int,
                                 timeframe: str = '5min',
                                 trade_sample: Optional[Dict[str, Any]] = None):
        """Online learning from real trade outcomes."""
        try:
            features = await self.prepare_features(df)
            if features.size == 0:
                return

            symbol = str((trade_sample or {}).get("symbol") or "EURUSD")
            scaler_key = make_scaler_key(symbol, timeframe)
            scaler = self.scalers.get(scaler_key, self.scalers.get(timeframe))
            if scaler is None:
                scaler = StandardScaler()
                self.scalers[scaler_key] = scaler
                self.scalers[timeframe] = scaler
            scaler.partial_fit(features)

            model_key = make_model_key(symbol, timeframe, 'rf' if timeframe != '15min' else 'gbm')
            sgd = self.models.get("sgd_online")

            if sgd is not None and hasattr(sgd, "partial_fit"):
                active_scaler = self.scalers.get(scaler_key, self.scalers.get(timeframe, self.scalers.get("5min")))
                if active_scaler is not None:
                    X_scaled = active_scaler.transform(features)
                    try:
                        sgd.partial_fit(X_scaled, [actual_direction], classes=self._sgd_classes)
                    except ValueError:
                        sgd.partial_fit(X_scaled, [actual_direction])
                    logger.info(f"Online-updated sgd_online with outcome={actual_direction}")
            sample = trade_sample or {
                "features": features.flatten().tolist(),
                "label": int(actual_direction),
                "timeframe": timeframe,
                "recorded_at": datetime.now().isoformat(),
            }

            if self.retraining_service is not None:
                self.retraining_service.notify_new_trade_sample(sample)
            elif model_key in self.models:
                self._pending_retrain_buffer.append(sample)
                if len(self._pending_retrain_buffer) >= 50:
                    logger.info(f"Retrain buffer full ({len(self._pending_retrain_buffer)} samples) — schedule batch retrain")
        except Exception as e:
            logger.error(f"Online learning error: {e}")
    
    async def get_model_performance(self) -> Dict:
        """Get performance metrics of models"""
        performance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'classes_'):
                # Simplified metrics
                performance[model_name] = {
                    'is_trained': True,
                    'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown',
                    'n_classes': len(model.classes_) if hasattr(model, 'classes_') else 'unknown'
                }
            else:
                performance[model_name] = {
                    'is_trained': False
                }
        
        return performance

    async def start_retraining_scheduler(self):
        if self.retraining_service is not None:
            await self.retraining_service.start()

    async def stop_retraining_scheduler(self):
        if self.retraining_service is not None:
            await self.retraining_service.stop()

    async def run_retraining_now(
        self,
        reason: str = 'manual',
        targets: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        if self.retraining_service is None:
            return {"status": "unavailable"}
        return await self.retraining_service.run_retraining(reason=reason, targets=targets)

    def get_retraining_status(self) -> Dict[str, Any]:
        if self.retraining_service is None:
            return {"running": False}
        return self.retraining_service.get_status()

# Global instance
ai_predictor: Optional[HybridAIPredictor] = None

def get_ai_predictor() -> HybridAIPredictor:
    """
    Lazy singleton accessor to avoid heavy imports (joblib load)
    at module import time.
    """
    global ai_predictor
    if ai_predictor is None:
        ai_predictor = HybridAIPredictor(autoload_models=True)
    return ai_predictor