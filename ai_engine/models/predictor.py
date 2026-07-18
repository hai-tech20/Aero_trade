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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ai_engine.models.model_keys import make_model_key, make_scaler_key
from ai_engine.training.retraining_service import RetrainingService
from config.settings import MODEL_DIR
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

        self.retraining_service = RetrainingService(self)
    
    def _initialize_models(self):
        """Initialize ML models with optimized parameters"""
        logger.info("Initializing AI models...")
        
        # Base prototypes used to materialize per-symbol models on demand
        self.models['rf_5min'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,  # Safer on Windows/locked-down environments
            verbose=0
        )
        
        # Gradient Boosting (15-minute predictions)
        self.models['gbm_15min'] = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        
        # Online-adjustment model (supports partial_fit)
        self.models['sgd_online'] = SGDClassifier(
            loss='log_loss', max_iter=1000, random_state=42, warm_start=True
        )
        self._sgd_classes = np.array([0, 1])

        # Lightweight model for 1-minute (if data available)
        self.models['rf_1min'] = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=1
        )
        
        # Initialize default scalers for each timeframe; per-symbol scalers are loaded/trained on demand
        for timeframe in ['1min', '5min', '15min']:
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
        
        # Use recent data
        recent = df.iloc[-lookback:].copy()
        
        # 1. Price-based features
        returns = recent['close'].pct_change().dropna()
        
        # Mean return
        features.append(returns.mean())
        feature_names.append('mean_return')
        
        # Volatility (std of returns)
        features.append(returns.std())
        feature_names.append('volatility')
        
        # Skewness
        features.append(returns.skew())
        feature_names.append('skewness')
        
        # 2. Technical indicators
        close = recent['close'].values
        
        # RSI (simplified) - always produce a value (fallback to neutral 50)
        gains = np.where(np.diff(close) > 0, np.diff(close), 0)
        losses = np.where(np.diff(close) < 0, -np.diff(close), 0)
        
        rsi = 50.0
        if len(gains) > 0 and len(losses) > 0:
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else float(np.mean(gains))
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else float(np.mean(losses))
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        features.append(float(rsi))
        feature_names.append('rsi')
        
        # Moving averages
        ma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
        ma10 = np.mean(close[-10:]) if len(close) >= 10 else close[-1]
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        
        features.extend([
            (close[-1] - ma5) / ma5,
            (ma5 - ma10) / ma10,
            (ma10 - ma20) / ma20
        ])
        feature_names.extend(['ma5_diff', 'ma5_10_diff', 'ma10_20_diff'])
        
        # 3. Volume features (always produce; fallback to 0 if unavailable/unstable)
        volume_pct = 0.0
        if 'volume' in recent.columns:
            volume = recent['volume'].values
            denom = float(np.mean(volume[-5:])) if len(volume) >= 5 else float(np.mean(volume))
            if denom != 0:
                volume_pct = float((volume[-1] - denom) / denom)
        features.append(volume_pct)
        feature_names.append('volume_pct')
        
        # 4. High-low range
        high_low_ratio = (recent['high'].iloc[-1] - recent['low'].iloc[-1]) / recent['close'].iloc[-1]
        features.append(high_low_ratio)
        feature_names.append('high_low_ratio')
        
        # 5. Momentum
        momentum_5 = (close[-1] - close[-5]) / close[-5] if len(close) >= 5 else 0
        momentum_10 = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0
        
        features.extend([momentum_5, momentum_10])
        feature_names.extend(['momentum_5', 'momentum_10'])
        
        self.feature_names = feature_names
        
        return np.array(features).reshape(1, -1)
    
    async def predict(self, df: pd.DataFrame, 
                     timeframe: str = '5min',
                     model_type: str = 'auto',
                     symbol: str = 'EURUSD') -> Prediction:
        """
        Make AI prediction on market data
        """
        try:
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
            
            # Select model
            if model_type == 'auto':
                preferred_family = 'gbm' if timeframe == '15min' else 'rf'
                symbol_key = make_model_key(symbol, timeframe, preferred_family)
                model_key = symbol_key if symbol_key in self.models else None
                if model_key is None:
                    logger.warning(f"No active symbol-qualified model for {symbol} {timeframe}")
                    return self._get_neutral_prediction(timeframe)
            else:
                model_key = f'{model_type}_{timeframe}'
            
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
            
            # Make prediction
            try:
                prediction = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                
                # Get confidence
                confidence = float(np.max(probabilities))
                
                # Determine direction (assuming binary classification: 1=buy, 0=sell)
                # Adjust based on your training labels
                direction = 1 if prediction == 1 else -1
                
                # Calculate price target (simplified)
                current_price = float(df['close'].iloc[-1])
                if direction == 1:
                    price_target = current_price * (1 + confidence * 0.02)  # 2% target
                else:
                    price_target = current_price * (1 - confidence * 0.02)
                
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
            if model_name in self.models:
                logger.info(f"Training {model_name}...")
                self.models[model_name].fit(X_train, y_train)
                self.is_trained = True

                if save:
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    parts = model_name.split('_')
                    timeframe = parts[-1]
                    suffix = f"_v{version_suffix}" if version_suffix else ""
                    model_path = MODEL_DIR / f"{model_name}{suffix}.joblib"
                    resolved_scaler_name = scaler_name or f"scaler_{timeframe}"
                    scaler_path = MODEL_DIR / f"{resolved_scaler_name}{suffix}.joblib"
                    active_scaler = self.scalers.get(resolved_scaler_name, self.scalers.get(timeframe))
                    if active_scaler is None:
                        active_scaler = StandardScaler()
                        self.scalers[resolved_scaler_name] = active_scaler
                        self.scalers[timeframe] = active_scaler
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