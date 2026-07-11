"""
Hybrid AI Predictor - Combines ML models for forex predictions
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from utils.logger import TradingLogger
from config.settings import MODEL_DIR
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
        self._pending_retrain_buffer: List = []
        
        # Initialize models
        self._initialize_models()
        
        # Load pre-trained models if available
        if autoload_models:
            self._load_saved_models()
    
    def _initialize_models(self):
        """Initialize ML models with optimized parameters"""
        logger.info("Initializing AI models...")
        
        # Fast Random Forest (5-minute predictions)
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
        
        # Initialize scalers for each timeframe
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
                            timeframe = stem.replace("scaler_", "", 1)
                            self.scalers[timeframe] = loaded
                            logger.success(f"Loaded scaler: {timeframe}")
                            continue

                        self.models[stem] = loaded
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
                     model_type: str = 'auto') -> Prediction:
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
                # Try to find model for this timeframe
                model_key = f'rf_{timeframe}' if f'rf_{timeframe}' in self.models else None
                if model_key is None:
                    # Try gbm for 15min
                    if timeframe == '15min' and 'gbm_15min' in self.models:
                        model_key = 'gbm_15min'
                    # Fallback to any available model
                    elif len(self.models) > 0:
                        model_key = list(self.models.keys())[0]
                    else:
                        logger.warning("No models available")
                        return self._get_neutral_prediction(timeframe)
            else:
                model_key = f'{model_type}_{timeframe}'
            
            if model_key not in self.models:
                logger.warning(f"Model {model_key} not found, using default")
                if len(self.models) > 0:
                    model_key = list(self.models.keys())[0]
                else:
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
            if timeframe in self.scalers:
                try:
                    scaler = self.scalers[timeframe]
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
                   model_name: str = 'rf_5min', save: bool = True):
        """
        Train a model
        """
        try:
            if model_name in self.models:
                logger.info(f"Training {model_name}...")
                self.models[model_name].fit(X_train, y_train)
                self.is_trained = True
                
                if save:
                    # Save model
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    model_path = MODEL_DIR / f"{model_name}.joblib"
                    joblib.dump(self.models[model_name], model_path)
                    logger.success(f"Model saved: {model_path}")
                    
                    # Also save scaler
                    scaler_path = MODEL_DIR / f"scaler_{model_name.split('_')[1]}.joblib"
                    joblib.dump(self.scalers[model_name.split('_')[1]], scaler_path)
            
        except Exception as e:
            logger.error(f"Training error: {e}")
    
    async def update_with_new_data(self, df: pd.DataFrame,
                                 actual_direction: int,
                                 timeframe: str = '5min'):
        """Online learning from real trade outcomes."""
        try:
            features = await self.prepare_features(df)
            if features.size == 0:
                return

            if timeframe in self.scalers:
                self.scalers[timeframe].partial_fit(features)

            model_key = f"rf_{timeframe}"
            sgd = self.models.get("sgd_online")

            if sgd is not None and hasattr(sgd, "partial_fit"):
                X = self.scalers.get(timeframe, self.scalers.get("5min"))
                if X:
                    X_scaled = X.transform(features)
                    try:
                        sgd.partial_fit(X_scaled, [actual_direction], classes=self._sgd_classes)
                    except ValueError:
                        sgd.partial_fit(X_scaled, [actual_direction])
                    logger.info(f"Online-updated sgd_online with outcome={actual_direction}")
            elif model_key in self.models:
                self._pending_retrain_buffer.append((features, actual_direction))
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