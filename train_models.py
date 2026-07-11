"""
AI Model Training Script
"""

import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from ai_engine.models.predictor import ai_predictor
from data.market_data import market_data

async def generate_training_data(symbol: str = "EURUSD",
                               timeframe: str = "5min",
                               samples: int = 5000) -> tuple:
    """Generate training data — prefers real parquet history over mock."""
    from config.settings import get_data_path, DATA_DIR
    logger.info(f"Generating training data for {symbol} {timeframe}...")

    history_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
    if history_path.exists():
        logger.info(f"Loading real history from {history_path}")
        df = pd.read_parquet(history_path)
    else:
        df = await market_data.get_latest_data(symbol, timeframe, lookback=samples * 2)
        if df is not None and len(df) > 100:
            logger.warning("[MOCK DATA] Training on synthetic data — run download_all_history.py first")
    
    if df is None or len(df) < 100:
        logger.error("Insufficient data for training")
        return None, None, None, None
    
    # Prepare features and labels
    X = []
    y = []
    
    lookback = 20  # Use last 20 candles for features
    horizon = 5    # Predict 5 candles ahead
    
    for i in range(lookback, len(df) - horizon):
        # Extract window
        window = df.iloc[i-lookback:i]
        
        # Prepare features
        features = await ai_predictor.prepare_features(window)
        
        if features.size > 0:
            # Calculate label: 1 if price goes up, 0 if down
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + horizon]
            
            label = 1 if future_price > current_price else 0
            
            X.append(features.flatten())
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train/test
    split_idx = int(len(X) * 0.8)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    logger.success(f"Generated {len(X_train)} training and {len(X_test)} test samples")
    
    return X_train, y_train, X_test, y_test

async def train_all_models():
    """Train all AI models"""
    logger.info("=" * 60)
    logger.info("🤖 AI MODEL TRAINING STARTED")
    logger.info("=" * 60)
    
    try:
        # Connect to market data
        await market_data.connect()
        
        # Generate training data
        X_train, y_train, X_test, y_test = await generate_training_data()
        
        if X_train is None:
            logger.error("Failed to generate training data")
            return False
        
        # Train Random Forest model
        logger.info("Training Random Forest model...")
        await ai_predictor.train(X_train, y_train, model_name='rf_5min', save=True)
        
        # Train Gradient Boosting model
        logger.info("Training Gradient Boosting model...")
        await ai_predictor.train(X_train, y_train, model_name='gbm_15min', save=True)
        
        # Train 1-minute model if enough data
        if len(X_train) > 1000:
            logger.info("Training 1-minute model...")
            await ai_predictor.train(X_train, y_train, model_name='rf_1min', save=True)
        
        # Evaluate models
        await evaluate_models(X_test, y_test)
        
        # Get model performance
        performance = await ai_predictor.get_model_performance()
        
        logger.success("=" * 60)
        logger.success("✅ AI MODELS TRAINED SUCCESSFULLY")
        logger.success("=" * 60)
        
        for model_name, stats in performance.items():
            if stats.get('is_trained'):
                logger.info(f"  {model_name}: {stats.get('n_features', '?')} features, "
                           f"{stats.get('n_classes', '?')} classes")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await market_data.disconnect()

async def evaluate_models(X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model performance"""
    if X_test is None or len(X_test) == 0:
        return
    
    logger.info("Evaluating models...")
    
    for model_name in ['rf_5min', 'gbm_15min', 'rf_1min']:
        if model_name in ai_predictor.models and hasattr(ai_predictor.models[model_name], 'predict'):
            try:
                model = ai_predictor.models[model_name]
                
                if hasattr(model, 'classes_'):
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate accuracy
                    accuracy = np.mean(y_pred == y_test)
                    
                    # Calculate precision (for class 1 - buy)
                    if len(y_test) > 0:
                        true_positives = np.sum((y_pred == 1) & (y_test == 1))
                        predicted_positives = np.sum(y_pred == 1)
                        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
                    else:
                        precision = 0
                    
                    logger.info(f"  {model_name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}")
                    
            except Exception as e:
                logger.warning(f"  {model_name} evaluation failed: {e}")

async def main():
    """Main training function"""
    success = await train_all_models()
    
    if success:
        logger.success("Training completed successfully!")
        logger.info("Models saved to: ai_engine/models/saved/")
    else:
        logger.error("Training failed!")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())