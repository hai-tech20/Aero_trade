"""
AI Model Training Script
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from ai_engine.models.predictor import get_ai_predictor
from data.market_data import market_data

async def generate_training_data(symbol: str = "EURUSD",
                               timeframe: str = "5min",
                               samples: int = 5000) -> tuple:
    """Generate training data from verified real parquet history only."""
    from config.settings import DATA_DIR

    predictor = get_ai_predictor()
    logger.info(f"Generating training data for {symbol} {timeframe}...")

    history_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
    if not history_path.exists():
        logger.error(f"Verified real history not found: {history_path}")
        return None, None, None, None, None

    df = pd.read_parquet(history_path)
    if df is None or len(df) < 100:
        logger.error(f"Insufficient verified real data in {history_path}")
        return None, None, None, None, None

    if "time" in df.columns:
        time_series = pd.to_datetime(df["time"])
    else:
        time_series = pd.to_datetime(df.index)
    logger.info(
        f"Verified real training data: rows={len(df)} | range={time_series.min()} -> {time_series.max()} | path={history_path}"
    )
    
    # Prepare features and labels
    X = []
    y = []
    
    lookback = 20  # Use last 20 candles for features
    horizon = 5    # Predict 5 candles ahead
    
    for i in range(lookback, len(df) - horizon):
        # Extract window
        window = df.iloc[i-lookback:i]
        
        # Prepare features
        features = await predictor.prepare_features(window)
        
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

    dataset_info = {
        "symbol": symbol,
        "timeframe": timeframe,
        "history_path": str(history_path),
        "history_rows": int(len(df)),
        "date_range": {
            "start": str(time_series.min()),
            "end": str(time_series.max()),
        },
        "training_samples": int(len(X)),
        "source": "verified_real_history",
    }

    return X_train, y_train, X_test, y_test, dataset_info

async def train_all_models():
    """Train all AI models"""
    logger.info("=" * 60)
    logger.info("🤖 AI MODEL TRAINING STARTED")
    logger.info("=" * 60)
    
    try:
        predictor = get_ai_predictor()
        await market_data.connect()

        X_train, y_train, X_test, y_test, dataset_info = await generate_training_data()

        if X_train is None:
            logger.error("Failed to generate training data")
            return False
        
        # Train Random Forest model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_metadata = {
            "trained_at": datetime.now().isoformat(),
            "training_source": "verified_real_history",
            "verified_real_history": True,
            "history_path": dataset_info["history_path"],
            "history_rows": dataset_info["history_rows"],
            "history_date_range": dataset_info["date_range"],
            "training_samples": dataset_info["training_samples"],
        }

        logger.info("Training Random Forest model...")
        await predictor.train(
            X_train, y_train, model_name='rf_5min', save=True,
            metadata={**base_metadata, "model_name": "rf_5min", "version": version},
            version_suffix=version, promote=True,
        )

        logger.info("Training Gradient Boosting model...")
        await predictor.train(
            X_train, y_train, model_name='gbm_15min', save=True,
            metadata={**base_metadata, "model_name": "gbm_15min", "version": version},
            version_suffix=version, promote=True,
        )

        if len(X_train) > 1000:
            logger.info("Training 1-minute model...")
            await predictor.train(
                X_train, y_train, model_name='rf_1min', save=True,
                metadata={**base_metadata, "model_name": "rf_1min", "version": version},
                version_suffix=version, promote=True,
            )

        await evaluate_models(X_test, y_test)

        performance = await predictor.get_model_performance()
        
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
    predictor = get_ai_predictor()
    """Evaluate model performance"""
    if X_test is None or len(X_test) == 0:
        return
    
    logger.info("Evaluating models...")
    
    for model_name in ['rf_5min', 'gbm_15min', 'rf_1min']:
        if model_name in predictor.models and hasattr(predictor.models[model_name], 'predict'):
            try:
                model = predictor.models[model_name]
                
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