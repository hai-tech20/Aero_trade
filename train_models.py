"""
AI Model Training Script
"""

import argparse
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger
from ai_engine.models.predictor import get_ai_predictor
from ai_engine.models.model_keys import make_scaler_key
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
    if df is None or len(df) < 200:
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
    
    timeframe_mapping = {
        "1min": (40, 5),
        "5min": (30, 5),
        "15min": (25, 5),
        "1h": (20, 4),
        "4h": (18, 3),
        "1d": (14, 3),
    }
    lookback, horizon = timeframe_mapping.get(timeframe, (20, 5))
    
    for i in range(lookback, len(df) - horizon):
        window = df.iloc[i-lookback:i]
        features = await predictor.prepare_features(window)
        if features.size == 0:
            continue

        current_price = float(df['close'].iloc[i])
        future_price = float(df['close'].iloc[i + horizon])
        price_diff = future_price - current_price
        atr_window = df['high'].iloc[i-lookback:i].astype(float).to_numpy()
        low_window = df['low'].iloc[i-lookback:i].astype(float).to_numpy()
        close_window = df['close'].iloc[i-lookback:i].astype(float).to_numpy()
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

async def train_all_models(
    symbols: Optional[list[str]] = None,
    timeframes: Optional[list[str]] = None,
    repeat: int = 1,
):
    """Train AI models for all available symbols and timeframes."""
    from config.settings import TRADING_CONFIG

    logger.info("=" * 60)
    logger.info("🤖 AI MODEL TRAINING STARTED")
    logger.info("=" * 60)
    
    try:
        predictor = get_ai_predictor()
        await market_data.connect()

        symbols_to_train = symbols or list(TRADING_CONFIG.get("pairs", ["EURUSD"]))
        timeframes_to_train = timeframes or list(TRADING_CONFIG.get("timeframes", ["5min"]))
        trained_models = 0

        supported_timeframes = ["1min", "5min", "15min", "1h", "4h", "1d"]
        repeat = max(1, int(repeat))
        for loop in range(repeat):
            logger.info(f"Training pass {loop + 1}/{repeat}")
            for symbol in symbols_to_train:
                for timeframe in timeframes_to_train:
                    if timeframe not in supported_timeframes:
                        logger.info(f"Skipping unsupported timeframe {timeframe}")
                        continue

                    X_train, y_train, X_test, y_test, dataset_info = await generate_training_data(symbol, timeframe)
                    if X_train is None:
                        continue

                    version = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_metadata = {
                        "trained_at": datetime.now().isoformat(),
                        "training_source": "verified_real_history",
                        "verified_real_history": True,
                        "history_path": dataset_info["history_path"],
                        "history_rows": dataset_info["history_rows"],
                        "history_date_range": dataset_info["date_range"],
                        "training_samples": dataset_info["training_samples"],
                        "symbol": symbol,
                        "timeframe": timeframe,
                    }

                    model_name = f"gbm_{symbol}_{timeframe}" if timeframe == "15min" else f"rf_{symbol}_{timeframe}"
                    logger.info(f"Training model {model_name} for {symbol} {timeframe}...")
                    await predictor.train(
                        X_train, y_train, model_name=model_name, save=True,
                        metadata={**base_metadata, "model_name": model_name, "version": version},
                        version_suffix=version, promote=True,
                    )

                    await evaluate_models(X_test, y_test, symbol=symbol, timeframe=timeframe, model_name=model_name)
                    trained_models += 1

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

async def evaluate_models(
    X_test: np.ndarray,
    y_test: np.ndarray,
    symbol: str = "EURUSD",
    timeframe: str = "5min",
    model_name: str = "rf_5min",
):
    predictor = get_ai_predictor()
    """Evaluate model performance"""
    if X_test is None or len(X_test) == 0:
        return
    
    logger.info(f"Evaluating model {model_name} for {symbol} {timeframe}...")
    
    model = predictor.models.get(model_name)
    if model is None or not hasattr(model, 'predict'):
        logger.warning(f"Model {model_name} not available for evaluation")
        return

    try:
        if hasattr(model, 'classes_'):
            scaler_key = make_scaler_key(symbol, timeframe)
            scaler = predictor.scalers.get(scaler_key) or predictor.scalers.get(timeframe)
            if scaler is not None and hasattr(scaler, 'transform'):
                try:
                    X_eval = scaler.transform(X_test)
                except Exception as scale_error:
                    logger.warning(f"  Could not scale evaluation features for {model_name}: {scale_error}")
                    X_eval = X_test
            else:
                X_eval = X_test

            y_pred = model.predict(X_eval)
            accuracy = np.mean(y_pred == y_test)
            if len(y_test) > 0:
                true_positives = np.sum((y_pred == 1) & (y_test == 1))
                predicted_positives = np.sum(y_pred == 1)
                precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            else:
                precision = 0
            logger.info(f"  {model_name}: Accuracy={accuracy:.3f}, Precision={precision:.3f}")
    except Exception as e:
        logger.warning(f"  {model_name} evaluation failed: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train AI models for configured trading pairs")
    parser.add_argument("--pairs", nargs="*", help="Symbol list to train (default: configured pairs)")
    parser.add_argument("--timeframes", nargs="*", help="Timeframes to train (default: configured timeframes)")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat training cycles to solidify models")
    return parser.parse_args()


async def main():
    """Main training function"""
    args = parse_args()
    from config.settings import TRADING_CONFIG

    symbols = args.pairs or list(TRADING_CONFIG.get("pairs", ["EURUSD"]))
    timeframes = args.timeframes or list(TRADING_CONFIG.get("timeframes", ["5min"]))
    success = await train_all_models(symbols=symbols, timeframes=timeframes, repeat=args.repeat)
    
    if success:
        logger.success("Training completed successfully!")
        logger.info("Models saved to: ai_engine/models/saved/")
    else:
        logger.error("Training failed!")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())