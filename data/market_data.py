"""
Market Data Feed - Real-time and historical price data
"""


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, cast

from utils.logger import logger
from config.settings import DATA_DIR, TRADING_CONFIG, DATA_CONFIG

class MarketDataFeed:
    """Real-time market data feed with caching"""
    
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        self.is_connected = False
        
        logger.info("Market Data Feed initialized")
    
    async def connect(self):
        """Connect to data source"""
        try:
            await self._load_historical_data()
            self.is_connected = True
            logger.success("Market Data Feed connected")
        except Exception as e:
            logger.error(f"Failed to connect to data feed: {e}")
            self.is_connected = False
    
    async def _load_historical_data(self):
        """Load historical data from files"""
        for symbol in TRADING_CONFIG["pairs"]:
            for timeframe in TRADING_CONFIG["timeframes"]:
                file_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
                if not file_path.exists():
                    file_path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
                
                if file_path.exists():
                    try:
                        df = cast(pd.DataFrame, pd.read_parquet(file_path))
                        self.data_cache[f"{symbol}_{timeframe}"] = df
                        logger.debug(f"Loaded {symbol} {timeframe}: {len(df)} bars")
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
    
    async def get_latest_data(self, symbol: str, timeframe: str = "5min", 
                            lookback: int = 100) -> Optional[pd.DataFrame]:
        """Get latest market data - tries MT5 first, then cache"""
        cache_key = f"{symbol}_{timeframe}"
        
        try:
            from core.broker.mt5_bridge import mt5_bridge
            if mt5_bridge.connected:
                mt5_data = mt5_bridge.get_historical_data(symbol, timeframe, count=lookback)
                if mt5_data is not None and len(mt5_data) > 0:
                    self.data_cache[cache_key] = cast(pd.DataFrame, mt5_data)
                    return cast(pd.DataFrame, mt5_data.tail(lookback).copy() if len(mt5_data) >= lookback else mt5_data.copy())
        except Exception as e:
            logger.debug(f"MT5 data not available for {symbol}: {e}")
        
        if cache_key in self.data_cache:
            df = cast(pd.DataFrame, self.data_cache[cache_key])
            if len(df) >= lookback:
                return df.iloc[-lookback:].copy()
            return df.copy()
        
        logger.warning(f"No real data available for {cache_key}. Connect a broker and run download_all_history.py.")
        return None
    
    async def update_data(self, symbol: str, timeframe: str, 
                         new_data: pd.DataFrame):
        """Update data with new candles"""
        cache_key = f"{symbol}_{timeframe}"
        
        try:
            if cache_key in self.data_cache:
                # Append new data
                df = cast(pd.DataFrame, self.data_cache[cache_key])
                
                # Remove duplicates
                new_data = cast(pd.DataFrame, new_data[~new_data.index.isin(df.index)])
                
                if len(new_data) > 0:
                    df = cast(pd.DataFrame, pd.concat([df, new_data]))
                    df = cast(pd.DataFrame, df[~df.index.duplicated(keep='last')])
                    
                    # Keep only last N days
                    cutoff = datetime.now() - timedelta(days=DATA_CONFIG["history_days"])
                    df = cast(pd.DataFrame, df[df.index >= cutoff])
                    
                    self.data_cache[cache_key] = df
                    
                    # Save to disk
                    await self._save_data(symbol, timeframe, df)
                    
                    logger.debug(f"Updated {symbol} {timeframe}: +{len(new_data)} bars")
            
        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {e}")
    
    async def _save_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Save data to disk"""
        try:
            file_path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
            df.to_parquet(file_path, compression='gzip')
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        try:
            from core.broker.mt5_bridge import mt5_bridge
            if mt5_bridge.connected:
                tick = mt5_bridge.get_market_price(symbol)
                if tick:
                    return tick.get("bid") or tick.get("ask")
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def get_technical_indicators(self, symbol: str, timeframe: str = "5min") -> Dict:
        """Calculate technical indicators"""
        try:
            df = await self.get_latest_data(symbol, timeframe, lookback=100)
            
            if df is None or len(df) < 20:
                return {}
            
            close = df['close'].to_numpy(dtype=float)
            
            # Moving Averages
            ma5 = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
            ma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
            ma50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
            
            # RSI
            returns = np.diff(close[-15:]) if len(close) >= 15 else np.diff(close)
            gains = returns[returns > 0].sum()
            losses = -returns[returns < 0].sum()
            
            rsi = 0
            if losses != 0:
                rs = gains / losses if gains > 0 else 0
                rsi = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            if len(close) >= 20:
                sma = np.mean(close[-20:])
                std = np.std(close[-20:])
                upper_band = sma + (std * 2)
                lower_band = sma - (std * 2)
                bb_position = (close[-1] - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.5
            else:
                upper_band = lower_band = bb_position = 0
            
            # MACD (simplified)
            if len(close) >= 26:
                ema12 = self._calculate_ema(close[-26:], 12)
                ema26 = self._calculate_ema(close[-26:], 26)
                macd = ema12[-1] - ema26[-1]
            else:
                macd = 0
            
            return {
                'price': float(close[-1]),
                'ma5': float(ma5),
                'ma20': float(ma20),
                'ma50': float(ma50),
                'rsi': float(rsi),
                'bollinger_upper': float(upper_band),
                'bollinger_lower': float(lower_band),
                'bollinger_position': float(bb_position),
                'macd': float(macd),
                'trend': 'up' if ma5 > ma20 > ma50 else 'down' if ma5 < ma20 < ma50 else 'sideways',
                'volatility': float(np.std(close[-20:]) / np.mean(close[-20:]) if len(close) >= 20 else 0)
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {}
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.array([np.mean(prices)] * len(prices))
        
        ema = np.zeros_like(prices)
        multiplier = 2 / (period + 1)
        
        # Start with SMA
        ema[period-1] = np.mean(prices[:period])
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    async def get_market_state(self, symbol: str) -> Dict:
        """Get comprehensive market state"""
        try:
            price = await self.get_current_price(symbol)
            indicators = await self.get_technical_indicators(symbol)
            
            if price is None:
                return {}
            
            # Determine market regime
            regime = "neutral"
            if indicators:
                rsi = indicators.get('rsi', 50)
                volatility = indicators.get('volatility', 0)
                
                if rsi > 70:
                    regime = "overbought"
                elif rsi < 30:
                    regime = "oversold"
                elif volatility > 0.02:
                    regime = "volatile"
                elif indicators.get('trend') == 'up':
                    regime = "trending_up"
                elif indicators.get('trend') == 'down':
                    regime = "trending_down"
            
            return {
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'regime': regime,
                'indicators': indicators
            }
            
        except Exception as e:
            logger.error(f"Error getting market state for {symbol}: {e}")
            return {}
    
    async def disconnect(self):
        """Disconnect from data feed"""
        # Save all data before disconnecting
        for cache_key, df in self.data_cache.items():
            try:
                symbol, timeframe = cache_key.split('_')
                await self._save_data(symbol, timeframe, df)
            except Exception:
                pass
        
        self.is_connected = False
        logger.info("Market Data Feed disconnected")

# Global instance
market_data = MarketDataFeed()