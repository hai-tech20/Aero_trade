"""
Market Data Feed - Real-time and historical price data
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from utils.logger import logger
from config.settings import DATA_DIR, TRADING_CONFIG, DATA_CONFIG

class MarketDataFeed:
    """Real-time market data feed with caching"""
    
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}
        self.is_connected = False
        
        # Mock data for testing (replace with real API)
        self.mock_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.2800,
            "USDJPY": 145.00,
            "XAUUSD": 1950.00
        }
        
        logger.info("Market Data Feed initialized")
    
    async def connect(self):
        """Connect to data source"""
        try:
            # Load historical data if exists
            await self._load_historical_data()
            
            self.is_connected = True
            logger.success("Market Data Feed connected")
            
        except Exception as e:
            logger.error(f"Failed to connect to data feed: {e}")
            # Fallback to mock data
            await self._initialize_mock_data()
    
    async def _load_historical_data(self):
        """Load historical data from files"""
        for symbol in TRADING_CONFIG["pairs"]:
            for timeframe in TRADING_CONFIG["timeframes"]:
                file_path = DATA_DIR / f"{symbol}_{timeframe}_history.parquet"
                if not file_path.exists():
                    file_path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
                
                if file_path.exists():
                    try:
                        df = pd.read_parquet(file_path)
                        self.data_cache[f"{symbol}_{timeframe}"] = df
                        logger.debug(f"Loaded {symbol} {timeframe}: {len(df)} bars")
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
    
    async def _initialize_mock_data(self):
        """Initialize with mock data for testing"""
        logger.info("Initializing with mock data...")
        
        for symbol in TRADING_CONFIG["pairs"]:
            for timeframe in TRADING_CONFIG["timeframes"]:
                # Generate 1000 candles of mock data
                df = self._generate_mock_data(symbol, timeframe, bars=1000)
                self.data_cache[f"{symbol}_{timeframe}"] = df
        
        self.is_connected = True
        logger.info("Mock data initialized")
    
    def _generate_mock_data(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        """Generate realistic mock price data"""
        np.random.seed(hash(symbol) % 10000)
        
        # Base price from mock prices
        base_price = self.mock_prices.get(symbol, 1.0)
        
        # Generate random walk with drift
        n = bars
        returns = np.random.normal(0.0001, 0.005, n)  # Small positive drift
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        # Convert timeframe to pandas frequency format
        timeframe_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        pandas_freq = timeframe_map.get(timeframe, '5T')
        dates = pd.date_range(end=datetime.now(), periods=n, freq=pandas_freq)
        
        df = pd.DataFrame(index=dates)
        df['open'] = prices * (1 + np.random.normal(0, 0.0005, n))
        df['high'] = df['open'] * (1 + np.abs(np.random.normal(0.001, 0.002, n)))
        df['low'] = df['open'] * (1 - np.abs(np.random.normal(0.001, 0.002, n)))
        df['close'] = prices
        df['volume'] = np.random.randint(1000, 10000, n)
        
        # Ensure high > low
        df['high'] = np.maximum(df[['high', 'open', 'close']].max(axis=1), df['low'] * 1.0001)
        df['low'] = np.minimum(df[['low', 'open', 'close']].min(axis=1), df['high'] * 0.9999)
        
        return df
    
    async def get_latest_data(self, symbol: str, timeframe: str = "5min", 
                            lookback: int = 100) -> Optional[pd.DataFrame]:
        """Get latest market data - tries MT5 first, then cache, then mock"""
        cache_key = f"{symbol}_{timeframe}"
        
        try:
            # Try to get from MT5 if available
            try:
                from core.broker.mt5_bridge import mt5_bridge
                if mt5_bridge.connected:
                    mt5_data = mt5_bridge.get_historical_data(symbol, timeframe, count=lookback)
                    if mt5_data is not None and len(mt5_data) > 0:
                        # Update cache
                        self.data_cache[cache_key] = mt5_data
                        return mt5_data.tail(lookback).copy() if len(mt5_data) >= lookback else mt5_data.copy()
            except Exception as e:
                logger.debug(f"MT5 data not available for {symbol}: {e}")
            
            # Try cache
            if cache_key in self.data_cache:
                df = self.data_cache[cache_key]
                
                # Return last N rows
                if len(df) >= lookback:
                    return df.iloc[-lookback:].copy()
                else:
                    return df.copy()
            else:
                # Generate new data if not in cache
                logger.warning(
                    f"[MOCK DATA] No real history for {cache_key} — generating synthetic data. "
                    "Connect a broker and run download_all_history.py for real training data."
                )
                df = self._generate_mock_data(symbol, timeframe, bars=lookback)
                self.data_cache[cache_key] = df
                return df
                
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    async def update_data(self, symbol: str, timeframe: str, 
                         new_data: pd.DataFrame):
        """Update data with new candles"""
        cache_key = f"{symbol}_{timeframe}"
        
        try:
            if cache_key in self.data_cache:
                # Append new data
                df = self.data_cache[cache_key]
                
                # Remove duplicates
                new_data = new_data[~new_data.index.isin(df.index)]
                
                if len(new_data) > 0:
                    df = pd.concat([df, new_data])
                    df = df[~df.index.duplicated(keep='last')]
                    
                    # Keep only last N days
                    cutoff = datetime.now() - timedelta(days=DATA_CONFIG["history_days"])
                    df = df[df.index >= cutoff]
                    
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
            # For now, return from mock prices with small random movement
            if symbol in self.mock_prices:
                # Add small random walk
                change = np.random.normal(0, 0.0005)
                self.mock_prices[symbol] *= (1 + change)
                return self.mock_prices[symbol]
            
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
            
            close = df['close'].values
            
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
            except:
                pass
        
        self.is_connected = False
        logger.info("Market Data Feed disconnected")

# Global instance
market_data = MarketDataFeed()