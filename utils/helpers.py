"""
Utility functions for trading bot
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from utils.logger import logger

class Timer:
    """Performance timer"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def _enter_(self):
        self.start_time = time.perf_counter()
        return self
    
    def _exit_(self, *args):
        elapsed = time.perf_counter() - self.start_time
        logger.debug(f"{self.name} took {elapsed*1000:.2f}ms")

async def async_timer(coro):
    """Time an async operation"""
    start = time.perf_counter()
    result = await coro
    elapsed = time.perf_counter() - start
    return result, elapsed

def calculate_volatility(prices: List[float], window: int = 20) -> float:
    """Calculate volatility as standard deviation of returns"""
    if len(prices) < 2:
        return 0.0
    
    returns = np.diff(prices) / prices[:-1]
    if len(returns) >= window:
        return np.std(returns[-window:])
    return np.std(returns) if len(returns) > 0 else 0.0

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] range"""
    if features.size == 0:
        return features
    
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    return (features - min_vals) / range_vals

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if not returns:
        return 0.0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def format_price(price: float, decimals: int = 4) -> str:
    """Format price with appropriate decimals"""
    return f"{price:.{decimals}f}"

def format_percentage(value: float) -> str:
    """Format percentage"""
    return f"{value*100:.2f}%"

async def safe_execute(coro, default: Any = None, error_msg: str = "Error"):
    """Safely execute async coroutine with error handling"""
    try:
        return await coro
    except Exception as e:
        logger.error(f"{error_msg}: {e}")
        return default

def generate_trade_id() -> str:
    """Generate unique trade ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = np.random.randint(1000, 9999)
    return f"TRD-{timestamp}-{random_part}"