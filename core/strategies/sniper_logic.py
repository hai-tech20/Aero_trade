"""
Sniper Entry Logic - Advanced Trading Concepts
Implements liquidity sweeps, order blocks, and CHoCH detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import logger

@dataclass
class OrderBlock:
    """Order block structure"""
    price_high: float
    price_low: float
    is_bullish: bool  # True for bullish OB, False for bearish OB
    timestamp: datetime
    strength: float  # 0.0 to 1.0
    tested: bool = False
    broken: bool = False

@dataclass
class LiquiditySweep:
    """Liquidity sweep structure"""
    price_level: float
    direction: str  # 'BUY' or 'SELL'
    timestamp: datetime
    strength: float  # 0.0 to 1.0
    swept: bool = False

@dataclass
class CHoCH:
    """Change of Character structure"""
    price_level: float
    direction: str  # 'BULLISH' or 'BEARISH'
    timestamp: datetime
    timeframe: str
    confirmed: bool = False

class SniperLogic:
    """
    Advanced trading logic for sniper entries
    Implements liquidity sweeps, order blocks, and CHoCH detection
    """
    
    def __init__(self):
        self.order_blocks: Dict[str, List[OrderBlock]] = {}  # symbol -> list of OBs
        self.liquidity_levels: Dict[str, List[float]] = {}  # symbol -> list of levels
        self.choch_history: Dict[str, List[CHoCH]] = {}  # symbol -> list of CHoCHs
        self.max_order_blocks = 20  # Keep last 20 order blocks per symbol

        # Tuning knobs (higher = fewer, higher-quality signals)
        self.swing_lookback_5m = 60         # candles to analyze for structure
        self.swing_left_right_5m = 2        # pivot width for swing detection
        self.choch_break_buffer_atr = 0.15  # require close beyond swing by this ATR fraction
        self.choch_confirm_closes = 1       # consecutive closes beyond level

        self.ob_lookback_1m = 80            # candles to search for OB
        self.ob_displacement_atr = 1.2      # displacement threshold after OB candle
        self.ob_max_age_minutes = 120       # discard very old OBs
        self.ob_retest_required = True      # require touch + rejection for entry
        
    def detect_liquidity_sweep(self, df: pd.DataFrame, symbol: str) -> Optional[LiquiditySweep]:
        """
        Detect liquidity sweeps (stop hunts)
        Looks for wicks that break previous highs/lows then reverse
        """
        try:
            if len(df) < 20:
                return None
            
            # Get recent candles
            recent = df.tail(20)
            current = recent.iloc[-1]
            prev = recent.iloc[-2]
            
            # Find recent swing highs and lows
            highs = recent['high'].values
            lows = recent['low'].values
            
            # Look for liquidity sweep patterns
            # Bullish sweep: price wicks below recent low then closes above
            if current['low'] < np.min(lows[:-1]) and current['close'] > current['open']:
                # Check if this is a strong reversal
                body_size = abs(current['close'] - current['open'])
                wick_size = current['low'] - min(current['open'], current['close'])
                
                if wick_size > body_size * 0.5:  # Significant wick
                    strength = min(1.0, wick_size / body_size)
                    return LiquiditySweep(
                        price_level=np.min(lows[:-1]),
                        direction='BUY',
                        timestamp=current.name if hasattr(current.name, 'to_pydatetime') else datetime.now(),
                        strength=strength,
                        swept=True
                    )
            
            # Bearish sweep: price wicks above recent high then closes below
            if current['high'] > np.max(highs[:-1]) and current['close'] < current['open']:
                body_size = abs(current['close'] - current['open'])
                wick_size = current['high'] - max(current['open'], current['close'])
                
                if wick_size > body_size * 0.5:
                    strength = min(1.0, wick_size / body_size)
                    return LiquiditySweep(
                        price_level=np.max(highs[:-1]),
                        direction='SELL',
                        timestamp=current.name if hasattr(current.name, 'to_pydatetime') else datetime.now(),
                        strength=strength,
                        swept=True
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting liquidity sweep: {e}")
            return None
    
    def detect_order_block(self, df: pd.DataFrame, symbol: str) -> Optional[OrderBlock]:
        """
        Detect order blocks (institutional order zones)
        Looks for strong candles followed by consolidation
        """
        try:
            if len(df) < 10:
                return None
            
            recent = df.tail(10)
            current = recent.iloc[-1]
            prev = recent.iloc[-2]
            
            # Strong bullish candle (large body, small wicks)
            if prev['close'] > prev['open']:
                body_size = prev['close'] - prev['open']
                upper_wick = prev['high'] - prev['close']
                lower_wick = prev['open'] - prev['low']
                
                # Strong bullish OB: body > 70% of range, small wicks
                if body_size > (prev['high'] - prev['low']) * 0.7:
                    if upper_wick < body_size * 0.3 and lower_wick < body_size * 0.3:
                        # Check if followed by consolidation or pullback
                        if current['close'] <= prev['close']:
                            strength = min(1.0, body_size / (prev['high'] - prev['low']))
                            return OrderBlock(
                                price_high=prev['high'],
                                price_low=prev['low'],
                                is_bullish=True,
                                timestamp=prev.name if hasattr(prev.name, 'to_pydatetime') else datetime.now(),
                                strength=strength
                            )
            
            # Strong bearish candle
            if prev['close'] < prev['open']:
                body_size = prev['open'] - prev['close']
                upper_wick = prev['high'] - prev['open']
                lower_wick = prev['close'] - prev['low']
                
                if body_size > (prev['high'] - prev['low']) * 0.7:
                    if upper_wick < body_size * 0.3 and lower_wick < body_size * 0.3:
                        if current['close'] >= prev['close']:
                            strength = min(1.0, body_size / (prev['high'] - prev['low']))
                            return OrderBlock(
                                price_high=prev['high'],
                                price_low=prev['low'],
                                is_bullish=False,
                                timestamp=prev.name if hasattr(prev.name, 'to_pydatetime') else datetime.now(),
                                strength=strength
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting order block: {e}")
            return None

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range (simple)."""
        if df is None or len(df) < period + 1:
            return 0.0
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values
        tr = []
        for i in range(1, len(df)):
            tr.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
        if not tr:
            return 0.0
        return float(np.mean(tr[-period:]))

    def _find_swings(self, df: pd.DataFrame, left_right: int = 2) -> Tuple[List[int], List[int]]:
        """
        Find swing highs/lows indices using a small fractal window.
        Returns (swing_high_idxs, swing_low_idxs)
        """
        if df is None or len(df) < (left_right * 2 + 3):
            return [], []
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        sh, sl = [], []
        for i in range(left_right, len(df) - left_right):
            h = highs[i]
            l = lows[i]
            if h == np.max(highs[i - left_right : i + left_right + 1]):
                if h > highs[i - 1] and h > highs[i + 1]:
                    sh.append(i)
            if l == np.min(lows[i - left_right : i + left_right + 1]):
                if l < lows[i - 1] and l < lows[i + 1]:
                    sl.append(i)
        return sh, sl

    def _trend_bias_5m(self, df_5min: pd.DataFrame) -> str:
        """Lightweight trend bias filter using MA slope."""
        if df_5min is None or len(df_5min) < 30:
            return "neutral"
        close = df_5min["close"].astype(float)
        ma_fast = close.rolling(10).mean()
        ma_slow = close.rolling(30).mean()
        if ma_fast.iloc[-1] > ma_slow.iloc[-1] and ma_fast.iloc[-1] > ma_fast.iloc[-3]:
            return "bull"
        if ma_fast.iloc[-1] < ma_slow.iloc[-1] and ma_fast.iloc[-1] < ma_fast.iloc[-3]:
            return "bear"
        return "neutral"
    
    def detect_choch(self, df_1min: pd.DataFrame, df_5min: pd.DataFrame, symbol: str) -> Optional[CHoCH]:
        """
        Detect Change of Character (CHoCH)
        Looks for trend changes on 1min and 5min timeframes
        """
        try:
            if len(df_1min) < 50 or len(df_5min) < 20:
                return None

            df_5 = df_5min.tail(self.swing_lookback_5m).copy()
            atr5 = self._atr(df_5, period=14)
            buffer_ = max(0.0, atr5 * float(self.choch_break_buffer_atr))

            swing_highs, swing_lows = self._find_swings(df_5, left_right=self.swing_left_right_5m)
            if not swing_highs or not swing_lows:
                return None

            # Use last meaningful swings (ignore the last few candles)
            last_idx = len(df_5) - 1
            usable_highs = [i for i in swing_highs if i <= last_idx - 2]
            usable_lows = [i for i in swing_lows if i <= last_idx - 2]
            if not usable_highs or not usable_lows:
                return None

            last_swing_high = float(df_5["high"].iloc[usable_highs[-1]])
            last_swing_low = float(df_5["low"].iloc[usable_lows[-1]])
            close_now = float(df_5["close"].iloc[-1])
            closes = df_5["close"].astype(float).values

            # Optional trend bias: avoid taking bullish CHoCH into strong bear bias and vice versa
            bias = self._trend_bias_5m(df_5)

            # Bullish CHoCH: close breaks above last swing high by buffer and holds for N closes
            if close_now > (last_swing_high + buffer_):
                if all(float(c) > (last_swing_high + buffer_) for c in closes[-self.choch_confirm_closes :]):
                    if bias in ("neutral", "bull"):
                        # Confirm on 1m with momentum (higher close vs 10 bars ago)
                        df_1 = df_1min.tail(30)
                        if float(df_1["close"].iloc[-1]) > float(df_1["close"].iloc[-10]):
                            return CHoCH(
                                price_level=last_swing_high,
                                direction="BULLISH",
                                timestamp=datetime.now(),
                                timeframe="5min",
                                confirmed=True,
                            )

            # Bearish CHoCH: close breaks below last swing low by buffer and holds for N closes
            if close_now < (last_swing_low - buffer_):
                if all(float(c) < (last_swing_low - buffer_) for c in closes[-self.choch_confirm_closes :]):
                    if bias in ("neutral", "bear"):
                        df_1 = df_1min.tail(30)
                        if float(df_1["close"].iloc[-1]) < float(df_1["close"].iloc[-10]):
                            return CHoCH(
                                price_level=last_swing_low,
                                direction="BEARISH",
                                timestamp=datetime.now(),
                                timeframe="5min",
                                confirmed=True,
                            )

            return None
            
        except Exception as e:
            logger.error(f"Error detecting CHoCH: {e}")
            return None

    def detect_order_block_v2(self, df: pd.DataFrame, symbol: str) -> Optional[OrderBlock]:
        """
        More selective order block detection:
        - Bullish OB: last bearish candle before strong bullish displacement
        - Bearish OB: last bullish candle before strong bearish displacement
        """
        try:
            if df is None or len(df) < 40:
                return None

            d = df.tail(self.ob_lookback_1m).copy()
            atr = self._atr(d, period=14)
            if atr <= 0:
                return None

            # Look for displacement in last ~10 candles and find the "origin" candle
            # Displacement: big body candle relative to ATR, followed by continuation
            for offset in range(2, 12):
                i = len(d) - 1 - offset
                if i <= 5:
                    break

                c = d.iloc[i]
                body = abs(float(c["close"]) - float(c["open"]))
                rng = float(c["high"]) - float(c["low"])
                if rng <= 0:
                    continue

                # big candle = displacement
                if body < self.ob_displacement_atr * atr:
                    continue

                is_bull_disp = float(c["close"]) > float(c["open"])
                is_bear_disp = float(c["close"]) < float(c["open"])
                if not (is_bull_disp or is_bear_disp):
                    continue

                # Find last opposite candle before displacement: that's the OB candle
                # (keep it close to displacement)
                search_start = max(1, i - 6)
                search_end = i
                ob_idx = None
                if is_bull_disp:
                    for j in range(search_end - 1, search_start - 1, -1):
                        cj = d.iloc[j]
                        if float(cj["close"]) < float(cj["open"]):
                            ob_idx = j
                            break
                else:
                    for j in range(search_end - 1, search_start - 1, -1):
                        cj = d.iloc[j]
                        if float(cj["close"]) > float(cj["open"]):
                            ob_idx = j
                            break
                if ob_idx is None:
                    continue

                ob_c = d.iloc[ob_idx]
                ob_high = float(ob_c["high"])
                ob_low = float(ob_c["low"])
                is_bullish_ob = is_bull_disp

                # Score strength: displacement body vs ATR, plus cleanliness (body/range)
                clean = body / rng
                strength = float(np.clip((body / atr) / 3.0, 0.0, 1.0)) * 0.7 + float(np.clip(clean, 0.0, 1.0)) * 0.3

                return OrderBlock(
                    price_high=ob_high,
                    price_low=ob_low,
                    is_bullish=is_bullish_ob,
                    timestamp=ob_c.name if hasattr(ob_c.name, "to_pydatetime") else datetime.now(),
                    strength=float(np.clip(strength, 0.0, 1.0)),
                )

            return None
        except Exception as e:
            logger.error(f"Error detecting order block v2: {e}")
            return None

    def _is_rejection_candle(self, candle: pd.Series, direction: str) -> bool:
        """Simple rejection candle check at OB zone."""
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        rng = max(1e-12, h - l)
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        if body / rng < 0.25:
            return False  # too doji-ish

        if direction == "BUY":
            return lower_wick / rng > 0.35 and c > o
        return upper_wick / rng > 0.35 and c < o
    
    def check_order_block_entry(self, symbol: str, current_price: float, 
                               direction: str) -> Optional[OrderBlock]:
        """
        Check if price is at an order block for entry
        """
        if symbol not in self.order_blocks:
            return None
        
        now = datetime.now()
        for ob in self.order_blocks[symbol]:
            if ob.broken:
                continue

            # Discard very old blocks
            try:
                age_min = (now - ob.timestamp).total_seconds() / 60.0
                if age_min > self.ob_max_age_minutes:
                    continue
            except Exception:
                pass
            
            # Check if price is in order block zone
            if ob.is_bullish and direction == 'BUY':
                if ob.price_low <= current_price <= ob.price_high:
                    # Check if OB hasn't been tested too many times
                    if not ob.tested or ob.strength > 0.7:
                        return ob
            
            elif not ob.is_bullish and direction == 'SELL':
                if ob.price_low <= current_price <= ob.price_high:
                    if not ob.tested or ob.strength > 0.7:
                        return ob
        
        return None
    
    def update_order_blocks(self, symbol: str, df: pd.DataFrame):
        """Update and maintain order blocks for a symbol"""
        # Prefer the more selective OB detector; fall back to legacy if needed
        ob = self.detect_order_block_v2(df, symbol) or self.detect_order_block(df, symbol)
        
        if ob:
            if symbol not in self.order_blocks:
                self.order_blocks[symbol] = []
            
            self.order_blocks[symbol].append(ob)
            
            # Keep only recent order blocks
            if len(self.order_blocks[symbol]) > self.max_order_blocks:
                self.order_blocks[symbol] = self.order_blocks[symbol][-self.max_order_blocks:]
            
            # Mark old OBs as broken if price has moved away significantly
            current_price = df['close'].iloc[-1]
            for old_ob in self.order_blocks[symbol][:-1]:  # All except the newest
                if old_ob.is_bullish:
                    if current_price < old_ob.price_low * 0.999:  # 0.1% below
                        old_ob.broken = True
                else:
                    if current_price > old_ob.price_high * 1.001:  # 0.1% above
                        old_ob.broken = True
    
    def get_sniper_signal(self, symbol: str, df_1min: pd.DataFrame, 
                         df_5min: pd.DataFrame, macro_bias: Optional[str] = None) -> Dict:
        """
        Get comprehensive sniper entry signal
        Combines liquidity sweeps, order blocks, and CHoCH
        """
        signal = {
            'direction': None,
            'confidence': 0.0,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'reason': '',
            'liquidity_sweep': None,
            'order_block': None,
            'choch': None
        }
        
        try:
            # Precompute simple volatility; used to avoid micro-noise entries
            atr1 = self._atr(df_1min.tail(100), period=14)

            # 1. Check for liquidity sweep
            sweep = self.detect_liquidity_sweep(df_1min, symbol)
            if sweep and sweep.swept:
                signal['liquidity_sweep'] = sweep
                signal['direction'] = sweep.direction
                signal['confidence'] += sweep.strength * 0.35
            
            # 2. Check for CHoCH
            choch = self.detect_choch(df_1min, df_5min, symbol)
            if choch and choch.confirmed:
                signal['choch'] = choch
                if choch.direction == 'BULLISH':
                    if signal['direction'] is None:
                        signal['direction'] = 'BUY'
                    signal['confidence'] += 0.35
                elif choch.direction == 'BEARISH':
                    if signal['direction'] is None:
                        signal['direction'] = 'SELL'
                    signal['confidence'] += 0.35
            
            # 3. Check for order block entry OR just use recent price action
            current_price = df_1min['close'].iloc[-1]
            ob = self.check_order_block_entry(symbol, current_price, signal['direction'])
            if ob:
                signal['order_block'] = ob
                signal['confidence'] += ob.strength * 0.3
                signal['entry_price'] = ob.price_low if ob.is_bullish else ob.price_high
                
                # Calculate SL/TP
                if ob.is_bullish:
                    signal['stop_loss'] = ob.price_low * 0.9995  # 0.05% below OB
                    signal['take_profit'] = ob.price_high * 1.002  # 0.2% above OB
                else:
                    signal['stop_loss'] = ob.price_high * 1.0005  # 0.05% above OB
                    signal['take_profit'] = ob.price_low * 0.998  # 0.2% below OB
            
            # If no direction yet, add a simple moving average crossover check
            if signal['direction'] is None and len(df_1min) >= 20:
                close = df_1min['close'].values
                ma5 = np.mean(close[-5:])
                ma20 = np.mean(close[-20:])
                if ma5 > ma20 and close[-1] > ma5:
                    signal['direction'] = 'BUY'
                    signal['confidence'] += 0.4
                elif ma5 < ma20 and close[-1] < ma5:
                    signal['direction'] = 'SELL'
                    signal['confidence'] += 0.4
            
            # Apply macro (1H/4H) bias as a soft top-down filter — boosts confidence when
            # the lower-timeframe signal agrees with the bigger picture, reduces it when it
            # doesn't. This is soft (not a hard block) so symbols without HTF data yet keep
            # trading exactly as before.
            if macro_bias in ("bullish", "bearish") and signal['direction']:
                signal_dir_bullish = signal['direction'] == 'BUY'
                bias_bullish = macro_bias == "bullish"
                if signal_dir_bullish == bias_bullish:
                    signal['confidence'] = min(1.0, signal['confidence'] * 1.15)
                else:
                    signal['confidence'] = signal['confidence'] * 0.5

            # Build reason string
            reasons = []
            if sweep:
                reasons.append(f"Liquidity sweep ({sweep.direction})")
            if choch:
                reasons.append(f"CHoCH ({choch.direction})")
            if signal.get('order_block'):
                reasons.append("Order block entry")
            if signal['direction'] and not reasons:
                reasons.append("Moving average signal")
            
            signal['reason'] = " | ".join(reasons) if reasons else "No signal"
            
            # Cap confidence at 1.0
            signal['confidence'] = min(1.0, signal['confidence'])
            
            return signal
            
        except Exception as e:
            logger.error(f"Error getting sniper signal: {e}")
            return signal

# Global instance
sniper_logic = SniperLogic()
