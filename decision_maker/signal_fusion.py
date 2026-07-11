"""
Signal Fusion Engine — AERO TRADE
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

from utils.logger import logger
from config.settings import SIGNAL_FUSION_CONFIG


class SignalSource(Enum):
    AI = "ai"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    PRICE_ACTION = "price_action"
    MARKET_STRUCTURE = "market_structure"


@dataclass
class Signal:
    source: SignalSource
    direction: int
    confidence: float
    strength: float
    timestamp: float
    metadata: Dict


@dataclass
class FusedSignal:
    action: str
    strength: float
    confidence: float
    sources: List[str]
    timestamp: float
    details: Dict


class SignalFusion:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.signal_history: List[Signal] = []
        self.max_history_size = 1000
        weights = SIGNAL_FUSION_CONFIG.get("weights", {})
        self.source_weights = {
            SignalSource.AI: weights.get("ai", 0.30),
            SignalSource.TECHNICAL: weights.get("technical", 0.20),
            SignalSource.PRICE_ACTION: weights.get("price_action", 0.15),
            SignalSource.MARKET_STRUCTURE: weights.get("market_structure", 0.10),
            SignalSource.SENTIMENT: weights.get("sentiment", 0.25),
        }
        self.confidence_thresholds = SIGNAL_FUSION_CONFIG.get("min_confidence", {
            "ai": 0.65, "technical": 0.60, "combined": 0.65,
        })
        logger.success("Signal Fusion Engine initialized")

    def _default_config(self) -> Dict:
        return {
            "enable_fusion": True,
            "min_sources": 2,
            "time_window_seconds": 300,
            "decay_factor": 0.95,
            "require_confirmation": True,
        }

    async def add_signal(self, signal: Signal):
        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]

    async def fuse_signals(self, symbol: str = "EURUSD") -> FusedSignal:
        try:
            recent_cutoff = datetime.now().timestamp() - self.config["time_window_seconds"]
            recent_signals = [s for s in self.signal_history if s.timestamp >= recent_cutoff and s.confidence > 0.05]
            if not recent_signals:
                return self._get_neutral_signal("No signals")

            signals_by_source = {}
            for s in recent_signals:
                signals_by_source.setdefault(s.source, []).append(s)

            weighted_votes, total_weight, contributing = [], 0.0, []
            for source, signals in signals_by_source.items():
                if source not in self.source_weights:
                    continue
                avg_dir = np.mean([s.direction * s.confidence for s in signals])
                avg_conf = np.mean([s.confidence for s in signals])
                threshold = self.confidence_thresholds.get(source.value if hasattr(source, "value") else str(source), 0.5)
                if isinstance(source, SignalSource):
                    threshold = self.confidence_thresholds.get(source.value, 0.5)
                if avg_conf >= threshold * 0.8:
                    weight = self.source_weights[source] * avg_conf
                    weighted_votes.append(avg_dir * weight)
                    total_weight += weight
                    contributing.append(source.value if isinstance(source, SignalSource) else str(source))

            if total_weight == 0 or len(contributing) < self.config["min_sources"]:
                return self._get_neutral_signal(f"Insufficient sources: {len(contributing)}")

            final_strength = sum(weighted_votes) / total_weight
            final_confidence = total_weight / sum(self.source_weights.values())

            if final_confidence < self.confidence_thresholds.get("combined", 0.65):
                return self._get_neutral_signal(f"Low confidence: {final_confidence:.2f}")

            action = "HOLD" if abs(final_strength) < 0.15 else ("BUY" if final_strength > 0 else "SELL")
            return FusedSignal(
                action=action, strength=float(final_strength), confidence=float(final_confidence),
                sources=contributing, timestamp=datetime.now().timestamp(),
                details={"symbol": symbol, "signal_count": len(recent_signals)},
            )
        except Exception as e:
            logger.error(f"Fusion error: {e}")
            return self._get_neutral_signal(str(e))

    def _get_neutral_signal(self, reason: str) -> FusedSignal:
        return FusedSignal("HOLD", 0.0, 0.0, [], datetime.now().timestamp(), {"reason": reason})

    async def calculate_position_size(self, fused_signal: FusedSignal, account_balance: float, risk_per_trade: float = 0.02) -> float:
        if fused_signal.action == "HOLD":
            return 0.0
        base = account_balance * risk_per_trade
        return max(base * fused_signal.confidence * min(1.0, abs(fused_signal.strength) * 2) * 0.001, 0.01)

    async def get_signal_statistics(self) -> Dict:
        return {"total_signals": len(self.signal_history)}


signal_fusion = SignalFusion()
