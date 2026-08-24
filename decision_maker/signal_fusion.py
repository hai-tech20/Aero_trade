"""
Signal Fusion Engine — AERO TRADE MASTERY EDITION
===================================================
  - STRICT: minimum 3 contributing sources (never 2!)
  - Weight redistribution: market_structure 0.08 moved 50/50 to AI + price_action
  - HARD checks: AI above floor, macro-bias alignment, PA confirmation, combined floor 0.70
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

from utils.logger import logger
from config.settings import SIGNAL_FUSION_CONFIG, AI_CONFIG


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
        raw_weights = SIGNAL_FUSION_CONFIG.get("weights", {})
        # Redistribution: since MARKET_STRUCTURE is never emitted, fold its 0.08 into
        # AI (+0.04) and PRICE_ACTION (+0.04) so they sum to 1.00.
        ai_w = float(raw_weights.get("ai", 0.38))
        pa_w = float(raw_weights.get("price_action", 0.30))
        ms_w = float(raw_weights.get("market_structure", 0.08))
        tech_w = float(raw_weights.get("technical", 0.14))
        sent_w = float(raw_weights.get("sentiment", 0.10))
        ai_w += ms_w * 0.5
        pa_w += ms_w * 0.5
        total = ai_w + pa_w + tech_w + sent_w + ms_w
        ai_w /= total; pa_w /= total; tech_w /= total; sent_w /= total; ms_w /= total
        self.source_weights = {
            SignalSource.AI: ai_w,
            SignalSource.TECHNICAL: tech_w,
            SignalSource.PRICE_ACTION: pa_w,
            SignalSource.MARKET_STRUCTURE: ms_w,
            SignalSource.SENTIMENT: sent_w,
        }
        self.confidence_thresholds = SIGNAL_FUSION_CONFIG.get("min_confidence", {
            "ai": 0.55,
            "technical": 0.40,
            "price_action": 0.45,
            "sentiment": 0.20,
            "combined": 0.65,
        })
        # Strict thresholds from config:
        self.min_sources_strict = int(SIGNAL_FUSION_CONFIG.get("min_sources", 3))
        self.require_ai = bool(SIGNAL_FUSION_CONFIG.get("require_ai_above_floor", True))
        self.require_pa = bool(SIGNAL_FUSION_CONFIG.get("require_price_action_confirmation", True))
        self.require_macro = bool(SIGNAL_FUSION_CONFIG.get("require_macro_bias_alignment", True))
        self.require_conf = bool(SIGNAL_FUSION_CONFIG.get("require_confirmation", True))
        self.final_confidence_floor = float(SIGNAL_FUSION_CONFIG.get("final_combined_confidence_floor", 0.70))
        self.ai_min_conf = float(self.confidence_thresholds.get("ai", AI_CONFIG.get("min_confidence", 0.55)))
        logger.success("Signal Fusion Engine initialized (Mastery strict mode)")

    def _default_config(self) -> Dict:
        return {
            "enable_fusion": True,
            "min_sources": 3,
            "time_window_seconds": 300,
            "decay_factor": 0.95,
            "require_confirmation": True,
        }

    async def add_signal(self, signal: Signal):
        self.signal_history.append(signal)
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]

    async def fuse_signals(self, symbol: str = "EURUSD",
                            macro_bias: Optional[str] = None,
                            confluence_score: float = 0.0) -> FusedSignal:
        """
        Strict multi-source fusion.

        Parameters
        ----------
        macro_bias : Optional[str] — "bullish" / "bearish" / None (neutral).
            If provided and `require_macro` is True, counter-HTF trades are hard blocked
            — but only when `confluence_score` shows real multi-timeframe agreement
            behind that bias (>=0.55). A bias from one barely-trending timeframe
            shouldn't outright veto a trade the same way 3 aligned timeframes should;
            weak confluence instead gets a stricter confidence floor upstream
            (aero_trade_bot._process_symbol) rather than a hard block here.
        confluence_score : float — 0..1, from core.analysis.macro_bias.get_macro_bias.
        """
        try:
            recent_cutoff = datetime.now().timestamp() - self.config["time_window_seconds"]
            recent_signals = [s for s in self.signal_history if s.timestamp >= recent_cutoff and s.confidence > 0.05]
            if not recent_signals:
                return self._get_neutral_signal("No signals")

            signals_by_source: Dict[SignalSource, List[Signal]] = {}
            for s in recent_signals:
                signals_by_source.setdefault(s.source, []).append(s)

            # Build per-source aggregations
            per_source: Dict[str, Dict[str, float]] = {}
            weighted_votes: List[float] = []
            total_weight: float = 0.0
            contributing: List[str] = []
            ai_direction: Optional[int] = None
            ai_conf: float = 0.0
            pa_direction: Optional[int] = None
            pa_conf: float = 0.0

            for source, signals in signals_by_source.items():
                if source not in self.source_weights:
                    continue
                if not signals:
                    continue
                avg_dir = float(np.mean([s.direction * s.confidence for s in signals]))
                avg_conf = float(np.mean([s.confidence for s in signals]))
                source_name = source.value if isinstance(source, SignalSource) else str(source)
                per_source[source_name] = {"dir": avg_dir, "conf": avg_conf, "raw_dir": int(np.sign(avg_dir) if abs(avg_dir) > 1e-9 else 0)}

                threshold = float(self.confidence_thresholds.get(source_name, 0.30))
                if avg_conf < threshold * 0.8:
                    continue

                weight = self.source_weights[source] * avg_conf
                weighted_votes.append(avg_dir * weight)
                total_weight += weight
                contributing.append(source_name)

                if source == SignalSource.AI:
                    ai_conf = avg_conf
                    ai_direction = 1 if avg_dir > 0 else (-1 if avg_dir < 0 else 0)
                if source == SignalSource.PRICE_ACTION:
                    pa_conf = avg_conf
                    pa_direction = 1 if avg_dir > 0 else (-1 if avg_dir < 0 else 0)

            # ====================================================================
            # HARD BLOCK 1 — not enough contributing sources (require >= 3)
            # ====================================================================
            if total_weight <= 0 or len(contributing) < self.min_sources_strict:
                return self._get_neutral_signal(
                    f"Insufficient sources: {len(contributing)}/{self.min_sources_strict} "
                    f"({','.join(contributing) or 'none'})"
                )

            # ====================================================================
            # HARD BLOCK 2 — AI must be present + above its floor
            # ====================================================================
            if self.require_ai:
                if ai_direction is None or ai_direction == 0:
                    return self._get_neutral_signal("AI signal required but absent or flat")
                if ai_conf < self.ai_min_conf:
                    return self._get_neutral_signal(
                        f"AI confidence {ai_conf:.2f} below floor {self.ai_min_conf:.2f}"
                    )

            # ====================================================================
            # HARD BLOCK 3 — Price-Action confirmation must agree with AI direction
            # ====================================================================
            if self.require_pa:
                if pa_direction is None or pa_direction == 0:
                    return self._get_neutral_signal("Price-action confirmation required but absent")
                if ai_direction is not None and pa_direction != ai_direction:
                    return self._get_neutral_signal(
                        f"AI (dir={ai_direction}) vs Price-Action (dir={pa_direction}) DISAGREE — blocked"
                    )

            final_strength = float(sum(weighted_votes) / total_weight) if total_weight > 0 else 0.0
            pre_confidence = float(total_weight / sum(self.source_weights.values()))
            # Final combined confidence: weighted avg + AI confidence bonus
            final_confidence = float(min(1.0, 0.6 * pre_confidence + 0.4 * ai_conf))

            # ====================================================================
            # HARD BLOCK 4 — Final combined confidence >= 0.70
            # ====================================================================
            if final_confidence < self.final_confidence_floor:
                return self._get_neutral_signal(
                    f"Combined confidence {final_confidence:.2f} below floor {self.final_confidence_floor:.2f}"
                )

            # ====================================================================
            # Direction call
            # ====================================================================
            if abs(final_strength) < 0.12:
                return self._get_neutral_signal("Direction too flat / no conviction")
            action = "BUY" if final_strength > 0 else "SELL"
            proposed_dir = 1 if action == "BUY" else -1

            # ====================================================================
            # HARD BLOCK 5 — Macro / HTF bias alignment (if provided)
            # ====================================================================
            if self.require_macro and macro_bias in ("bullish", "bearish"):
                bias_bull = macro_bias == "bullish"
                signal_bull = proposed_dir == 1
                if bias_bull != signal_bull and confluence_score >= 0.55:
                    return self._get_neutral_signal(
                        f"Counter-HTF trade blocked: signal={action} vs macro={macro_bias} "
                        f"(confluence={confluence_score:.2f})"
                    )

            # ====================================================================
            # HARD BLOCK 6 — Technical + Sentiment must NOT strongly disagree
            # ====================================================================
            tech = per_source.get("technical", {"dir": 0.0, "conf": 0.0})
            if tech["conf"] >= 0.45:
                tech_dir = int(np.sign(tech["dir"])) if abs(tech["dir"]) > 1e-9 else 0
                if tech_dir != 0 and tech_dir != proposed_dir:
                    return self._get_neutral_signal(
                        f"Technical filter disagrees (tech dir={tech_dir} vs action {action})"
                    )

            details = {
                "symbol": symbol,
                "signal_count": len(recent_signals),
                "ai_confidence": round(ai_conf, 4),
                "pa_confidence": round(pa_conf, 4),
                "per_source": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
                               for k, v in per_source.items()},
                "macro_bias": str(macro_bias or "neutral"),
                "confluence_score": round(confluence_score, 4),
                "weighted_strength": round(final_strength, 4),
            }

            return FusedSignal(
                action=action, strength=final_strength, confidence=final_confidence,
                sources=contributing, timestamp=datetime.now().timestamp(),
                details=details,
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
        # Size scales with confidence; very conservative minimum lot floor
        size_raw = base * fused_signal.confidence * min(1.0, abs(fused_signal.strength) * 2) * 0.001
        return max(size_raw, 0.01)

    async def get_signal_statistics(self) -> Dict:
        return {"total_signals": len(self.signal_history)}


signal_fusion = SignalFusion()
