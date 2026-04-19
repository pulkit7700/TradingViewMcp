"""
Swarm Intelligence Trading — Signal Aggregation
================================================
Ensemble signal aggregation with PSO-optimized weights.
Entry/exit decision logic using R-A model zones (Couzin 2002).
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from .swarm_base import AgentSignal, SwarmState


@dataclass
class AggregatedSignal:
    """Final trading signal from ensemble of all agents."""
    ticker: str
    action: str               # 'BUY', 'SELL', 'HOLD', 'WATCH'
    direction: float          # -1 to 1
    confidence: float         # 0 to 1
    strength: float           # 0 to 1
    zone: str                 # 'ZOR', 'ZOO', 'ZOA' (R-A model)
    component_signals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    # Sector-first scan fields
    sector: Optional[str] = None
    sector_rank: Optional[int] = None        # 1 = top sector

    # Concrete trade plan (populated by scan_sectors, not run_tick)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    estimated_hold_days: Optional[int] = None
    atr: Optional[float] = None
    entry_type: Optional[str] = None        # 'support' | 'resistance' | 'breakout'
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


class SignalAggregator:
    """
    Ensemble aggregation: weighted vote from all swarm agents.
    Default weights: 30% ACO + 20% Vicsek + 15% Boids + 20% Leader + 15% Topological
    Weights can be PSO-optimized.
    """

    DEFAULT_WEIGHTS = {
        'aco': 0.30,
        'vicsek': 0.20,
        'boids': 0.15,
        'leader': 0.20,
        'topological': 0.15,
    }

    def __init__(self, config: dict):
        self.config = config
        self.base_weights = dict(config.get('ensemble_weights', self.DEFAULT_WEIGHTS.copy()))
        self.weights = self.base_weights.copy()
        self.entry_threshold = config.get('entry_threshold', 0.45)
        self.exit_threshold = config.get('exit_threshold', 0.35)
        self.signal_history: List[AggregatedSignal] = []

        # R-A Model zone thresholds (Couzin 2002)
        self.zor_pct = config.get('zor_pct', 0.02)   # 2% loss = repulsion
        self.zoo_pct = config.get('zoo_pct', 0.05)    # 2-5% = orientation
        self.zoa_pct = config.get('zoa_pct', 0.15)    # 5-15% = attraction

    def aggregate(self, signals: Dict[str, AgentSignal], ticker: str,
                  current_price: float = 0.0, entry_price: float = 0.0) -> AggregatedSignal:
        """Aggregate all agent signals into a single trading decision."""

        if not signals:
            return AggregatedSignal(ticker=ticker, action='HOLD', direction=0.0,
                                   confidence=0.0, strength=0.0, zone='ZOO')

        # Weighted ensemble
        total_direction = 0.0
        total_weight = 0.0
        component_signals = {}

        for agent_name, weight in self.weights.items():
            sig = signals.get(agent_name)
            if sig is not None and hasattr(sig, 'direction'):
                effective_weight = weight * sig.confidence
                total_direction += sig.direction * effective_weight
                total_weight += effective_weight
                component_signals[agent_name] = {
                    'direction': sig.direction,
                    'confidence': sig.confidence,
                    'signal_type': sig.signal_type,
                }

        if total_weight == 0:
            return AggregatedSignal(ticker=ticker, action='HOLD', direction=0.0,
                                   confidence=0.0, strength=0.0, zone='ZOO',
                                   component_signals=component_signals)

        final_direction = total_direction / total_weight
        strength = abs(final_direction)
        confidence = min(total_weight / sum(self.weights.values()), 1.0)

        # R-A Zone determination
        zone = self._determine_zone(current_price, entry_price)

        # Action determination
        action = self._determine_action(final_direction, strength, confidence, zone)

        agg = AggregatedSignal(
            ticker=ticker,
            action=action,
            direction=float(final_direction),
            confidence=float(confidence),
            strength=float(strength),
            zone=zone,
            component_signals=component_signals,
            metadata={
                'total_weight': float(total_weight),
                'n_active_agents': len(component_signals),
                'entry_threshold': self.entry_threshold,
            }
        )
        self.signal_history.append(agg)
        if len(self.signal_history) > 500:
            self.signal_history = self.signal_history[-250:]
        return agg

    def _determine_zone(self, current_price: float, entry_price: float) -> str:
        """R-A Model zone assignment (Couzin 2002)."""
        if entry_price <= 0 or current_price <= 0:
            return 'ZOA'  # No position: in attraction zone (looking to enter)

        pct_change = (current_price - entry_price) / entry_price

        if pct_change <= -self.zor_pct:
            return 'ZOR'  # Repulsion: stop-loss territory
        elif abs(pct_change) <= self.zoo_pct:
            return 'ZOO'  # Orientation: monitoring
        else:
            return 'ZOA'  # Attraction: profit zone or entry zone

    def _determine_action(self, direction: float, strength: float,
                          confidence: float, zone: str) -> str:
        """Determine trading action from signal + zone."""

        # ZOR overrides everything: get out
        if zone == 'ZOR':
            return 'SELL'

        # Strong signal + high confidence = entry
        if strength > self.entry_threshold and confidence > 0.4:
            if direction > 0:
                return 'BUY'
            elif direction < 0:
                return 'SELL'

        # Moderate signal = watch
        if strength > self.exit_threshold:
            return 'WATCH'

        return 'HOLD'

    def update_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights (typically from PSO optimization)."""
        total = sum(new_weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in new_weights.items()}


class ConfidenceScorer:
    """
    Dynamic confidence scoring based on agent historical accuracy.
    Adjusts ensemble weights based on which agents have been most accurate recently.
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def compute_dynamic_weights(self, agents: dict,
                                 base_weights: Dict[str, float]) -> Dict[str, float]:
        """Reweight agents by their historical accuracy."""
        dynamic_weights = {}
        for name, weight in base_weights.items():
            agent = agents.get(name)
            if agent is None:
                dynamic_weights[name] = weight
                continue

            if isinstance(agent, list):
                # ACO: average accuracy across ants
                accuracies = [a.get_accuracy(self.lookback) for a in agent]
                accuracy = np.mean(accuracies) if accuracies else 0.5
            else:
                accuracy = agent.get_accuracy(self.lookback)

            # Scale weight by accuracy (but don't let it go to zero)
            dynamic_weights[name] = weight * (0.5 + accuracy)

        # Normalize
        total = sum(dynamic_weights.values())
        if total > 0:
            dynamic_weights = {k: v / total for k, v in dynamic_weights.items()}
        return dynamic_weights
