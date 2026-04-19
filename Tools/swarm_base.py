"""
Swarm Intelligence Trading — Base Classes
==========================================
Core dataclasses and abstract base for all swarm agents.
Based on Wang et al. (2024) SI survey.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class AgentSignal:
    """Signal emitted by any swarm agent."""
    ticker: str
    signal_type: str          # 'entry', 'exit', 'hold', 'watch'
    direction: float          # -1.0 (strong short) to 1.0 (strong long)
    confidence: float         # 0.0 to 1.0
    strength: float           # magnitude of signal
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None

    def __post_init__(self):
        self.direction = float(np.clip(self.direction, -1.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.strength = float(np.clip(self.strength, 0.0, 1.0))
        if self.timestamp is None:
            self.timestamp = pd.Timestamp.now()


class BaseAgent(ABC):
    """Abstract base for all swarm agents."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.signal_history: List[AgentSignal] = []

    @abstractmethod
    def compute(self, market_data: dict, swarm_state: 'SwarmState') -> AgentSignal:
        """Compute signal from market data and shared swarm state."""
        pass

    def update_history(self, signal: AgentSignal):
        self.signal_history.append(signal)
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-500:]

    def get_accuracy(self, lookback: int = 50) -> float:
        recent = self.signal_history[-lookback:]
        if not recent:
            return 0.5
        correct = sum(1 for s in recent if s.metadata.get('was_correct', False))
        return correct / len(recent)


@dataclass
class SwarmState:
    """Shared memory for all swarm agents — like the environment ants deposit pheromone on."""
    tickers: List[str]
    n_assets: int

    # ACO: pheromone matrix [n_assets x n_assets]
    pheromone_matrix: np.ndarray = field(default_factory=lambda: np.ones((1, 1)))

    # Vicsek: consensus direction per asset [-pi, pi]
    consensus_direction: np.ndarray = field(default_factory=lambda: np.zeros(1))

    # R-A: zone assignment per asset {'TICKER': 'ZOR'|'ZOO'|'ZOA'}
    zone_assignments: Dict[str, str] = field(default_factory=dict)

    # PSO state
    particle_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    particle_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    pbest_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    pbest_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    gbest_position: np.ndarray = field(default_factory=lambda: np.array([]))
    gbest_score: float = -np.inf

    # Leadership scores per asset
    leader_scores: Dict[str, float] = field(default_factory=dict)

    # Topological neighbors (Ballerini: 7 nearest)
    topological_neighbors: Dict[str, List[str]] = field(default_factory=dict)

    # Aggregated signals from all agents
    agent_signals: Dict[str, AgentSignal] = field(default_factory=dict)

    # Correlation matrix cache
    correlation_matrix: Optional[pd.DataFrame] = None

    # Feature cache per ticker
    features_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def initialize(self):
        """Initialize all arrays to proper dimensions."""
        n = self.n_assets
        self.pheromone_matrix = np.ones((n, n)) * 0.1
        self.consensus_direction = np.zeros(n)
        self.zone_assignments = {t: 'ZOO' for t in self.tickers}
        # O(1) ticker lookup
        self._ticker_index = {t: i for i, t in enumerate(self.tickers)}

    def deposit_pheromone(self, i: int, j: int, quality: float):
        """Deposit pheromone without evaporation (evaporate once per tick instead)."""
        self.pheromone_matrix[i, j] += quality
        self.pheromone_matrix[j, i] += quality  # symmetric

    def update_pheromone(self, i: int, j: int, quality: float, rho: float = 0.1):
        """Legacy: ACO pheromone update with evaporation (kept for compatibility)."""
        self.pheromone_matrix[i, j] += quality
        self.pheromone_matrix[j, i] += quality

    def evaporate_pheromone(self, rho: float = 0.1):
        """Global pheromone evaporation — call once per tick, not per ant."""
        self.pheromone_matrix *= (1 - rho)

    def get_ticker_index(self, ticker: str) -> int:
        return self._ticker_index.get(ticker, -1)

    def get_pheromone_strength(self, ticker: str) -> float:
        idx = self.get_ticker_index(ticker)
        if idx < 0:
            return 0.0
        return float(self.pheromone_matrix[idx].sum())

    def expand_tickers(self, new_tickers: List[str]):
        """Expand pheromone matrix and index to include new tickers (e.g. from full sector scan)."""
        if not hasattr(self, '_ticker_index'):
            self._ticker_index = {t: i for i, t in enumerate(self.tickers)}
        existing = set(self.tickers)
        to_add = [t for t in new_tickers if t not in existing]
        if not to_add:
            return
        n_old = self.n_assets
        n_new = n_old + len(to_add)
        # Grow pheromone matrix, preserving existing values
        new_pm = np.ones((n_new, n_new)) * 0.1
        new_pm[:n_old, :n_old] = self.pheromone_matrix
        self.pheromone_matrix = new_pm
        # Grow consensus_direction
        new_dir = np.zeros(n_new)
        new_dir[:n_old] = self.consensus_direction
        self.consensus_direction = new_dir
        # Extend ticker list and index
        for t in to_add:
            self._ticker_index[t] = len(self.tickers)
            self.tickers = list(self.tickers) + [t]
            self.zone_assignments[t] = 'ZOO'
        self.n_assets = n_new
