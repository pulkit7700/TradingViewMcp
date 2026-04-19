"""
Swarm Intelligence Trading — Orchestrator & Backtest Engine
===========================================================
AgentOrchestrator: spawns and coordinates all swarm agents.
BacktestEngine: walk-forward optimization and signal evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

from .swarm_base import BaseAgent, AgentSignal, SwarmState
from .swarm_agents import (
    BoidsMomentumAgent, VicsekConsensusAgent, ACOPathAgent,
    PSOOptimizerAgent, LeaderFollowerAgent, TopologicalAgent,
)
from .swarm_aggregator import SignalAggregator, AggregatedSignal, ConfidenceScorer
from .swarm_market_data import (
    fetch_price_data, fetch_multi_ticker, fetch_close_matrix,
    compute_swarm_features, compute_correlation_matrix, get_topological_neighbors,
    get_vix_level, compute_noise_from_vix, get_options_flow_signals,
    fetch_all_features_parallel, fetch_features_in_batches,
    compute_trade_levels, compute_support_resistance, compute_sector_score,
    SECTOR_ETFS, SECTOR_STOCKS, MARKET_LEADERS, cached_fetch,
)

logger = logging.getLogger(__name__)


@dataclass
class SectorScanResult:
    """Output of AgentOrchestrator.scan_sectors()."""
    sector_scores: Dict[str, float]                  # sector_name -> score [0,1]
    sector_ranking: List[str]                        # best to worst
    top_sectors: List[str]                           # selected top N
    sector_signals: Dict[str, 'AggregatedSignal']    # ETF ticker -> signal
    stock_signals: Dict[str, 'AggregatedSignal']     # stock ticker -> signal + trade plan
    scan_timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from backtesting the swarm system."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    trades: List[dict] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    signals_log: List[dict] = field(default_factory=list)


class AgentOrchestrator:
    """
    Spawns and coordinates all swarm agents.
    Manages the shared SwarmState and runs tick-by-tick simulation.
    """

    def __init__(self, tickers: List[str], config: dict = None):
        self.tickers = tickers
        self.config = config or self._default_config()
        self.agents: Dict = {}
        self.swarm_state = SwarmState(tickers=tickers, n_assets=len(tickers))
        self.swarm_state.initialize()
        self.aggregator = SignalAggregator(self.config)
        self.confidence_scorer = ConfidenceScorer()
        self._spawned = False
        # Cached VIX + options flow (fetched once per run, not per ticker)
        self._cached_vix: Optional[float] = None
        self._cached_vix_noise: Optional[float] = None
        self._cached_options_flow: Dict[str, dict] = {}
        self._run_timing: Dict[str, float] = {}  # Performance tracking

    @staticmethod
    def _default_config() -> dict:
        return {
            # Boids
            'boids_c_sep': 1.5, 'boids_c_align': 1.0, 'boids_c_coh': 0.5,
            'boids_sep_radius': 0.02,
            # Vicsek
            'vicsek_corr_threshold': 0.4, 'vicsek_noise_scaling': 1.0,
            # ACO
            'aco_alpha': 1.0, 'aco_beta': 2.0, 'aco_rho': 0.1, 'aco_q': 100.0,
            # PSO
            'pso_w_max': 0.9, 'pso_w_min': 0.4, 'pso_c1': 2.0, 'pso_c2': 2.0,
            'pso_max_iter': 100,
            # Leader
            'leader_vol_z_threshold': 2.0, 'leader_weight': 0.6,
            'democratic_weight': 0.4,
            # Topological
            'topo_n_neighbors': 7, 'topo_systemic_ticker': '^NSEI',
            # Ensemble
            'ensemble_weights': {
                'aco': 0.30, 'vicsek': 0.20, 'boids': 0.15,
                'leader': 0.20, 'topological': 0.15,
            },
            'entry_threshold': 0.45, 'exit_threshold': 0.35,
            # R-A zones
            'zor_pct': 0.02, 'zoo_pct': 0.05, 'zoa_pct': 0.15,
            # ACO ants and PSO particles
            'n_aco_ants': 20, 'n_pso_particles': 30,
        }

    def spawn_agents(self) -> 'AgentOrchestrator':
        """Spawn all swarm agent types."""
        n_aco = self.config.get('n_aco_ants', 20)
        n_pso = self.config.get('n_pso_particles', 30)

        self.agents['boids'] = BoidsMomentumAgent('boids', self.config)
        self.agents['vicsek'] = VicsekConsensusAgent('vicsek', self.config)
        self.agents['leader'] = LeaderFollowerAgent('leader', self.config)
        self.agents['topological'] = TopologicalAgent('topological', self.config)

        # Multiple ACO ants
        self.agents['aco_ants'] = [
            ACOPathAgent(f'aco_ant_{i}', self.config, ant_id=i) for i in range(n_aco)
        ]

        # PSO optimizer
        self.agents['pso'] = PSOOptimizerAgent('pso', self.config, n_particles=n_pso)

        self._spawned = True
        logger.info(f"Spawned agents: Boids, Vicsek, {n_aco} ACO ants, PSO({n_pso}), Leader, Topological")
        return self

    def initialize_market_data(self, period: str = "2y"):
        """Pre-fetch and cache all market data."""
        logger.info(f"Fetching market data for {len(self.tickers)} tickers...")

        # Fetch correlation matrix
        all_tickers = list(set(self.tickers + list(SECTOR_ETFS.values()) + MARKET_LEADERS))
        corr = compute_correlation_matrix(all_tickers, period)
        self.swarm_state.correlation_matrix = corr

        # Compute topological neighbors
        if not corr.empty:
            self.swarm_state.topological_neighbors = get_topological_neighbors(
                corr, n_neighbors=self.config.get('topo_n_neighbors', 7)
            )

        # Fetch and cache features for all tickers in parallel
        parallel_results = fetch_all_features_parallel(all_tickers, period=period, max_workers=12)
        self.swarm_state.features_cache.update(parallel_results)
        logger.info(f"Cached features for {len(self.swarm_state.features_cache)} tickers (parallel fetch)")

    def run_tick(self, ticker: str, market_data: dict = None) -> AggregatedSignal:
        """Run all agents for one ticker at one timestep."""
        if not self._spawned:
            self.spawn_agents()

        if market_data is None:
            market_data = self._build_market_data(ticker)

        signals = {}

        # Run base agents
        for agent_name in ['boids', 'vicsek', 'leader', 'topological']:
            agent = self.agents.get(agent_name)
            if agent:
                try:
                    signals[agent_name] = agent.compute(market_data, self.swarm_state)
                except Exception as e:
                    logger.warning(f"Agent {agent_name} error: {e}")

        # Run ACO ants and aggregate
        aco_ants = self.agents.get('aco_ants', [])
        if aco_ants:
            ant_signals = []
            for ant in aco_ants:
                try:
                    ant_signals.append(ant.compute(market_data, self.swarm_state))
                except Exception as e:
                    logger.warning(f"ACO ant error: {e}")
            if ant_signals:
                signals['aco'] = self._aggregate_ant_signals(ant_signals, ticker)

        # Run PSO
        pso = self.agents.get('pso')
        if pso:
            try:
                pso_signal = pso.compute(market_data, self.swarm_state)
                # PSO doesn't vote directly; it optimizes parameters for other agents
            except Exception as e:
                logger.warning(f"PSO error: {e}")

        # Dynamic weight adjustment based on accuracy
        dynamic_weights = self.confidence_scorer.compute_dynamic_weights(
            self.agents, self.aggregator.base_weights
        )
        self.aggregator.update_weights(dynamic_weights)

        # Aggregate signals
        current_price = 0.0
        features = market_data.get('features')
        if features is not None and not features.empty and 'Close' in features.columns:
            current_price = float(features['Close'].iloc[-1])

        return self.aggregator.aggregate(signals, ticker, current_price=current_price)

    def run_all_tickers(self) -> Dict[str, AggregatedSignal]:
        """Run all agents for all tickers with performance tracking."""
        t0 = time.time()

        # Cache VIX once for the entire run
        self._cached_vix = get_vix_level()
        self._cached_vix_noise = compute_noise_from_vix(self._cached_vix)
        self._run_timing['vix_fetch'] = time.time() - t0

        # Batch-fetch options flow for all tickers in parallel
        t1 = time.time()
        self._cached_options_flow = self._fetch_options_flow_batch(self.tickers)
        self._run_timing['options_flow_fetch'] = time.time() - t1

        # Evaporate pheromone ONCE per tick (not per ant)
        rho = self.config.get('aco_rho', 0.1)
        self.swarm_state.evaporate_pheromone(rho / len(self.swarm_state.tickers))

        # Run agents for all tickers
        t2 = time.time()
        results = {}
        for ticker in self.tickers:
            results[ticker] = self.run_tick(ticker)
        self._run_timing['agent_execution'] = time.time() - t2
        self._run_timing['total'] = time.time() - t0

        logger.info(f"Swarm run complete: {len(results)} tickers in {self._run_timing['total']:.1f}s "
                     f"(VIX: {self._run_timing['vix_fetch']:.1f}s, "
                     f"Options: {self._run_timing['options_flow_fetch']:.1f}s, "
                     f"Agents: {self._run_timing['agent_execution']:.1f}s)")
        return results

    def _fetch_options_flow_batch(self, tickers: List[str]) -> Dict[str, dict]:
        """Fetch options flow for all tickers in parallel."""
        results = {}
        def _fetch_one(t):
            try:
                return t, get_options_flow_signals(t)
            except Exception:
                return t, None
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_fetch_one, t) for t in tickers]
            for f in as_completed(futures):
                t, flow = f.result()
                if flow is not None:
                    results[t] = flow
        return results

    def _build_market_data(self, ticker: str) -> dict:
        """Build market data dict using cached VIX and options flow."""
        features = self.swarm_state.features_cache.get(ticker, pd.DataFrame())
        return {
            'ticker': ticker,
            'features': features,
            'vix_level': self._cached_vix or 20.0,
            'vix_noise': self._cached_vix_noise or 0.2,
            'options_flow': self._cached_options_flow.get(ticker),
        }

    def _aggregate_ant_signals(self, ant_signals: List[AgentSignal],
                                ticker: str) -> AgentSignal:
        """Aggregate multiple ACO ant signals by confidence weighting."""
        if not ant_signals:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)
        directions = [s.direction for s in ant_signals]
        confidences = [s.confidence for s in ant_signals]
        total_conf = sum(confidences) + 1e-9
        avg_direction = sum(d * c for d, c in zip(directions, confidences)) / total_conf
        avg_confidence = float(np.mean(confidences))
        return AgentSignal(
            ticker=ticker, signal_type='entry' if abs(avg_direction) > 0.5 else 'hold',
            direction=float(avg_direction), confidence=avg_confidence,
            strength=float(abs(avg_direction)),
            metadata={'n_ants': len(ant_signals),
                     'ant_agreement': float(1.0 - np.std(directions) / (abs(np.mean(directions)) + 1e-9))}
        )

    def get_swarm_state_summary(self) -> dict:
        """Get a summary of the current swarm state for visualization."""
        ss = self.swarm_state
        return {
            'n_tickers': ss.n_assets,
            'pheromone_mean': float(ss.pheromone_matrix.mean()),
            'pheromone_max': float(ss.pheromone_matrix.max()),
            'zones': dict(ss.zone_assignments),
            'leaders': dict(ss.leader_scores),
            'n_cached_features': len(ss.features_cache),
            'consensus_directions': {
                t: float(ss.consensus_direction[i])
                for i, t in enumerate(ss.tickers)
                if i < len(ss.consensus_direction)
            },
        }

    def scan_sectors(
        self,
        top_n: int = 3,
        stocks_per_sector: int = 6,
        period: str = "2y",
        rr_min: float = 1.5,
        atr_multiplier: float = 1.5,
    ) -> SectorScanResult:
        """
        Two-phase sector-first scan.

        Phase 1: Score all 11 sector ETFs with swarm agents + pheromone strength.
                 Select top_n sectors.
        Phase 2: Run swarm agents on liquid stocks within top sectors.
                 Attach entry/exit/stop/RR trade plan to every actionable signal.
        """
        t_total = time.time()

        # ── Phase 1: Score sectors ─────────────────────────────────────────
        t0 = time.time()
        etf_tickers = list(SECTOR_ETFS.values())
        all_init = etf_tickers + MARKET_LEADERS

        # Ensure ETF data is cached
        missing = [t for t in all_init if t not in self.swarm_state.features_cache]
        if missing:
            parallel_results = fetch_all_features_parallel(missing, period=period, max_workers=8)
            self.swarm_state.features_cache.update(parallel_results)

        # Cache VIX once for this scan
        if self._cached_vix is None:
            self._cached_vix = get_vix_level()
            self._cached_vix_noise = compute_noise_from_vix(self._cached_vix)

        # Run agents on each sector ETF
        sector_signals: Dict[str, AggregatedSignal] = {}
        for etf in etf_tickers:
            try:
                market_data = self._build_market_data(etf)
                sector_signals[etf] = self.run_tick(etf, market_data)
            except Exception as e:
                logger.warning(f"Sector ETF {etf} failed: {e}")

        # Score each sector
        etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
        sector_scores: Dict[str, float] = {}
        for etf, sig in sector_signals.items():
            sector_name = etf_to_sector.get(etf, etf)
            features = self.swarm_state.features_cache.get(etf, pd.DataFrame())
            pheromone = self.swarm_state.get_pheromone_strength(etf)
            sector_scores[sector_name] = compute_sector_score(features, pheromone)

        sector_ranking = sorted(sector_scores, key=sector_scores.get, reverse=True)
        top_sectors = sector_ranking[:top_n]
        timing_phase1 = time.time() - t0
        logger.info(f"Phase 1 complete in {timing_phase1:.1f}s. Top sectors: {top_sectors}")

        # ── Phase 2: Scan stocks in top sectors ───────────────────────────
        t1 = time.time()

        # Build stock universe from top sectors
        ticker_to_sector: Dict[str, str] = {}
        selected_stocks: List[str] = []
        seen: set = set()
        for sector in top_sectors:
            stocks = SECTOR_STOCKS.get(sector, [])[:stocks_per_sector]
            for s in stocks:
                if s not in seen:
                    selected_stocks.append(s)
                    ticker_to_sector[s] = sector
                    seen.add(s)

        # Expand SwarmState so pheromone matrix covers selected stock tickers
        self.swarm_state.expand_tickers(selected_stocks)

        # Fetch missing stock features in parallel
        missing_stocks = [s for s in selected_stocks if s not in self.swarm_state.features_cache]
        if missing_stocks:
            stock_features = fetch_all_features_parallel(missing_stocks, period=period, max_workers=12)
            self.swarm_state.features_cache.update(stock_features)

        # Fetch options flow for stocks in parallel
        stock_options = self._fetch_options_flow_batch(selected_stocks)
        self._cached_options_flow.update(stock_options)

        # Evaporate pheromone once before stock scan
        rho = self.config.get('aco_rho', 0.1)
        self.swarm_state.evaporate_pheromone(rho / max(len(self.swarm_state.tickers), 1))

        # Run agents on each stock, attach trade plan
        stock_signals: Dict[str, AggregatedSignal] = {}
        for ticker in selected_stocks:
            try:
                market_data = self._build_market_data(ticker)
                agg = self.run_tick(ticker, market_data)

                # Attach sector annotation
                agg.sector = ticker_to_sector.get(ticker)
                agg.sector_rank = (sector_ranking.index(agg.sector) + 1
                                   if agg.sector in sector_ranking else None)

                # Compute and attach trade levels for actionable signals
                raw_df = self.swarm_state.features_cache.get(ticker, pd.DataFrame())
                if not raw_df.empty and agg.action in ('BUY', 'SELL', 'WATCH'):
                    levels = compute_trade_levels(
                        raw_df, agg.direction,
                        atr_multiplier=atr_multiplier,
                        rr_min=rr_min,
                    )
                    if levels and levels.get('risk_reward', 0) >= rr_min:
                        agg.entry_price         = levels['entry_price']
                        agg.stop_loss           = levels['stop_loss']
                        agg.take_profit         = levels['take_profit']
                        agg.risk_reward         = levels['risk_reward']
                        agg.estimated_hold_days = levels['estimated_hold_days']
                        agg.atr                 = levels['atr']
                        agg.entry_type          = levels['entry_type']
                        agg.support_levels      = levels['levels']['support']
                        agg.resistance_levels   = levels['levels']['resistance']

                stock_signals[ticker] = agg
            except Exception as e:
                logger.warning(f"Stock scan failed for {ticker}: {e}")

        timing_phase2 = time.time() - t1
        logger.info(f"Phase 2 complete in {timing_phase2:.1f}s. "
                    f"{sum(1 for s in stock_signals.values() if s.entry_price)} trade plans generated.")

        return SectorScanResult(
            sector_scores=sector_scores,
            sector_ranking=sector_ranking,
            top_sectors=top_sectors,
            sector_signals=sector_signals,
            stock_signals=stock_signals,
            scan_timing={
                'phase1_sector_scoring': timing_phase1,
                'phase2_stock_scan': timing_phase2,
                'total': time.time() - t_total,
            },
        )

    def scan_all_sectors(
        self,
        period: str = "2y",
        rr_min: float = 1.5,
        atr_multiplier: float = 1.5,
        batch_size: int = 50,
        progress_callback=None,
    ) -> SectorScanResult:
        """
        Full-universe sector scan across ALL stocks in SECTOR_STOCKS.

        Phase 1: Score all 11 sector ETFs with swarm agents + pheromone.
        Phase 2: Run swarm agents on ALL stocks across ALL 11 sectors in
                 batches of `batch_size`.

        Args:
            period: yfinance period string.
            rr_min: Minimum risk-reward ratio for trade plans.
            atr_multiplier: ATR multiplier for stop calculation.
            batch_size: Number of tickers per yfinance batch.
            progress_callback: Optional callable with signature:
                (phase: str, sector: str, batch_num: int, total_batches: int,
                 n_success: int, total_stocks: int)
        """
        t_total = time.time()

        # ── Phase 1: Score ALL sectors (same as scan_sectors) ──────────
        t0 = time.time()
        etf_tickers = list(SECTOR_ETFS.values())
        all_init = etf_tickers + MARKET_LEADERS

        # Ensure ETF data is cached
        missing = [t for t in all_init if t not in self.swarm_state.features_cache]
        if missing:
            parallel_results = fetch_all_features_parallel(missing, period=period, max_workers=8)
            self.swarm_state.features_cache.update(parallel_results)

        # Cache VIX once
        if self._cached_vix is None:
            self._cached_vix = get_vix_level()
            self._cached_vix_noise = compute_noise_from_vix(self._cached_vix)

        # Run agents on each sector ETF
        sector_signals: Dict[str, AggregatedSignal] = {}
        for etf in etf_tickers:
            try:
                market_data = self._build_market_data(etf)
                sector_signals[etf] = self.run_tick(etf, market_data)
            except Exception as e:
                logger.warning(f"Sector ETF {etf} failed: {e}")

        # Score each sector
        etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
        sector_scores: Dict[str, float] = {}
        for etf, sig in sector_signals.items():
            sector_name = etf_to_sector.get(etf, etf)
            features = self.swarm_state.features_cache.get(etf, pd.DataFrame())
            pheromone = self.swarm_state.get_pheromone_strength(etf)
            sector_scores[sector_name] = compute_sector_score(features, pheromone)

        # Add scores for sectors that don't have ETFs (use average of existing)
        for sector in SECTOR_STOCKS:
            if sector not in sector_scores:
                sector_scores[sector] = np.mean(list(sector_scores.values())) if sector_scores else 0.5

        sector_ranking = sorted(sector_scores, key=sector_scores.get, reverse=True)
        timing_phase1 = time.time() - t0
        logger.info(f"Phase 1 complete in {timing_phase1:.1f}s. Ranked {len(sector_ranking)} sectors.")

        if progress_callback:
            progress_callback('phase1_complete', '', 0, 0, 0, 0)

        # ── Phase 2: Scan ALL stocks across ALL sectors ────────────────
        t1 = time.time()

        # Build full universe
        ticker_to_sector: Dict[str, str] = {}
        all_stocks: List[str] = []
        seen: set = set()
        for sector in sector_ranking:  # process in ranked order
            stocks = SECTOR_STOCKS.get(sector, [])
            for s in stocks:
                if s not in seen:
                    all_stocks.append(s)
                    ticker_to_sector[s] = sector
                    seen.add(s)

        total_stocks = len(all_stocks)
        logger.info(f"Phase 2: scanning {total_stocks} stocks across {len(sector_ranking)} sectors")

        # Expand SwarmState so pheromone matrix covers all stock tickers
        self.swarm_state.expand_tickers(all_stocks)
        logger.info(f"SwarmState expanded to {self.swarm_state.n_assets} assets for full sector scan")

        # Fetch features in batches with progress
        total_batches = (total_stocks + batch_size - 1) // batch_size
        batch_count = [0]

        def _batch_progress(batch_num, total_b, n_success):
            batch_count[0] = batch_num
            if progress_callback:
                # Determine which sector we're currently in
                current_idx = min(batch_num * batch_size, total_stocks - 1)
                current_sector = ticker_to_sector.get(all_stocks[current_idx], '')
                progress_callback(
                    'fetching', current_sector,
                    batch_num, total_b, n_success, total_stocks
                )

        # Only fetch tickers not already cached
        missing_stocks = [s for s in all_stocks if s not in self.swarm_state.features_cache]
        if missing_stocks:
            fetched = fetch_features_in_batches(
                missing_stocks, period=period, batch_size=batch_size,
                max_workers=12, progress_callback=_batch_progress,
            )
            self.swarm_state.features_cache.update(fetched)

        # Evaporate pheromone once before stock scan
        rho = self.config.get('aco_rho', 0.1)
        self.swarm_state.evaporate_pheromone(rho / max(len(self.swarm_state.tickers), 1))

        # Run agents on each stock, attach trade plan
        stock_signals: Dict[str, AggregatedSignal] = {}
        n_processed = 0
        for ticker in all_stocks:
            try:
                market_data = self._build_market_data(ticker)
                agg = self.run_tick(ticker, market_data)

                # Attach sector annotation
                agg.sector = ticker_to_sector.get(ticker)
                agg.sector_rank = (
                    sector_ranking.index(agg.sector) + 1
                    if agg.sector in sector_ranking else None
                )

                # Compute and attach trade levels for actionable signals
                raw_df = self.swarm_state.features_cache.get(ticker, pd.DataFrame())
                if not raw_df.empty and agg.action in ('BUY', 'SELL', 'WATCH'):
                    levels = compute_trade_levels(
                        raw_df, agg.direction,
                        atr_multiplier=atr_multiplier,
                        rr_min=rr_min,
                    )
                    if levels and levels.get('risk_reward', 0) >= rr_min:
                        agg.entry_price         = levels['entry_price']
                        agg.stop_loss           = levels['stop_loss']
                        agg.take_profit         = levels['take_profit']
                        agg.risk_reward         = levels['risk_reward']
                        agg.estimated_hold_days = levels['estimated_hold_days']
                        agg.atr                 = levels['atr']
                        agg.entry_type          = levels['entry_type']
                        agg.support_levels      = levels['levels']['support']
                        agg.resistance_levels   = levels['levels']['resistance']

                stock_signals[ticker] = agg
            except Exception as e:
                logger.warning(f"Stock scan failed for {ticker}: {e}")

            n_processed += 1
            if progress_callback and n_processed % 25 == 0:
                current_sector = ticker_to_sector.get(ticker, '')
                progress_callback(
                    'analyzing', current_sector,
                    n_processed, total_stocks, len(stock_signals), total_stocks
                )

        timing_phase2 = time.time() - t1
        n_plans = sum(1 for s in stock_signals.values() if s.entry_price is not None)
        logger.info(
            f"Phase 2 complete in {timing_phase2:.1f}s. "
            f"{len(stock_signals)}/{total_stocks} stocks analyzed, "
            f"{n_plans} trade plans generated."
        )

        if progress_callback:
            progress_callback('complete', '', 0, 0, n_plans, total_stocks)

        return SectorScanResult(
            sector_scores=sector_scores,
            sector_ranking=sector_ranking,
            top_sectors=sector_ranking,  # all sectors are "top" in full scan
            sector_signals=sector_signals,
            stock_signals=stock_signals,
            scan_timing={
                'phase1_sector_scoring': timing_phase1,
                'phase2_stock_scan': timing_phase2,
                'total': time.time() - t_total,
                'total_stocks': total_stocks,
                'stocks_analyzed': len(stock_signals),
                'trade_plans': n_plans,
            },
        )

    def get_agent_activity_data(self) -> dict:
        """Extract detailed per-agent metadata for the live activity dashboard."""
        activity = {
            'timing': dict(self._run_timing),
            'agents': {},
        }

        # Boids
        boids = self.agents.get('boids')
        if boids and boids.signal_history:
            last = boids.signal_history[-1]
            activity['agents']['boids'] = {
                'last_signal': {'direction': last.direction, 'confidence': last.confidence,
                                'ticker': last.ticker, 'type': last.signal_type},
                'metadata': last.metadata,
                'total_signals': len(boids.signal_history),
            }

        # Vicsek
        vicsek = self.agents.get('vicsek')
        if vicsek and vicsek.signal_history:
            last = vicsek.signal_history[-1]
            activity['agents']['vicsek'] = {
                'last_signal': {'direction': last.direction, 'confidence': last.confidence,
                                'ticker': last.ticker, 'type': last.signal_type},
                'metadata': last.metadata,
                'total_signals': len(vicsek.signal_history),
            }

        # ACO ants — aggregate stats
        aco_ants = self.agents.get('aco_ants', [])
        if aco_ants:
            all_dirs = []
            all_confs = []
            ant_details = []
            for ant in aco_ants:
                if ant.signal_history:
                    last = ant.signal_history[-1]
                    all_dirs.append(last.direction)
                    all_confs.append(last.confidence)
                    ant_details.append({
                        'ant_id': ant.ant_id,
                        'direction': last.direction,
                        'confidence': last.confidence,
                        'tau': last.metadata.get('tau', 0),
                        'eta': last.metadata.get('eta', 0),
                    })
            if all_dirs:
                activity['agents']['aco'] = {
                    'n_ants': len(aco_ants),
                    'avg_direction': float(np.mean(all_dirs)),
                    'direction_std': float(np.std(all_dirs)),
                    'agreement': float(1.0 - np.std(all_dirs) / (abs(np.mean(all_dirs)) + 1e-9)),
                    'avg_confidence': float(np.mean(all_confs)),
                    'ant_details': ant_details[:20],  # top 20 for display
                    'pheromone_mean': float(self.swarm_state.pheromone_matrix.mean()),
                    'pheromone_max': float(self.swarm_state.pheromone_matrix.max()),
                }

        # PSO
        pso = self.agents.get('pso')
        if pso and pso.signal_history:
            last = pso.signal_history[-1]
            ss = self.swarm_state
            activity['agents']['pso'] = {
                'last_signal': {'direction': last.direction, 'confidence': last.confidence,
                                'ticker': last.ticker},
                'metadata': last.metadata,
                'gbest_score': float(ss.gbest_score) if ss.gbest_score > -np.inf else 0.0,
                'gbest_params': {n: float(v) for n, v in zip(pso.PARAM_NAMES, ss.gbest_position)} if len(ss.gbest_position) == len(pso.PARAM_NAMES) else {},
                'iteration': pso.current_iter,
                'n_particles': pso.n_particles,
                'particle_fitness_stats': {
                    'mean': float(np.mean(ss.pbest_scores[ss.pbest_scores > -np.inf])) if np.any(ss.pbest_scores > -np.inf) else 0.0,
                    'max': float(np.max(ss.pbest_scores)) if np.any(ss.pbest_scores > -np.inf) else 0.0,
                    'pct_explored': float(np.mean(ss.pbest_scores > -np.inf)) * 100,
                },
                'total_signals': len(pso.signal_history),
            }

        # Leader
        leader = self.agents.get('leader')
        if leader and leader.signal_history:
            last = leader.signal_history[-1]
            activity['agents']['leader'] = {
                'last_signal': {'direction': last.direction, 'confidence': last.confidence,
                                'ticker': last.ticker, 'type': last.signal_type},
                'metadata': last.metadata,
                'leader_scores': dict(self.swarm_state.leader_scores),
                'total_signals': len(leader.signal_history),
            }

        # Topological
        topo = self.agents.get('topological')
        if topo and topo.signal_history:
            last = topo.signal_history[-1]
            activity['agents']['topological'] = {
                'last_signal': {'direction': last.direction, 'confidence': last.confidence,
                                'ticker': last.ticker, 'type': last.signal_type},
                'metadata': last.metadata,
                'n_neighbor_groups': len(self.swarm_state.topological_neighbors),
                'total_signals': len(topo.signal_history),
            }

        # Ensemble weights (dynamic)
        activity['ensemble_weights'] = dict(self.aggregator.weights)

        # VIX
        activity['vix'] = {
            'level': self._cached_vix or 20.0,
            'noise': self._cached_vix_noise or 0.2,
        }

        return activity


class BacktestEngine:
    """
    Walk-forward backtesting for the swarm trading system.
    Splits data into train/validation windows for optimization.
    """

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.config = orchestrator.config

    def backtest_ticker(self, ticker: str, period: str = "2y",
                        train_pct: float = 0.7) -> BacktestResult:
        """Run full backtest on a single ticker."""
        df = cached_fetch(ticker, period, "1d")
        if df.empty or len(df) < 60:
            return BacktestResult(0, 0, 0, 0, 0)

        features = compute_swarm_features(df)
        if features.empty:
            return BacktestResult(0, 0, 0, 0, 0)

        split_idx = int(len(features) * train_pct)
        test_features = features.iloc[split_idx:]

        trades = []
        equity = [1.0]
        in_position = False
        entry_price = 0.0
        position_direction = 0.0
        signals_log = []

        for i in range(len(test_features)):
            row_features = test_features.iloc[:i+1]
            market_data = {
                'ticker': ticker,
                'features': row_features,
                'vix_noise': 0.2,
                'options_flow': None,
            }

            signal = self.orchestrator.run_tick(ticker, market_data)
            signals_log.append({
                'date': str(test_features.index[i]),
                'action': signal.action,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'zone': signal.zone,
            })

            current_price = float(test_features['Close'].iloc[i])

            if not in_position:
                if signal.action == 'BUY' and signal.confidence > 0.5:
                    in_position = True
                    entry_price = current_price
                    position_direction = 1.0
                elif signal.action == 'SELL' and signal.confidence > 0.5:
                    in_position = True
                    entry_price = current_price
                    position_direction = -1.0
            else:
                pct_change = (current_price - entry_price) / entry_price * position_direction

                should_exit = (
                    signal.action == 'SELL' and position_direction > 0 or
                    signal.action == 'BUY' and position_direction < 0 or
                    signal.zone == 'ZOR' or
                    pct_change >= self.config.get('zoa_pct', 0.15) or
                    pct_change <= -self.config.get('zor_pct', 0.02)
                )

                if should_exit:
                    trades.append({
                        'entry_date': str(test_features.index[max(0, i-1)]),
                        'exit_date': str(test_features.index[i]),
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'direction': position_direction,
                        'return': pct_change,
                    })
                    equity.append(equity[-1] * (1 + pct_change))
                    in_position = False

        # Compute metrics
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, signals_log=signals_log)

        returns = [t['return'] for t in trades]
        equity_series = pd.Series(equity)
        running_max = equity_series.cummax()
        drawdowns = (equity_series - running_max) / running_max

        total_return = equity[-1] / equity[0] - 1
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252 / max(len(returns), 1))
        max_dd = float(drawdowns.min())
        win_rate = sum(1 for r in returns if r > 0) / len(returns)

        return BacktestResult(
            total_return=float(total_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            n_trades=len(trades),
            trades=trades,
            equity_curve=equity_series,
            signals_log=signals_log,
        )
