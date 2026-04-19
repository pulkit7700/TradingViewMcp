"""
Swarm Intelligence Trading — Agent Implementations
===================================================
Six bio-inspired agents based on Wang et al. (2024) survey:
  1. BoidsMomentumAgent    — Separation/Alignment/Cohesion (Reynolds 1987)
  2. VicsekConsensusAgent  — Direction alignment + VIX noise (Vicsek 1995)
  3. ACOPathAgent          — Ant Colony pheromone trails (Dorigo)
  4. PSOOptimizerAgent     — Particle Swarm parameter optimization (Kennedy-Eberhart 1995)
  5. LeaderFollowerAgent   — Institutional flow detection (Nagy 2010)
  6. TopologicalAgent      — Correlation network (Ballerini 2008)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .swarm_base import BaseAgent, AgentSignal, SwarmState


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BOIDS MOMENTUM AGENT — Reynolds 1987
# ═══════════════════════════════════════════════════════════════════════════════

class BoidsMomentumAgent(BaseAgent):
    """
    Implements Reynolds' three rules on price momentum:
      Separation: avoid concentration risk in correlated positions
      Alignment:  follow weighted average momentum of neighbors
      Cohesion:   pull toward sector average exposure
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.c_sep = config.get('boids_c_sep', 1.5)
        self.c_align = config.get('boids_c_align', 1.0)
        self.c_coh = config.get('boids_c_coh', 0.5)
        self.separation_radius = config.get('boids_sep_radius', 0.02)

    def compute(self, market_data: dict, swarm_state: SwarmState) -> AgentSignal:
        ticker = market_data.get('ticker', '')
        features = market_data.get('features')
        if features is None or features.empty:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)

        # Get neighbors from topological state
        neighbors = swarm_state.topological_neighbors.get(ticker, [])
        if not neighbors:
            # Fallback: use own momentum
            mom = float(features['momentum_5d'].iloc[-1]) if 'momentum_5d' in features.columns else 0.0
            direction = np.clip(mom * 10, -1, 1)
            return AgentSignal(ticker=ticker, signal_type='hold',
                             direction=direction, confidence=0.3, strength=abs(direction))

        # Separation: repulsion from highly correlated positions that are losing
        v_sep = 0.0
        corr_matrix = swarm_state.correlation_matrix
        if corr_matrix is not None and ticker in corr_matrix.columns:
            for nb in neighbors:
                if nb in corr_matrix.columns:
                    corr = corr_matrix.loc[ticker, nb]
                    nb_features = swarm_state.features_cache.get(nb)
                    if nb_features is not None and not nb_features.empty:
                        nb_mom = float(nb_features['momentum_1d'].iloc[-1]) if 'momentum_1d' in nb_features.columns else 0.0
                        # If highly correlated neighbor is dropping, repel (separation)
                        if abs(corr) > (1 - self.separation_radius) and nb_mom < -0.01:
                            v_sep -= corr * nb_mom

        # Alignment: average momentum of neighbors
        v_align = 0.0
        n_valid = 0
        for nb in neighbors:
            nb_features = swarm_state.features_cache.get(nb)
            if nb_features is not None and not nb_features.empty and 'momentum_5d' in nb_features.columns:
                v_align += float(nb_features['momentum_5d'].iloc[-1])
                n_valid += 1
        if n_valid > 0:
            v_align /= n_valid

        # Cohesion: pull toward sector average
        own_mom = float(features['momentum_5d'].iloc[-1]) if 'momentum_5d' in features.columns else 0.0
        v_coh = v_align - own_mom  # attract toward group average

        # Combined velocity
        combined = self.c_sep * v_sep + self.c_align * v_align + self.c_coh * v_coh
        direction = float(np.clip(combined * 20, -1, 1))  # scale to [-1, 1]
        confidence = min(0.3 + n_valid * 0.1, 0.85)
        strength = abs(direction)
        signal_type = 'entry' if strength > 0.5 else ('watch' if strength > 0.25 else 'hold')

        signal = AgentSignal(
            ticker=ticker, signal_type=signal_type, direction=direction,
            confidence=confidence, strength=strength,
            metadata={'v_sep': v_sep, 'v_align': v_align, 'v_coh': v_coh,
                     'n_neighbors': n_valid}
        )
        self.update_history(signal)
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VICSEK CONSENSUS AGENT — Vicsek 1995
# ═══════════════════════════════════════════════════════════════════════════════

class VicsekConsensusAgent(BaseAgent):
    """
    Vicsek alignment model: each asset aligns direction with correlated neighbors,
    plus noise calibrated from VIX. High VIX = high noise = less consensus.

    theta(t+1) = avg_theta_neighbors + noise
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.correlation_threshold = config.get('vicsek_corr_threshold', 0.4)
        self.noise_scaling = config.get('vicsek_noise_scaling', 1.0)

    def compute(self, market_data: dict, swarm_state: SwarmState) -> AgentSignal:
        ticker = market_data.get('ticker', '')
        features = market_data.get('features')
        vix_noise = market_data.get('vix_noise', 0.2)

        if features is None or features.empty:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)

        # Convert momentum to angle [-pi, pi]
        own_mom = float(features['momentum_5d'].iloc[-1]) if 'momentum_5d' in features.columns else 0.0
        own_theta = np.arctan2(own_mom, abs(own_mom) + 1e-9)

        # Gather neighbor directions
        neighbors = swarm_state.topological_neighbors.get(ticker, [])
        neighbor_thetas = []
        corr_matrix = swarm_state.correlation_matrix

        for nb in neighbors:
            nb_features = swarm_state.features_cache.get(nb)
            if nb_features is not None and not nb_features.empty:
                # Check correlation threshold
                if corr_matrix is not None and ticker in corr_matrix.columns and nb in corr_matrix.columns:
                    if abs(corr_matrix.loc[ticker, nb]) < self.correlation_threshold:
                        continue
                nb_mom = float(nb_features['momentum_5d'].iloc[-1]) if 'momentum_5d' in nb_features.columns else 0.0
                neighbor_thetas.append(np.arctan2(nb_mom, abs(nb_mom) + 1e-9))

        if not neighbor_thetas:
            neighbor_thetas = [own_theta]

        # Vicsek average direction (Eq. 5)
        sin_avg = np.mean([np.sin(t) for t in neighbor_thetas])
        cos_avg = np.mean([np.cos(t) for t in neighbor_thetas])
        avg_theta = np.arctan2(sin_avg, cos_avg)

        # Add VIX-calibrated noise
        noise = np.random.normal(0, vix_noise * self.noise_scaling)
        new_theta = avg_theta + noise

        # Update consensus in swarm state
        idx = swarm_state.get_ticker_index(ticker)
        if idx >= 0 and idx < len(swarm_state.consensus_direction):
            swarm_state.consensus_direction[idx] = new_theta

        # Convert back to direction signal
        direction = float(np.clip(np.sin(new_theta) * 2, -1, 1))
        consensus_strength = float(np.sqrt(sin_avg**2 + cos_avg**2))
        confidence = consensus_strength * (1 - vix_noise)  # high VIX reduces confidence
        confidence = float(np.clip(confidence, 0.1, 0.95))
        strength = abs(direction)
        signal_type = 'entry' if strength > 0.5 else ('watch' if strength > 0.25 else 'hold')

        signal = AgentSignal(
            ticker=ticker, signal_type=signal_type, direction=direction,
            confidence=confidence, strength=strength,
            metadata={'avg_theta': float(avg_theta), 'noise': float(noise),
                     'consensus_strength': consensus_strength,
                     'n_aligned': len(neighbor_thetas)}
        )
        self.update_history(signal)
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ACO PATH AGENT — Dorigo (Ant Colony Optimization)
# ═══════════════════════════════════════════════════════════════════════════════

class ACOPathAgent(BaseAgent):
    """
    Ant Colony Optimization for trade path finding.
    Pheromone = volume * price_momentum (reinforcement of successful trades).
    Heuristic = technical signal strength (RSI, MACD).

    Transition: p_ij = (tau_ij^alpha * eta_ij^beta) / sum
    Update:     tau(t+1) = (1-rho)*tau(t) + delta_tau
    """

    def __init__(self, name: str, config: dict, ant_id: int = 0):
        super().__init__(name, config)
        self.ant_id = ant_id
        self.alpha = config.get('aco_alpha', 1.0)    # pheromone importance
        self.beta = config.get('aco_beta', 2.0)       # heuristic importance
        self.rho = config.get('aco_rho', 0.1)         # evaporation rate
        self.q_constant = config.get('aco_q', 100.0)  # deposit constant

    def compute(self, market_data: dict, swarm_state: SwarmState) -> AgentSignal:
        ticker = market_data.get('ticker', '')
        features = market_data.get('features')

        if features is None or features.empty:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)

        idx = swarm_state.get_ticker_index(ticker)
        if idx < 0:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)

        # Compute heuristic (technical signal quality)
        rsi = float(features['rsi'].iloc[-1]) if 'rsi' in features.columns else 50.0
        macd_hist = float(features['macd_hist'].iloc[-1]) if 'macd_hist' in features.columns else 0.0
        volume_z = float(features['volume_zscore'].iloc[-1]) if 'volume_zscore' in features.columns else 0.0
        momentum = float(features['momentum_5d'].iloc[-1]) if 'momentum_5d' in features.columns else 0.0

        # Heuristic: composite signal strength
        rsi_signal = (rsi - 50) / 50  # [-1, 1]
        macd_signal = np.clip(macd_hist * 10, -1, 1)
        eta = abs(rsi_signal * 0.3 + macd_signal * 0.4 + np.clip(momentum * 10, -1, 1) * 0.3) + 0.01

        # Pheromone: volume-weighted momentum reinforcement
        pheromone_row = swarm_state.pheromone_matrix[idx]
        tau = float(np.mean(pheromone_row)) + 0.01

        # ACO transition probability logic
        score = (tau ** self.alpha) * (eta ** self.beta)

        # Deposit pheromone based on signal quality
        if abs(momentum) > 0.02 and abs(volume_z) > 1.0:
            deposit = self.q_constant * abs(momentum) * (1 + abs(volume_z))
            # Deposit on paths to correlated assets
            neighbors = swarm_state.topological_neighbors.get(ticker, [])
            for nb in neighbors:
                nb_idx = swarm_state.get_ticker_index(nb)
                if nb_idx >= 0:
                    swarm_state.update_pheromone(idx, nb_idx, deposit / len(neighbors),
                                                  rho=self.rho / len(swarm_state.tickers))

        # Direction from pheromone + heuristic
        direction_raw = rsi_signal * 0.2 + macd_signal * 0.3 + np.clip(momentum * 10, -1, 1) * 0.3 + \
                        np.clip((tau - 1.0) * 0.5, -0.2, 0.2)
        # Add ant-specific randomness (exploration)
        exploration_noise = np.random.normal(0, 0.05)
        direction = float(np.clip(direction_raw + exploration_noise, -1, 1))

        confidence = float(np.clip(0.3 + score * 0.1, 0.1, 0.9))
        strength = abs(direction)
        signal_type = 'entry' if strength > 0.5 else ('watch' if strength > 0.3 else 'hold')

        signal = AgentSignal(
            ticker=ticker, signal_type=signal_type, direction=direction,
            confidence=confidence, strength=strength,
            metadata={'tau': tau, 'eta': float(eta), 'score': float(score),
                     'ant_id': self.ant_id, 'rsi': rsi, 'macd_hist': macd_hist,
                     'volume_z': volume_z}
        )
        self.update_history(signal)
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PSO OPTIMIZER AGENT — Kennedy & Eberhart 1995
# ═══════════════════════════════════════════════════════════════════════════════

class PSOOptimizerAgent(BaseAgent):
    """
    Particle Swarm Optimization for trading parameter tuning.

    v(t+1) = w*v(t) + c1*r1*(pbest-x) + c2*r2*(gbest-x)
    x(t+1) = x(t) + v(t+1)

    Dimensions: [rsi_period, ma_fast, ma_slow, stop_loss_pct, take_profit_pct,
                 volume_threshold, momentum_lookback]
    Fitness: Sharpe ratio on historical data
    """

    PARAM_NAMES = ['rsi_period', 'ma_fast', 'ma_slow', 'stop_loss_pct',
                   'take_profit_pct', 'volume_threshold', 'momentum_lookback']
    PARAM_BOUNDS = {
        'rsi_period': (5, 30),
        'ma_fast': (5, 20),
        'ma_slow': (20, 60),
        'stop_loss_pct': (0.01, 0.05),
        'take_profit_pct': (0.02, 0.15),
        'volume_threshold': (1.0, 3.0),
        'momentum_lookback': (3, 30),
    }

    def __init__(self, name: str, config: dict, n_particles: int = 30):
        super().__init__(name, config)
        self.n_particles = n_particles
        self.n_dims = len(self.PARAM_NAMES)
        self.w_max = config.get('pso_w_max', 0.9)
        self.w_min = config.get('pso_w_min', 0.4)
        self.c1 = config.get('pso_c1', 2.0)
        self.c2 = config.get('pso_c2', 2.0)
        self.max_iter = config.get('pso_max_iter', 100)
        self.current_iter = 0
        self._initialized = False

    def _initialize_particles(self, swarm_state: SwarmState):
        """Initialize particle positions and velocities."""
        positions = np.zeros((self.n_particles, self.n_dims))
        for i, pname in enumerate(self.PARAM_NAMES):
            lo, hi = self.PARAM_BOUNDS[pname]
            positions[:, i] = np.random.uniform(lo, hi, self.n_particles)

        velocities = np.random.randn(self.n_particles, self.n_dims) * 0.1
        swarm_state.particle_positions = positions
        swarm_state.particle_velocities = velocities
        swarm_state.pbest_positions = positions.copy()
        swarm_state.pbest_scores = np.full(self.n_particles, -np.inf)
        swarm_state.gbest_position = positions[0].copy()
        swarm_state.gbest_score = -np.inf
        self._initialized = True

    def _compute_fitness(self, params: np.ndarray, features: pd.DataFrame) -> float:
        """Vectorized Sharpe ratio computation for given parameter set."""
        if features.empty or len(features) < 60:
            return -np.inf

        rsi_period = max(int(params[0]), 5)
        ma_fast = max(int(params[1]), 3)
        ma_slow = max(int(params[2]), ma_fast + 5)
        stop_loss = params[3]
        take_profit = params[4]
        vol_threshold = params[5]
        mom_lookback = max(int(params[6]), 3)

        close = features['Close'].values
        volume = features['Volume'].values
        n = len(close)

        # Vectorized RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(rsi_period, min_periods=rsi_period).mean().values
        avg_loss = pd.Series(loss).rolling(rsi_period, min_periods=rsi_period).mean().values
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_vals = 100 - (100 / (1 + rs))

        # Vectorized MAs
        ma_f = pd.Series(close).rolling(ma_fast, min_periods=ma_fast).mean().values
        ma_s = pd.Series(close).rolling(ma_slow, min_periods=ma_slow).mean().values

        # Volume z-score
        vol_series = pd.Series(volume)
        vol_mean = vol_series.rolling(20, min_periods=1).mean().values
        vol_std = vol_series.rolling(20, min_periods=1).std().values
        vol_z = (volume - vol_mean) / (vol_std + 1e-9)

        # Momentum
        mom = np.full(n, np.nan)
        mom[mom_lookback:] = (close[mom_lookback:] - close[:-mom_lookback]) / (close[:-mom_lookback] + 1e-9)

        # Entry/exit boolean arrays
        entry = (rsi_vals < 35) & (ma_f > ma_s) & (vol_z > vol_threshold) & (mom > 0)
        exit_sig = rsi_vals > 70

        # Vectorized trade simulation using numpy
        strategy_returns = []
        in_pos = False
        entry_price = 0.0
        for i in range(ma_slow, n):  # skip warmup
            if not in_pos:
                if entry[i]:
                    in_pos = True
                    entry_price = close[i]
            else:
                ret = (close[i] - entry_price) / entry_price
                if ret <= -stop_loss or ret >= take_profit or exit_sig[i]:
                    strategy_returns.append(ret)
                    in_pos = False

        if not strategy_returns:
            return -np.inf

        sr = np.array(strategy_returns)
        sharpe = (sr.mean() / (sr.std() + 1e-9)) * np.sqrt(252 / max(len(sr), 1))
        return float(sharpe)

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def compute(self, market_data: dict, swarm_state: SwarmState) -> AgentSignal:
        ticker = market_data.get('ticker', '')
        features = market_data.get('features')

        if features is None or features.empty:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)

        if not self._initialized:
            self._initialize_particles(swarm_state)

        # Inertia weight decay: w_max -> w_min
        w = self.w_max - (self.w_max - self.w_min) * (self.current_iter / max(self.max_iter, 1))

        positions = swarm_state.particle_positions
        velocities = swarm_state.particle_velocities

        # Evaluate fitness — sample subset of particles for speed when swarm is large
        n_eval = min(self.n_particles, len(positions))
        if n_eval > 20:
            # Evaluate a random subset + always include current gbest holder
            eval_indices = np.random.choice(n_eval, size=20, replace=False)
            gbest_holder = np.argmax(swarm_state.pbest_scores[:n_eval])
            eval_indices = np.unique(np.append(eval_indices, gbest_holder))
        else:
            eval_indices = np.arange(n_eval)

        cached_gbest_fitness = swarm_state.gbest_score
        for p in eval_indices:
            fitness = self._compute_fitness(positions[p], features)
            if fitness > swarm_state.pbest_scores[p]:
                swarm_state.pbest_scores[p] = fitness
                swarm_state.pbest_positions[p] = positions[p].copy()
            if fitness > swarm_state.gbest_score:
                swarm_state.gbest_score = fitness
                swarm_state.gbest_position = positions[p].copy()
                cached_gbest_fitness = fitness

        # Update velocities and positions (PSO Eq. 6)
        r1 = np.random.random(positions.shape)
        r2 = np.random.random(positions.shape)
        velocities = (w * velocities
                      + self.c1 * r1 * (swarm_state.pbest_positions - positions)
                      + self.c2 * r2 * (swarm_state.gbest_position - positions))
        positions = positions + velocities

        # Clamp to bounds — vectorized
        for i, pname in enumerate(self.PARAM_NAMES):
            lo, hi = self.PARAM_BOUNDS[pname]
            positions[:, i] = np.clip(positions[:, i], lo, hi)

        swarm_state.particle_positions = positions
        swarm_state.particle_velocities = velocities
        self.current_iter += 1

        # Use cached gbest fitness (no redundant recomputation)
        gbest = swarm_state.gbest_position
        gbest_fitness = cached_gbest_fitness

        # Use gbest params to generate current signal
        rsi_period = max(int(gbest[0]), 5)
        rsi_vals = self._compute_rsi(features['Close'], rsi_period)
        current_rsi = float(rsi_vals.iloc[-1]) if not rsi_vals.empty else 50.0

        ma_f = features['Close'].rolling(max(int(gbest[1]), 3)).mean()
        ma_s = features['Close'].rolling(max(int(gbest[2]), 10)).mean()
        ma_cross = 1.0 if (not ma_f.empty and not ma_s.empty and
                          ma_f.iloc[-1] > ma_s.iloc[-1]) else -1.0

        direction = float(np.clip(
            (50 - current_rsi) / 50 * 0.5 + ma_cross * 0.5, -1, 1
        ))
        confidence = float(np.clip(0.3 + gbest_fitness * 0.1, 0.1, 0.95)) if gbest_fitness > -np.inf else 0.3
        strength = abs(direction)
        signal_type = 'entry' if strength > 0.5 else 'hold'

        signal = AgentSignal(
            ticker=ticker, signal_type=signal_type, direction=direction,
            confidence=confidence, strength=strength,
            metadata={'gbest_fitness': float(gbest_fitness) if gbest_fitness > -np.inf else 0.0,
                     'gbest_params': {n: float(v) for n, v in zip(self.PARAM_NAMES, gbest)},
                     'iteration': self.current_iter, 'w': float(w)}
        )
        self.update_history(signal)
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LEADER-FOLLOWER AGENT — Nagy 2010 / Strandburg-Peshkin 2015
# ═══════════════════════════════════════════════════════════════════════════════

class LeaderFollowerAgent(BaseAgent):
    """
    Detects institutional/smart money leaders and follows their signals.
    Leader = volume z-score > 2 sigma + price-leading behavior.
    Follows pigeon flock hierarchy (Nagy 2010) and baboon democratic consensus.
    """

    # NSE broad-market ETFs as market leaders (high volume, scale-free correlation)
    MARKET_LEADERS = ['NIFTYBEES.NS', 'BANKBEES.NS', 'ITBEES.NS', 'JUNIORBEES.NS']
    # NSE sector indices as sector leaders
    SECTOR_LEADERS = ['^CNXIT', '^NSEBANK', '^CNXPHARMA', '^CNXAUTO', '^CNXFMCG', '^CNXMETAL', '^CNXENERGY']

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.volume_zscore_threshold = config.get('leader_vol_z_threshold', 2.0)
        self.leader_weight = config.get('leader_weight', 0.6)
        self.democratic_weight = config.get('democratic_weight', 0.4)

    def compute(self, market_data: dict, swarm_state: SwarmState) -> AgentSignal:
        ticker = market_data.get('ticker', '')
        features = market_data.get('features')
        all_features = swarm_state.features_cache

        if features is None or features.empty:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)

        # 1. Identify leaders by volume z-score
        leader_signals = []
        for leader_ticker in self.MARKET_LEADERS + self.SECTOR_LEADERS:
            leader_feat = all_features.get(leader_ticker)
            if leader_feat is not None and not leader_feat.empty:
                vol_z = float(leader_feat['volume_zscore'].iloc[-1]) if 'volume_zscore' in leader_feat.columns else 0.0
                mom = float(leader_feat['momentum_1d'].iloc[-1]) if 'momentum_1d' in leader_feat.columns else 0.0

                # Leader signal strength based on volume conviction
                if abs(vol_z) > self.volume_zscore_threshold:
                    leader_direction = np.sign(mom) * min(abs(vol_z) / 3.0, 1.0)
                    leader_signals.append({
                        'ticker': leader_ticker,
                        'direction': leader_direction,
                        'strength': abs(vol_z) / 3.0,
                        'type': 'market' if leader_ticker in self.MARKET_LEADERS else 'sector'
                    })
                    swarm_state.leader_scores[leader_ticker] = float(abs(vol_z))

        # 2. Democratic consensus (Strandburg-Peshkin): all neighbors vote
        democratic_direction = 0.0
        n_voters = 0
        neighbors = swarm_state.topological_neighbors.get(ticker, [])
        for nb in neighbors:
            nb_feat = all_features.get(nb)
            if nb_feat is not None and not nb_feat.empty and 'momentum_1d' in nb_feat.columns:
                democratic_direction += np.sign(float(nb_feat['momentum_1d'].iloc[-1]))
                n_voters += 1
        if n_voters > 0:
            democratic_direction /= n_voters

        # 3. Combine leader hierarchy + democratic vote
        if leader_signals:
            # Weight leaders by strength
            total_leader_weight = sum(s['strength'] for s in leader_signals)
            leader_direction = sum(s['direction'] * s['strength'] for s in leader_signals) / (total_leader_weight + 1e-9)
            direction = float(np.clip(
                self.leader_weight * leader_direction + self.democratic_weight * democratic_direction,
                -1, 1
            ))
            confidence = float(np.clip(0.4 + total_leader_weight * 0.1, 0.3, 0.95))
        else:
            direction = float(np.clip(democratic_direction, -1, 1))
            confidence = 0.3

        # 4. Options flow as additional leader signal
        options_flow = market_data.get('options_flow')
        if options_flow:
            pc_ratio = options_flow.get('put_call_ratio', 1.0)
            # High put/call = bearish smart money, low = bullish
            options_signal = np.clip((1.0 - pc_ratio) * 0.5, -0.3, 0.3)
            direction = float(np.clip(direction + options_signal, -1, 1))
            if options_flow.get('unusual_calls') or options_flow.get('unusual_puts'):
                confidence = min(confidence + 0.1, 0.95)

        strength = abs(direction)
        signal_type = 'entry' if strength > 0.5 else ('watch' if strength > 0.3 else 'hold')

        signal = AgentSignal(
            ticker=ticker, signal_type=signal_type, direction=direction,
            confidence=confidence, strength=strength,
            metadata={'n_leaders': len(leader_signals), 'democratic_vote': float(democratic_direction),
                     'n_voters': n_voters,
                     'leaders': [s['ticker'] for s in leader_signals]}
        )
        self.update_history(signal)
        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TOPOLOGICAL AGENT — Ballerini 2008 / Cavagna 2010
# ═══════════════════════════════════════════════════════════════════════════════

class TopologicalAgent(BaseAgent):
    """
    Uses topological interaction rules from starling research:
    - 7 nearest neighbors by correlation RANK (not distance)
    - Scale-free correlations: one large-cap affects all
    - Maximum entropy (Bialek 2012): pairwise interactions predict spread

    This agent detects systemic risk propagation and network-based signals.
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.n_neighbors = config.get('topo_n_neighbors', 7)
        # Nifty 50 index as systemic ticker (scale-free correlation, Cavagna 2010)
        self.systemic_ticker = config.get('topo_systemic_ticker', '^NSEI')

    def compute(self, market_data: dict, swarm_state: SwarmState) -> AgentSignal:
        ticker = market_data.get('ticker', '')
        features = market_data.get('features')

        if features is None or features.empty:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.0, strength=0.0)

        # 1. Topological neighbor analysis (Ballerini: 7 nearest by rank)
        neighbors = swarm_state.topological_neighbors.get(ticker, [])
        if not neighbors:
            mom = float(features['momentum_5d'].iloc[-1]) if 'momentum_5d' in features.columns else 0.0
            return AgentSignal(ticker=ticker, signal_type='hold',
                             direction=float(np.clip(mom * 10, -1, 1)),
                             confidence=0.2, strength=abs(mom * 10))

        # 2. Compute network momentum (average of topological neighbors)
        neighbor_signals = []
        for nb in neighbors[:self.n_neighbors]:
            nb_feat = swarm_state.features_cache.get(nb)
            if nb_feat is not None and not nb_feat.empty:
                nb_mom = float(nb_feat['momentum_5d'].iloc[-1]) if 'momentum_5d' in nb_feat.columns else 0.0
                nb_vol_z = float(nb_feat['volume_zscore'].iloc[-1]) if 'volume_zscore' in nb_feat.columns else 0.0
                neighbor_signals.append({
                    'momentum': nb_mom,
                    'volume_z': nb_vol_z,
                    'weight': abs(nb_vol_z) + 1.0  # volume-weighted
                })

        if not neighbor_signals:
            return AgentSignal(ticker=ticker, signal_type='hold', direction=0.0,
                             confidence=0.2, strength=0.0)

        # 3. Scale-free correlation: Nifty 50 (systemic ticker) affects all
        systemic_feat = swarm_state.features_cache.get(self.systemic_ticker)
        systemic_signal = 0.0
        if systemic_feat is not None and not systemic_feat.empty:
            systemic_mom = float(systemic_feat['momentum_1d'].iloc[-1]) if 'momentum_1d' in systemic_feat.columns else 0.0
            systemic_vol_z = float(systemic_feat['volume_zscore'].iloc[-1]) if 'volume_zscore' in systemic_feat.columns else 0.0
            systemic_signal = systemic_mom * (1 + abs(systemic_vol_z)) * 5  # amplified

        # 4. Network-weighted signal
        total_weight = sum(s['weight'] for s in neighbor_signals)
        network_direction = sum(s['momentum'] * s['weight'] for s in neighbor_signals) / (total_weight + 1e-9)
        network_direction *= 10  # scale up

        # 5. Maximum entropy: pairwise interaction spread
        own_mom = float(features['momentum_5d'].iloc[-1]) if 'momentum_5d' in features.columns else 0.0
        spread = own_mom - network_direction / 10
        entropy_signal = -spread * 5  # mean-reversion: if we diverge from network, pull back

        # 6. Combine
        direction = float(np.clip(
            network_direction * 0.4 + systemic_signal * 0.3 + entropy_signal * 0.3,
            -1, 1
        ))

        # Confidence based on network agreement
        momenta = [s['momentum'] for s in neighbor_signals]
        agreement = 1.0 - float(np.std(momenta) / (abs(np.mean(momenta)) + 1e-9))
        confidence = float(np.clip(0.3 + agreement * 0.5, 0.1, 0.9))

        strength = abs(direction)
        signal_type = 'entry' if strength > 0.5 else ('watch' if strength > 0.3 else 'hold')

        signal = AgentSignal(
            ticker=ticker, signal_type=signal_type, direction=direction,
            confidence=confidence, strength=strength,
            metadata={'n_neighbors': len(neighbor_signals),
                     'network_direction': float(network_direction),
                     'systemic_signal': float(systemic_signal),
                     'entropy_signal': float(entropy_signal),
                     'agreement': float(agreement)}
        )
        self.update_history(signal)
        return signal
