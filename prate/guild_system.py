"""
Guild System Module - Guild-specific proposal generators and performance tracking.

Implements:
- Guild-specific proposal generators for each trading archetype
- Style parameter schemas
- Guild performance tracking
- Inter-guild communication
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .types import GuildID, Action, Observation, Side, TradeIntent


@dataclass
class GuildStyle:
    """Style parameters for a guild."""
    aggression: float = 0.5  # 0.0 = passive, 1.0 = aggressive
    hold_time: float = 60.0  # Target hold time in seconds
    position_size_factor: float = 1.0  # Position sizing multiplier
    risk_tolerance: float = 0.5  # Risk tolerance (0.0-1.0)
    entry_threshold: float = 0.5  # Signal strength threshold for entry
    exit_threshold: float = 0.3  # Signal strength threshold for exit


@dataclass
class GuildPerformance:
    """Performance metrics for a guild."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_hold_time: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Recent performance window
    recent_pnls: List[float] = field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_trade(
        self,
        pnl: float,
        fees: float,
        hold_time: float,
        entry_price: float,
        exit_price: float
    ) -> None:
        """Update performance metrics with new trade."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_fees += fees
        
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
        
        # Update win rate
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        # Update average hold time
        self.avg_hold_time = (
            (self.avg_hold_time * (self.total_trades - 1) + hold_time) / 
            self.total_trades
        )
        
        # Track recent performance
        self.recent_pnls.append(pnl)
        if len(self.recent_pnls) > 100:
            self.recent_pnls.pop(0)
        
        self.recent_trades.append({
            'pnl': pnl,
            'fees': fees,
            'hold_time': hold_time,
            'entry_price': entry_price,
            'exit_price': exit_price
        })
        if len(self.recent_trades) > 100:
            self.recent_trades.pop(0)
        
        # Update Sharpe ratio (using recent PnLs)
        if len(self.recent_pnls) > 1:
            pnl_array = np.array(self.recent_pnls)
            self.sharpe_ratio = (
                np.mean(pnl_array) / (np.std(pnl_array) + 1e-8) * np.sqrt(252)
            )
        
        # Update profit factor
        gross_profit = sum(p for p in self.recent_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.recent_pnls if p < 0))
        if gross_loss > 0:
            self.profit_factor = gross_profit / gross_loss
        else:
            self.profit_factor = float('inf') if gross_profit > 0 else 0.0


class ProposalGenerator:
    """Base class for guild-specific proposal generators."""
    
    def __init__(self, guild_id: GuildID, style: GuildStyle):
        """
        Initialize proposal generator.
        
        Args:
            guild_id: Guild identifier
            style: Style parameters
        """
        self.guild_id = guild_id
        self.style = style
    
    def generate_proposal(
        self,
        observation: Observation,
        embedding: np.ndarray,
        context: Dict[str, Any]
    ) -> Optional[Action]:
        """
        Generate trading proposal.
        
        Args:
            observation: Current market observation
            embedding: Hilbert space embedding
            context: Additional context (memory, etc.)
            
        Returns:
            Action proposal or None
        """
        raise NotImplementedError("Subclasses must implement generate_proposal")
    
    def compute_signal_strength(
        self,
        observation: Observation,
        embedding: np.ndarray
    ) -> float:
        """
        Compute signal strength for this guild's strategy.
        
        Args:
            observation: Current market observation
            embedding: Hilbert space embedding
            
        Returns:
            Signal strength (0.0-1.0)
        """
        raise NotImplementedError("Subclasses must implement compute_signal_strength")


class TrendFollowGenerator(ProposalGenerator):
    """Trend-following proposal generator (G_TF)."""
    
    def __init__(self, style: GuildStyle):
        super().__init__(GuildID.TF, style)
    
    def compute_signal_strength(
        self,
        observation: Observation,
        embedding: np.ndarray
    ) -> float:
        """Compute trend-following signal strength."""
        # Use EMA slope and regime for trend detection
        ema_slope = observation.ema_slope
        trend_prob = observation.regime_soft.get('TREND', 0.0)
        
        # Combine signals
        signal = abs(ema_slope) * 100 * trend_prob
        return min(signal, 1.0)
    
    def generate_proposal(
        self,
        observation: Observation,
        embedding: np.ndarray,
        context: Dict[str, Any]
    ) -> Optional[Action]:
        """Generate trend-following proposal."""
        signal = self.compute_signal_strength(observation, embedding)
        
        # Check entry threshold
        if signal < self.style.entry_threshold:
            return None
        
        # Determine direction from EMA slope
        direction = 1.0 if observation.ema_slope > 0 else -1.0
        
        # Position size based on signal strength and aggression
        base_size = 0.1  # Base position size
        size_factor = signal * self.style.aggression * self.style.position_size_factor
        delta_q = direction * base_size * size_factor
        
        return Action(
            style=self.guild_id,
            delta_q=delta_q,
            params={
                'signal_strength': signal,
                'hold_time_target': self.style.hold_time,
                'stop_loss': -0.02 * self.style.risk_tolerance,
                'take_profit': 0.03 / self.style.risk_tolerance
            }
        )


class MeanRevertGenerator(ProposalGenerator):
    """Mean-reversion proposal generator (G_MR)."""
    
    def __init__(self, style: GuildStyle):
        super().__init__(GuildID.MR, style)
    
    def compute_signal_strength(
        self,
        observation: Observation,
        embedding: np.ndarray
    ) -> float:
        """Compute mean-reversion signal strength."""
        # Use RSI for overbought/oversold
        rsi = observation.rsi_short
        range_prob = observation.regime_soft.get('RANGE', 0.0)
        
        # Signal is stronger when RSI is extreme
        rsi_deviation = abs(rsi - 50.0) / 50.0
        signal = rsi_deviation * range_prob
        
        return min(signal, 1.0)
    
    def generate_proposal(
        self,
        observation: Observation,
        embedding: np.ndarray,
        context: Dict[str, Any]
    ) -> Optional[Action]:
        """Generate mean-reversion proposal."""
        signal = self.compute_signal_strength(observation, embedding)
        
        # Check entry threshold
        if signal < self.style.entry_threshold:
            return None
        
        # Revert against RSI signal
        rsi = observation.rsi_short
        if rsi > 70:  # Overbought - sell
            direction = -1.0
        elif rsi < 30:  # Oversold - buy
            direction = 1.0
        else:
            return None
        
        # Position size
        base_size = 0.08
        size_factor = signal * self.style.aggression * self.style.position_size_factor
        delta_q = direction * base_size * size_factor
        
        return Action(
            style=self.guild_id,
            delta_q=delta_q,
            params={
                'signal_strength': signal,
                'hold_time_target': self.style.hold_time,
                'stop_loss': -0.015 * self.style.risk_tolerance,
                'take_profit': 0.02 / self.style.risk_tolerance
            }
        )


class BreakoutGenerator(ProposalGenerator):
    """Breakout proposal generator (G_BR)."""
    
    def __init__(self, style: GuildStyle):
        super().__init__(GuildID.BR, style)
    
    def compute_signal_strength(
        self,
        observation: Observation,
        embedding: np.ndarray
    ) -> float:
        """Compute breakout signal strength."""
        # Use ATR for volatility and volume
        atr = observation.atr
        volx_prob = observation.regime_soft.get('VOLX', 0.0)
        pressure = abs(observation.pressure)
        
        # Breakout signal from volatility expansion
        signal = (atr / observation.mid) * 100 * volx_prob * (1 + pressure)
        
        return min(signal, 1.0)
    
    def generate_proposal(
        self,
        observation: Observation,
        embedding: np.ndarray,
        context: Dict[str, Any]
    ) -> Optional[Action]:
        """Generate breakout proposal."""
        signal = self.compute_signal_strength(observation, embedding)
        
        # Check entry threshold
        if signal < self.style.entry_threshold:
            return None
        
        # Direction from pressure
        direction = 1.0 if observation.pressure > 0 else -1.0
        
        # Aggressive position sizing for breakouts
        base_size = 0.12
        size_factor = signal * self.style.aggression * self.style.position_size_factor
        delta_q = direction * base_size * size_factor
        
        return Action(
            style=self.guild_id,
            delta_q=delta_q,
            params={
                'signal_strength': signal,
                'hold_time_target': self.style.hold_time * 0.5,  # Shorter holds
                'stop_loss': -0.025 * self.style.risk_tolerance,
                'take_profit': 0.04 / self.style.risk_tolerance
            }
        )


class LiquidityMakerGenerator(ProposalGenerator):
    """Liquidity maker proposal generator (G_LM)."""
    
    def __init__(self, style: GuildStyle):
        super().__init__(GuildID.LM, style)
    
    def compute_signal_strength(
        self,
        observation: Observation,
        embedding: np.ndarray
    ) -> float:
        """Compute liquidity making signal strength."""
        # Favor tight spreads and low volatility
        spread_ratio = observation.spread / observation.mid
        quiet_prob = observation.regime_soft.get('QUIET', 0.0)
        
        # Signal is stronger with tighter spreads
        signal = (1.0 - min(spread_ratio * 1000, 1.0)) * quiet_prob
        
        return min(signal, 1.0)
    
    def generate_proposal(
        self,
        observation: Observation,
        embedding: np.ndarray,
        context: Dict[str, Any]
    ) -> Optional[Action]:
        """Generate liquidity making proposal."""
        signal = self.compute_signal_strength(observation, embedding)
        
        # Check entry threshold
        if signal < self.style.entry_threshold:
            return None
        
        # Alternate sides to capture spread
        # Use book imbalance to pick side
        direction = 1.0 if observation.book_imbalance > 0 else -1.0
        
        # Small position sizes, frequent trades
        base_size = 0.05
        size_factor = signal * self.style.aggression * self.style.position_size_factor
        delta_q = direction * base_size * size_factor
        
        return Action(
            style=self.guild_id,
            delta_q=delta_q,
            params={
                'signal_strength': signal,
                'hold_time_target': self.style.hold_time * 0.3,  # Very short holds
                'post_only': True,  # Always post
                'spread_capture': observation.spread * 0.5
            }
        )


class GuildManager:
    """Manages multiple guilds and their interactions."""
    
    def __init__(self):
        """Initialize guild manager."""
        self.guilds: Dict[GuildID, ProposalGenerator] = {}
        self.performances: Dict[GuildID, GuildPerformance] = {}
        self.styles: Dict[GuildID, GuildStyle] = {}
        
        # Initialize default guilds
        self._initialize_default_guilds()
    
    def _initialize_default_guilds(self) -> None:
        """Initialize default guild configurations."""
        # Trend-follow
        tf_style = GuildStyle(
            aggression=0.6,
            hold_time=120.0,
            position_size_factor=1.0,
            risk_tolerance=0.6,
            entry_threshold=0.4
        )
        self.add_guild(GuildID.TF, TrendFollowGenerator(tf_style), tf_style)
        
        # Mean-revert
        mr_style = GuildStyle(
            aggression=0.5,
            hold_time=60.0,
            position_size_factor=0.8,
            risk_tolerance=0.5,
            entry_threshold=0.5
        )
        self.add_guild(GuildID.MR, MeanRevertGenerator(mr_style), mr_style)
        
        # Breakout
        br_style = GuildStyle(
            aggression=0.8,
            hold_time=45.0,
            position_size_factor=1.2,
            risk_tolerance=0.7,
            entry_threshold=0.6
        )
        self.add_guild(GuildID.BR, BreakoutGenerator(br_style), br_style)
        
        # Liquidity maker
        lm_style = GuildStyle(
            aggression=0.3,
            hold_time=20.0,
            position_size_factor=0.5,
            risk_tolerance=0.3,
            entry_threshold=0.3
        )
        self.add_guild(GuildID.LM, LiquidityMakerGenerator(lm_style), lm_style)
    
    def add_guild(
        self,
        guild_id: GuildID,
        generator: ProposalGenerator,
        style: GuildStyle
    ) -> None:
        """Add a guild to the manager."""
        self.guilds[guild_id] = generator
        self.performances[guild_id] = GuildPerformance()
        self.styles[guild_id] = style
    
    def get_proposals(
        self,
        observation: Observation,
        embedding: np.ndarray,
        context: Dict[str, Any]
    ) -> List[Tuple[GuildID, Action]]:
        """
        Get proposals from all guilds.
        
        Args:
            observation: Current market observation
            embedding: Hilbert space embedding
            context: Additional context
            
        Returns:
            List of (guild_id, action) tuples
        """
        proposals = []
        
        for guild_id, generator in self.guilds.items():
            action = generator.generate_proposal(observation, embedding, context)
            if action is not None:
                proposals.append((guild_id, action))
        
        return proposals
    
    def select_best_proposal(
        self,
        proposals: List[Tuple[GuildID, Action]]
    ) -> Optional[Tuple[GuildID, Action]]:
        """
        Select best proposal based on guild performance.
        
        Args:
            proposals: List of (guild_id, action) tuples
            
        Returns:
            Best (guild_id, action) tuple or None
        """
        if not proposals:
            return None
        
        # Score proposals based on guild performance and signal strength
        scores = []
        for guild_id, action in proposals:
            perf = self.performances[guild_id]
            
            # Performance score (Sharpe ratio + win rate)
            perf_score = (perf.sharpe_ratio + perf.win_rate) / 2.0
            
            # Signal strength
            signal_strength = action.params.get('signal_strength', 0.5)
            
            # Combined score
            score = 0.6 * signal_strength + 0.4 * perf_score
            scores.append((score, guild_id, action))
        
        # Select highest scoring proposal
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_guild, best_action = scores[0]
        
        return (best_guild, best_action)
    
    def update_performance(
        self,
        guild_id: GuildID,
        pnl: float,
        fees: float,
        hold_time: float,
        entry_price: float,
        exit_price: float
    ) -> None:
        """Update guild performance after trade completion."""
        if guild_id in self.performances:
            self.performances[guild_id].update_trade(
                pnl, fees, hold_time, entry_price, exit_price
            )
    
    def get_performance_summary(self) -> Dict[GuildID, Dict[str, float]]:
        """Get performance summary for all guilds."""
        summary = {}
        
        for guild_id, perf in self.performances.items():
            summary[guild_id] = {
                'total_trades': perf.total_trades,
                'win_rate': perf.win_rate,
                'total_pnl': perf.total_pnl,
                'sharpe_ratio': perf.sharpe_ratio,
                'profit_factor': perf.profit_factor,
                'avg_hold_time': perf.avg_hold_time
            }
        
        return summary
