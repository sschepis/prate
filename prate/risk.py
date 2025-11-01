"""
Risk kernel for KAM (Kolmogorov-Arnold-Moser) protection.
"""

from typing import Dict, Optional
from .types import TradeIntent


class RiskKernel:
    """
    Risk Kernel implementing KAM protection.
    
    Enforces limits on:
    - Per-trade risk
    - Daily drawdown
    - Leverage
    - Position exposure
    """
    
    def __init__(self, limits: Dict[str, float]):
        """
        Initialize risk kernel.
        
        Args:
            limits: Dictionary of risk limits
                - max_trade_risk_pct: Maximum risk per trade (%)
                - daily_dd_pct: Maximum daily drawdown (%)
                - var_max: Maximum VaR
                - leverage_cap: Maximum leverage
                - max_position: Maximum position size
        """
        self.limits = limits
        self.daily_pnl = 0.0
        self.initial_equity = 100000.0  # Default, should be set externally
    
    def vet_intent(
        self, 
        intent: TradeIntent, 
        account_state: Dict[str, float]
    ) -> Optional[TradeIntent]:
        """
        Vet trade intent against risk limits.
        
        Args:
            intent: Proposed trade intent
            account_state: Current account state (equity, positions, etc.)
            
        Returns:
            Vetted intent or None if rejected
        """
        equity = account_state.get('equity', self.initial_equity)
        position = account_state.get('position', 0.0)
        
        # Check daily drawdown
        daily_dd_pct = -100.0 * self.daily_pnl / self.initial_equity
        if daily_dd_pct >= self.limits.get('daily_dd_pct', 5.0):
            return None  # Halt trading
        
        # Check trade size vs max risk
        trade_value = abs(intent.qty * (intent.price or account_state.get('price', 0.0)))
        max_trade_value = equity * (self.limits.get('max_trade_risk_pct', 1.0) / 100.0)
        
        if trade_value > max_trade_value:
            # Clip quantity
            if intent.price and intent.price > 0:
                intent.qty = max_trade_value / intent.price
        
        # Check leverage
        new_position = position + intent.qty
        leverage = abs(new_position * (intent.price or account_state.get('price', 1.0))) / equity
        
        if leverage > self.limits.get('leverage_cap', 3.0):
            return None  # Reject
        
        # Check max position
        if abs(new_position) > self.limits.get('max_position', float('inf')):
            return None
        
        return intent
    
    def after_fill(self, fill: Dict[str, float], account_state: Dict[str, float]) -> None:
        """
        Update risk state after fill.
        
        Args:
            fill: Fill information (pnl, qty, price)
            account_state: Updated account state
        """
        pnl = fill.get('pnl', 0.0)
        self.daily_pnl += pnl
    
    def should_halt(self, metrics: Dict[str, float]) -> bool:
        """
        Check if trading should be halted based on metrics.
        
        Args:
            metrics: Current metrics (dd, var, entropy divergence)
            
        Returns:
            True if trading should halt
        """
        # Check daily drawdown
        if metrics.get('daily_dd', 0.0) <= -self.limits.get('daily_dd_pct', 5.0):
            return True
        
        # Check VaR
        if metrics.get('var_99', 0.0) > self.limits.get('var_max', 0.05):
            return True
        
        # Check entropy divergence (system stability)
        if metrics.get('entropy_diverged', False):
            return True
        
        return False
    
    def reset_daily(self) -> None:
        """Reset daily PnL tracker."""
        self.daily_pnl = 0.0
    
    def set_initial_equity(self, equity: float) -> None:
        """Set initial equity for drawdown calculations."""
        self.initial_equity = equity
