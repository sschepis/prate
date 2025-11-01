"""
Advanced Features for PRATE: Multi-symbol support, portfolio-level risk management,
cross-asset correlations, and adaptive parameter tuning.
"""

import copy
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class PortfolioState:
    """Portfolio state across multiple symbols."""
    total_equity: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> qty
    unrealized_pnl: Dict[str, float] = field(default_factory=dict)  # symbol -> pnl
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    max_equity: float = 0.0  # For drawdown tracking
    
    def update_equity(self, prices: Dict[str, float]) -> None:
        """Update total equity based on positions and current prices."""
        total_unrealized = sum(
            self.positions.get(sym, 0.0) * prices.get(sym, 0.0) 
            for sym in self.positions
        )
        self.total_equity = self.realized_pnl + total_unrealized
        self.max_equity = max(self.max_equity, self.total_equity)
    
    def get_drawdown(self) -> float:
        """Get current drawdown as percentage."""
        if self.max_equity == 0:
            return 0.0
        return (self.max_equity - self.total_equity) / self.max_equity


class PortfolioRiskManager:
    """
    Portfolio-level risk management across multiple symbols.
    
    Features:
    - Aggregate position limits
    - Portfolio-level drawdown tracking
    - Cross-asset correlation consideration
    - Symbol-specific risk budgets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize portfolio risk manager.
        
        Args:
            config: Configuration with portfolio limits
        """
        self.config = config
        self.max_portfolio_leverage = config.get('max_portfolio_leverage', 3.0)
        self.max_portfolio_drawdown = config.get('max_portfolio_drawdown', 0.15)
        self.max_symbol_concentration = config.get('max_symbol_concentration', 0.3)
        self.max_correlated_exposure = config.get('max_correlated_exposure', 0.5)
        
        # Risk budgets per symbol
        self.symbol_risk_budgets = config.get('symbol_risk_budgets', {})
        
        # Correlation matrix (to be updated)
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        # Historical data for adaptive tuning
        self.symbol_volatilities: Dict[str, float] = {}
        self.symbol_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def update_correlations(self, symbols: List[str], returns_history: Dict[str, List[float]]) -> None:
        """
        Update correlation matrix based on recent returns.
        
        Args:
            symbols: List of symbols
            returns_history: Dictionary of symbol -> list of returns
        """
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 in returns_history and sym2 in returns_history:
                    r1 = np.array(returns_history[sym1])
                    r2 = np.array(returns_history[sym2])
                    
                    if len(r1) > 10 and len(r2) > 10:
                        corr = np.corrcoef(r1, r2)[0, 1]
                        self.correlation_matrix[(sym1, sym2)] = corr
                        self.correlation_matrix[(sym2, sym1)] = corr
    
    def get_correlation(self, sym1: str, sym2: str) -> float:
        """Get correlation between two symbols."""
        if sym1 == sym2:
            return 1.0
        return self.correlation_matrix.get((sym1, sym2), 0.0)
    
    def check_portfolio_limits(self, state: PortfolioState, 
                               new_position: Dict[str, float],
                               prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if proposed positions violate portfolio limits.
        
        Args:
            state: Current portfolio state
            new_position: Proposed new positions {symbol: qty}
            prices: Current prices {symbol: price}
            
        Returns:
            (allowed, reason) tuple
        """
        # Update state with new positions
        test_positions = state.positions.copy()
        test_positions.update(new_position)
        
        # Check drawdown
        drawdown = state.get_drawdown()
        if drawdown >= self.max_portfolio_drawdown:
            return False, f"Portfolio drawdown {drawdown:.1%} exceeds limit {self.max_portfolio_drawdown:.1%}"
        
        # Check portfolio leverage
        total_notional = sum(
            abs(qty) * prices.get(sym, 0.0)
            for sym, qty in test_positions.items()
        )
        leverage = total_notional / max(state.total_equity, 1.0)
        
        if leverage > self.max_portfolio_leverage:
            return False, f"Portfolio leverage {leverage:.2f} exceeds limit {self.max_portfolio_leverage:.2f}"
        
        # Check symbol concentration
        for sym, qty in test_positions.items():
            position_value = abs(qty) * prices.get(sym, 0.0)
            concentration = position_value / max(state.total_equity, 1.0)
            
            if concentration > self.max_symbol_concentration:
                return False, f"Symbol {sym} concentration {concentration:.1%} exceeds limit {self.max_symbol_concentration:.1%}"
        
        # Check correlated exposure
        symbols = list(test_positions.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = self.get_correlation(sym1, sym2)
                
                if abs(corr) > 0.7:  # Highly correlated
                    pos1_value = test_positions[sym1] * prices.get(sym1, 0.0)
                    pos2_value = test_positions[sym2] * prices.get(sym2, 0.0)
                    
                    # If same direction and highly correlated
                    if np.sign(pos1_value) == np.sign(pos2_value):
                        combined_exposure = (abs(pos1_value) + abs(pos2_value)) / max(state.total_equity, 1.0)
                        
                        if combined_exposure > self.max_correlated_exposure:
                            return False, f"Correlated exposure {sym1}/{sym2} ({corr:.2f}) exceeds limit"
        
        return True, "OK"
    
    def allocate_risk_budget(self, symbols: List[str]) -> Dict[str, float]:
        """
        Allocate risk budget across symbols based on volatility.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dictionary of symbol -> risk budget (as fraction of total)
        """
        # Equal-weighted if no volatility data
        if not self.symbol_volatilities:
            equal_weight = 1.0 / max(len(symbols), 1)
            return {sym: equal_weight for sym in symbols}
        
        # Inverse volatility weighting
        inv_vol = {}
        for sym in symbols:
            vol = self.symbol_volatilities.get(sym, 1.0)
            inv_vol[sym] = 1.0 / max(vol, 0.01)
        
        total_inv_vol = sum(inv_vol.values())
        
        return {
            sym: inv_vol[sym] / total_inv_vol
            for sym in symbols
        }
    
    def update_symbol_stats(self, symbol: str, price_change: float) -> None:
        """
        Update symbol statistics for adaptive tuning.
        
        Args:
            symbol: Symbol identifier
            price_change: Price change (return)
        """
        self.symbol_returns[symbol].append(price_change)
        
        # Update volatility (rolling std)
        if len(self.symbol_returns[symbol]) >= 20:
            returns = list(self.symbol_returns[symbol])
            self.symbol_volatilities[symbol] = float(np.std(returns))


class AdaptiveParameterTuner:
    """
    Adaptive parameter tuning based on market conditions.
    
    Features:
    - Regime-based parameter adjustment
    - Performance feedback
    - Volatility-adaptive sizing
    """
    
    def __init__(self, base_params: Dict):
        """
        Initialize adaptive tuner.
        
        Args:
            base_params: Base parameter configuration
        """
        self.base_params = base_params
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.regime_params: Dict[str, Dict] = {}
        
        # Define regime-specific multipliers
        self.regime_multipliers = {
            'TREND': {'tau_mult': 1.0, 'risk_mult': 1.2},
            'RANGE': {'tau_mult': 0.8, 'risk_mult': 0.8},
            'VOLX': {'tau_mult': 0.6, 'risk_mult': 0.5},
            'QUIET': {'tau_mult': 1.2, 'risk_mult': 1.0},
        }
    
    def get_adapted_params(self, regime: str, volatility: float, 
                          recent_performance: Optional[float] = None) -> Dict:
        """
        Get adapted parameters based on current conditions.
        
        Args:
            regime: Current market regime
            volatility: Current volatility estimate
            recent_performance: Recent Sharpe or PnL metric
            
        Returns:
            Adapted parameter dictionary
        """
        # Deep copy to avoid modifying original
        params = copy.deepcopy(self.base_params)
        
        # Apply regime multipliers
        mult = self.regime_multipliers.get(regime, {'tau_mult': 1.0, 'risk_mult': 1.0})
        
        # Adjust entropy target
        if 'tau' in params and 'H_star' in params['tau']:
            params['tau']['H_star'] = self.base_params['tau']['H_star'] * mult['tau_mult']
        
        # Adjust risk limits based on volatility
        if 'risk' in params and 'max_position_size' in params['risk']:
            # Reduce position size in high volatility
            # vol_factor: higher vol -> lower factor -> smaller position
            vol_factor = max(0.5, min(2.0, 0.02 / max(volatility, 0.001)))
            params['risk']['max_position_size'] = (
                self.base_params['risk']['max_position_size'] * 
                vol_factor * mult['risk_mult']
            )
        
        # Adjust based on recent performance
        if recent_performance is not None and recent_performance < 0:
            # Reduce risk after losses
            if 'risk' in params and 'max_trade_risk_pct' in params['risk']:
                params['risk']['max_trade_risk_pct'] = (
                    self.base_params['risk']['max_trade_risk_pct'] * 0.7
                )
        
        return params
    
    def update_performance(self, param_set: str, performance: float) -> None:
        """
        Record performance for a parameter set.
        
        Args:
            param_set: Identifier for parameter set
            performance: Performance metric
        """
        self.performance_history[param_set].append(performance)
    
    def get_best_param_set(self) -> Optional[str]:
        """
        Get best performing parameter set.
        
        Returns:
            Identifier of best parameter set, or None
        """
        if not self.performance_history:
            return None
        
        avg_performance = {
            param_set: np.mean(list(perf_history))
            for param_set, perf_history in self.performance_history.items()
            if len(perf_history) >= 10
        }
        
        if not avg_performance:
            return None
        
        return max(avg_performance.items(), key=lambda x: x[1])[0]


class MultiSymbolCoordinator:
    """
    Coordinates trading across multiple symbols with portfolio-level risk management.
    """
    
    def __init__(self, symbols: List[str], config: Dict):
        """
        Initialize multi-symbol coordinator.
        
        Args:
            symbols: List of symbols to trade
            config: Configuration dictionary
        """
        self.symbols = symbols
        self.config = config
        
        self.portfolio_risk = PortfolioRiskManager(config.get('portfolio', {}))
        self.adaptive_tuner = AdaptiveParameterTuner(config)
        
        # Per-symbol state
        self.symbol_regimes: Dict[str, str] = {}
        self.symbol_volatilities: Dict[str, float] = {}
        
        # Portfolio state
        self.portfolio = PortfolioState(
            total_equity=config.get('initial_equity', 100000.0)
        )
        self.portfolio.max_equity = self.portfolio.total_equity
    
    def update_market_state(self, symbol: str, regime: str, 
                           volatility: float, price_change: float) -> None:
        """
        Update market state for a symbol.
        
        Args:
            symbol: Symbol identifier
            regime: Current regime
            volatility: Current volatility
            price_change: Recent price change
        """
        self.symbol_regimes[symbol] = regime
        self.symbol_volatilities[symbol] = volatility
        
        self.portfolio_risk.update_symbol_stats(symbol, price_change)
    
    def get_symbol_params(self, symbol: str) -> Dict:
        """
        Get adapted parameters for a symbol.
        
        Args:
            symbol: Symbol identifier
            
        Returns:
            Adapted parameter dictionary
        """
        regime = self.symbol_regimes.get(symbol, 'UNKNOWN')
        volatility = self.symbol_volatilities.get(symbol, 1.0)
        
        return self.adaptive_tuner.get_adapted_params(regime, volatility)
    
    def propose_trades(self, proposals: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Vet trade proposals across portfolio.
        
        Args:
            proposals: Dictionary of symbol -> trade proposal
            
        Returns:
            Dictionary of approved symbol -> trade proposal
        """
        # Get current prices (would come from market data)
        current_prices = {}  # Placeholder
        
        # Check each proposal
        approved = {}
        
        for symbol, proposal in proposals.items():
            new_position = {symbol: proposal.get('qty', 0.0)}
            
            allowed, reason = self.portfolio_risk.check_portfolio_limits(
                self.portfolio, new_position, current_prices
            )
            
            if allowed:
                approved[symbol] = proposal
        
        return approved
    
    def update_positions(self, fills: Dict[str, Dict]) -> None:
        """
        Update portfolio positions after fills.
        
        Args:
            fills: Dictionary of symbol -> fill information
        """
        for symbol, fill in fills.items():
            qty = fill.get('qty', 0.0)
            pnl = fill.get('pnl', 0.0)
            
            # Update position
            current_pos = self.portfolio.positions.get(symbol, 0.0)
            self.portfolio.positions[symbol] = current_pos + qty
            
            # Update PnL
            self.portfolio.realized_pnl += pnl
            self.portfolio.daily_pnl += pnl
    
    def allocate_capital(self) -> Dict[str, float]:
        """
        Allocate capital across symbols based on risk budgets.
        
        Returns:
            Dictionary of symbol -> capital allocation
        """
        risk_budgets = self.portfolio_risk.allocate_risk_budget(self.symbols)
        
        total_capital = self.portfolio.total_equity
        
        return {
            symbol: total_capital * budget
            for symbol, budget in risk_budgets.items()
        }
