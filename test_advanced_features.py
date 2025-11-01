"""
Tests for Advanced Features: Multi-symbol, Portfolio Risk, Adaptive Tuning.
"""

import pytest
import numpy as np

from prate.advanced_features import (
    PortfolioState,
    PortfolioRiskManager,
    AdaptiveParameterTuner,
    MultiSymbolCoordinator
)


# ===== PORTFOLIO STATE TESTS =====

def test_portfolio_state_init():
    """Test portfolio state initialization."""
    state = PortfolioState(total_equity=10000.0)
    
    assert state.total_equity == 10000.0
    assert len(state.positions) == 0
    assert state.realized_pnl == 0.0


def test_portfolio_state_update_equity():
    """Test equity update with positions."""
    state = PortfolioState(total_equity=10000.0)
    state.positions = {'BTCUSDT': 0.1, 'ETHUSDT': 1.0}
    state.realized_pnl = 1000.0
    
    prices = {'BTCUSDT': 50000.0, 'ETHUSDT': 3000.0}
    
    state.update_equity(prices)
    
    # 0.1 * 50000 + 1.0 * 3000 + 1000 = 9000
    expected_equity = 0.1 * 50000.0 + 1.0 * 3000.0 + 1000.0
    assert abs(state.total_equity - expected_equity) < 0.01


def test_portfolio_state_drawdown():
    """Test drawdown calculation."""
    state = PortfolioState(total_equity=10000.0)
    state.max_equity = 12000.0
    state.total_equity = 9000.0
    
    drawdown = state.get_drawdown()
    
    expected = (12000.0 - 9000.0) / 12000.0
    assert abs(drawdown - expected) < 0.001


# ===== PORTFOLIO RISK MANAGER TESTS =====

def test_portfolio_risk_manager_init():
    """Test portfolio risk manager initialization."""
    config = {
        'max_portfolio_leverage': 2.0,
        'max_portfolio_drawdown': 0.10,
        'max_symbol_concentration': 0.25
    }
    
    manager = PortfolioRiskManager(config)
    
    assert manager.max_portfolio_leverage == 2.0
    assert manager.max_portfolio_drawdown == 0.10
    assert manager.max_symbol_concentration == 0.25


def test_portfolio_risk_correlations():
    """Test correlation matrix update."""
    config = {}
    manager = PortfolioRiskManager(config)
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    returns_history = {
        'BTCUSDT': [0.01, -0.02, 0.015, -0.01, 0.02, 0.01, -0.005, 0.03, -0.01, 0.02, 0.01, -0.015],
        'ETHUSDT': [0.012, -0.018, 0.014, -0.008, 0.025, 0.009, -0.006, 0.028, -0.012, 0.022, 0.011, -0.014]
    }
    
    manager.update_correlations(symbols, returns_history)
    
    # Should have correlation between the two
    corr = manager.get_correlation('BTCUSDT', 'ETHUSDT')
    assert -1.0 <= corr <= 1.0
    assert corr > 0.5  # Should be positively correlated
    
    # Self-correlation should be 1
    assert manager.get_correlation('BTCUSDT', 'BTCUSDT') == 1.0


def test_portfolio_risk_drawdown_limit():
    """Test portfolio limit check on drawdown."""
    config = {
        'max_portfolio_drawdown': 0.10,
        'max_portfolio_leverage': 10.0,
        'max_symbol_concentration': 1.0
    }
    manager = PortfolioRiskManager(config)
    
    state = PortfolioState(total_equity=9000.0)
    state.max_equity = 12000.0  # 25% drawdown
    
    new_position = {'BTCUSDT': 0.01}
    prices = {'BTCUSDT': 50000.0}
    
    allowed, reason = manager.check_portfolio_limits(state, new_position, prices)
    
    assert not allowed
    assert 'drawdown' in reason.lower()


def test_portfolio_risk_leverage_limit():
    """Test portfolio limit check on leverage."""
    config = {
        'max_portfolio_leverage': 2.0,
        'max_portfolio_drawdown': 0.5,
        'max_symbol_concentration': 1.0
    }
    manager = PortfolioRiskManager(config)
    
    state = PortfolioState(total_equity=10000.0)
    state.max_equity = 10000.0
    
    # Try to take position worth 3x equity
    new_position = {'BTCUSDT': 0.6}  # 0.6 * 50000 = 30000
    prices = {'BTCUSDT': 50000.0}
    
    allowed, reason = manager.check_portfolio_limits(state, new_position, prices)
    
    assert not allowed
    assert 'leverage' in reason.lower()


def test_portfolio_risk_concentration_limit():
    """Test portfolio limit check on symbol concentration."""
    config = {
        'max_portfolio_leverage': 10.0,
        'max_portfolio_drawdown': 0.5,
        'max_symbol_concentration': 0.2
    }
    manager = PortfolioRiskManager(config)
    
    state = PortfolioState(total_equity=10000.0)
    state.max_equity = 10000.0
    
    # Try to take position worth 40% of equity
    new_position = {'BTCUSDT': 0.08}  # 0.08 * 50000 = 4000 (40%)
    prices = {'BTCUSDT': 50000.0}
    
    allowed, reason = manager.check_portfolio_limits(state, new_position, prices)
    
    assert not allowed
    assert 'concentration' in reason.lower()


def test_portfolio_risk_allowed():
    """Test that valid position is allowed."""
    config = {
        'max_portfolio_leverage': 3.0,
        'max_portfolio_drawdown': 0.2,
        'max_symbol_concentration': 0.5
    }
    manager = PortfolioRiskManager(config)
    
    state = PortfolioState(total_equity=10000.0)
    state.max_equity = 10000.0
    
    new_position = {'BTCUSDT': 0.02}  # 0.02 * 50000 = 1000 (10%)
    prices = {'BTCUSDT': 50000.0}
    
    allowed, reason = manager.check_portfolio_limits(state, new_position, prices)
    
    assert allowed
    assert reason == "OK"


def test_risk_budget_allocation():
    """Test risk budget allocation based on volatility."""
    config = {}
    manager = PortfolioRiskManager(config)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    # Set different volatilities
    manager.symbol_volatilities = {
        'BTCUSDT': 0.02,  # Low vol
        'ETHUSDT': 0.03,  # Medium vol
        'SOLUSDT': 0.06   # High vol
    }
    
    budgets = manager.allocate_risk_budget(symbols)
    
    # Should allocate more to low volatility
    assert budgets['BTCUSDT'] > budgets['SOLUSDT']
    
    # Should sum to 1
    assert abs(sum(budgets.values()) - 1.0) < 0.01


def test_update_symbol_stats():
    """Test updating symbol statistics."""
    config = {}
    manager = PortfolioRiskManager(config)
    
    # Add some returns
    for i in range(30):
        manager.update_symbol_stats('BTCUSDT', np.random.randn() * 0.02)
    
    assert 'BTCUSDT' in manager.symbol_volatilities
    assert manager.symbol_volatilities['BTCUSDT'] > 0


# ===== ADAPTIVE PARAMETER TUNER TESTS =====

def test_adaptive_tuner_init():
    """Test adaptive tuner initialization."""
    base_params = {
        'tau': {'H_star': 2.5},
        'risk': {'max_position_size': 10.0}
    }
    
    tuner = AdaptiveParameterTuner(base_params)
    
    assert tuner.base_params == base_params


def test_adaptive_tuner_regime_adaptation():
    """Test parameter adaptation based on regime."""
    base_params = {
        'tau': {'H_star': 2.5},
        'risk': {'max_position_size': 10.0, 'max_trade_risk_pct': 1.0}
    }
    
    tuner = AdaptiveParameterTuner(base_params)
    
    # Trending regime
    params_trend = tuner.get_adapted_params('TREND', 0.02)
    assert params_trend['tau']['H_star'] == 2.5  # No change in TREND
    
    # Volatile regime
    params_volx = tuner.get_adapted_params('VOLX', 0.02)
    assert params_volx['tau']['H_star'] < 2.5  # Reduced in VOLX


def test_adaptive_tuner_volatility_adaptation():
    """Test parameter adaptation based on volatility."""
    base_params = {
        'risk': {'max_position_size': 10.0}
    }
    
    tuner = AdaptiveParameterTuner(base_params)
    
    # Low volatility
    params_low = tuner.get_adapted_params('TREND', 0.01)
    
    # High volatility
    params_high = tuner.get_adapted_params('TREND', 0.05)
    
    # Should reduce position size in high volatility
    assert params_high['risk']['max_position_size'] < params_low['risk']['max_position_size']


def test_adaptive_tuner_performance_feedback():
    """Test parameter adaptation based on performance."""
    base_params = {
        'risk': {'max_trade_risk_pct': 2.0}
    }
    
    tuner = AdaptiveParameterTuner(base_params)
    
    # After losses
    params_loss = tuner.get_adapted_params('TREND', 0.02, recent_performance=-0.5)
    assert params_loss['risk']['max_trade_risk_pct'] < 2.0
    
    # After gains
    params_gain = tuner.get_adapted_params('TREND', 0.02, recent_performance=0.5)
    assert params_gain['risk']['max_trade_risk_pct'] == 2.0


def test_adaptive_tuner_performance_tracking():
    """Test performance tracking and best param set selection."""
    base_params = {}
    tuner = AdaptiveParameterTuner(base_params)
    
    # Record performance for different param sets
    for i in range(15):
        tuner.update_performance('params_a', 0.5 + np.random.randn() * 0.1)
        tuner.update_performance('params_b', 0.3 + np.random.randn() * 0.1)
    
    best = tuner.get_best_param_set()
    assert best == 'params_a'  # Should select higher performing set


# ===== MULTI-SYMBOL COORDINATOR TESTS =====

def test_coordinator_init():
    """Test multi-symbol coordinator initialization."""
    symbols = ['BTCUSDT', 'ETHUSDT']
    config = {'initial_equity': 50000.0}
    
    coordinator = MultiSymbolCoordinator(symbols, config)
    
    assert coordinator.symbols == symbols
    assert coordinator.portfolio.total_equity == 50000.0


def test_coordinator_market_state_update():
    """Test updating market state."""
    symbols = ['BTCUSDT']
    config = {}
    
    coordinator = MultiSymbolCoordinator(symbols, config)
    
    coordinator.update_market_state('BTCUSDT', 'TREND', 0.02, 0.01)
    
    assert coordinator.symbol_regimes['BTCUSDT'] == 'TREND'
    assert coordinator.symbol_volatilities['BTCUSDT'] == 0.02


def test_coordinator_symbol_params():
    """Test getting adapted parameters for symbol."""
    symbols = ['BTCUSDT']
    config = {
        'tau': {'H_star': 2.5},
        'risk': {'max_position_size': 10.0}
    }
    
    coordinator = MultiSymbolCoordinator(symbols, config)
    coordinator.update_market_state('BTCUSDT', 'VOLX', 0.05, 0.02)
    
    params = coordinator.get_symbol_params('BTCUSDT')
    
    assert 'tau' in params
    assert 'risk' in params


def test_coordinator_position_update():
    """Test updating positions after fills."""
    symbols = ['BTCUSDT', 'ETHUSDT']
    config = {'initial_equity': 10000.0}
    
    coordinator = MultiSymbolCoordinator(symbols, config)
    
    fills = {
        'BTCUSDT': {'qty': 0.1, 'pnl': 100.0},
        'ETHUSDT': {'qty': 1.0, 'pnl': 50.0}
    }
    
    coordinator.update_positions(fills)
    
    assert coordinator.portfolio.positions['BTCUSDT'] == 0.1
    assert coordinator.portfolio.positions['ETHUSDT'] == 1.0
    assert coordinator.portfolio.realized_pnl == 150.0


def test_coordinator_capital_allocation():
    """Test capital allocation across symbols."""
    symbols = ['BTCUSDT', 'ETHUSDT']
    config = {'initial_equity': 10000.0, 'portfolio': {}}
    
    coordinator = MultiSymbolCoordinator(symbols, config)
    
    # Set different volatilities
    coordinator.portfolio_risk.symbol_volatilities = {
        'BTCUSDT': 0.02,
        'ETHUSDT': 0.04
    }
    
    allocation = coordinator.allocate_capital()
    
    # Should allocate more to lower volatility symbol
    assert allocation['BTCUSDT'] > allocation['ETHUSDT']
    
    # Should sum to total equity
    assert abs(sum(allocation.values()) - 10000.0) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
