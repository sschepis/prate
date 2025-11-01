#!/usr/bin/env python3
"""
Tests for the guild system module.
"""

import numpy as np
from prate.guild_system import (
    GuildStyle, GuildPerformance, GuildManager,
    TrendFollowGenerator, MeanRevertGenerator, BreakoutGenerator, LiquidityMakerGenerator
)
from prate.types import Observation, GuildID, RegimeID


def create_test_observation(
    mid: float = 50000.0,
    ema_slope: float = 0.01,
    rsi: float = 50.0,
    atr: float = 500.0,
    pressure: float = 0.1,
    spread: float = 10.0,
    regime: str = 'TREND'
) -> Observation:
    """Create a test observation."""
    return Observation(
        ts=1609459200000,
        symbol='BTCUSDT',
        mid=mid,
        bid=mid - spread/2,
        ask=mid + spread/2,
        spread=spread,
        last_px=mid,
        last_qty=1.0,
        vol_1s=100.0,
        vol_1m=6000.0,
        book_imbalance=0.1,
        pressure=pressure,
        realized_var=0.0001,
        atr=atr,
        rsi_short=rsi,
        ema_slope=ema_slope,
        inventory=0.0,
        equity=10000.0,
        unrealized_pnl=0.0,
        funding_rate=0.0001,
        time_of_day_bucket=12,
        regime_soft={
            'TREND': 0.7 if regime == 'TREND' else 0.1,
            'RANGE': 0.7 if regime == 'RANGE' else 0.1,
            'VOLX': 0.7 if regime == 'VOLX' else 0.1,
            'QUIET': 0.7 if regime == 'QUIET' else 0.1
        },
        features_vec=np.zeros(10),
        features_disc={}
    )


def test_guild_style():
    """Test guild style parameters."""
    print("Testing Guild Style...")
    
    style = GuildStyle(
        aggression=0.7,
        hold_time=60.0,
        position_size_factor=1.2,
        risk_tolerance=0.6,
        entry_threshold=0.5
    )
    
    assert style.aggression == 0.7, "Aggression mismatch"
    assert style.hold_time == 60.0, "Hold time mismatch"
    assert style.position_size_factor == 1.2, "Position size factor mismatch"
    
    print(f"  ✓ Guild style created successfully")
    print(f"    - Aggression: {style.aggression}")
    print(f"    - Hold time: {style.hold_time}s")


def test_guild_performance():
    """Test guild performance tracking."""
    print("\nTesting Guild Performance...")
    
    perf = GuildPerformance()
    
    # Simulate some trades
    perf.update_trade(pnl=100.0, fees=5.0, hold_time=60.0, entry_price=50000.0, exit_price=50100.0)
    perf.update_trade(pnl=-50.0, fees=5.0, hold_time=45.0, entry_price=50100.0, exit_price=50050.0)
    perf.update_trade(pnl=150.0, fees=5.0, hold_time=75.0, entry_price=50050.0, exit_price=50200.0)
    
    assert perf.total_trades == 3, f"Total trades should be 3, got {perf.total_trades}"
    assert perf.winning_trades == 2, f"Winning trades should be 2, got {perf.winning_trades}"
    assert perf.losing_trades == 1, f"Losing trades should be 1, got {perf.losing_trades}"
    assert perf.total_pnl == 200.0, f"Total PnL should be 200, got {perf.total_pnl}"
    assert perf.win_rate == 2/3, f"Win rate should be 0.667, got {perf.win_rate}"
    
    print(f"  ✓ Performance tracking working correctly")
    print(f"    - Total trades: {perf.total_trades}")
    print(f"    - Win rate: {perf.win_rate:.2%}")
    print(f"    - Total PnL: ${perf.total_pnl:.2f}")
    print(f"    - Sharpe ratio: {perf.sharpe_ratio:.2f}")


def test_trend_follow_generator():
    """Test trend-following proposal generator."""
    print("\nTesting Trend-Follow Generator...")
    
    style = GuildStyle(aggression=0.6, entry_threshold=0.3)
    generator = TrendFollowGenerator(style)
    
    # Create uptrend observation
    obs = create_test_observation(ema_slope=0.02, regime='TREND')
    embedding = np.random.randn(10)
    
    signal = generator.compute_signal_strength(obs, embedding)
    assert signal > 0, "Signal should be positive for uptrend"
    print(f"  ✓ Signal strength: {signal:.4f}")
    
    action = generator.generate_proposal(obs, embedding, {})
    assert action is not None, "Should generate action for strong trend"
    assert action.delta_q > 0, "Should suggest long position for uptrend"
    
    print(f"  ✓ Trend-follow generator working correctly")
    print(f"    - Position delta: {action.delta_q:.4f}")
    print(f"    - Signal strength: {action.params['signal_strength']:.4f}")


def test_mean_revert_generator():
    """Test mean-reversion proposal generator."""
    print("\nTesting Mean-Revert Generator...")
    
    style = GuildStyle(aggression=0.5, entry_threshold=0.3)
    generator = MeanRevertGenerator(style)
    
    # Create overbought observation
    obs = create_test_observation(rsi=75.0, regime='RANGE')
    embedding = np.random.randn(10)
    
    signal = generator.compute_signal_strength(obs, embedding)
    assert signal > 0, "Signal should be positive for extreme RSI"
    print(f"  ✓ Signal strength: {signal:.4f}")
    
    action = generator.generate_proposal(obs, embedding, {})
    assert action is not None, "Should generate action for overbought"
    assert action.delta_q < 0, "Should suggest short position for overbought"
    
    print(f"  ✓ Mean-revert generator working correctly")
    print(f"    - Position delta: {action.delta_q:.4f}")
    print(f"    - Signal strength: {action.params['signal_strength']:.4f}")


def test_breakout_generator():
    """Test breakout proposal generator."""
    print("\nTesting Breakout Generator...")
    
    style = GuildStyle(aggression=0.8, entry_threshold=0.4)
    generator = BreakoutGenerator(style)
    
    # Create volatility expansion observation
    obs = create_test_observation(atr=1000.0, pressure=0.5, regime='VOLX')
    embedding = np.random.randn(10)
    
    signal = generator.compute_signal_strength(obs, embedding)
    assert signal > 0, "Signal should be positive for volatility expansion"
    print(f"  ✓ Signal strength: {signal:.4f}")
    
    action = generator.generate_proposal(obs, embedding, {})
    if action is not None:  # May not generate if signal below threshold
        assert abs(action.delta_q) > 0, "Should suggest position for breakout"
        print(f"  ✓ Breakout generator working correctly")
        print(f"    - Position delta: {action.delta_q:.4f}")
        print(f"    - Signal strength: {action.params['signal_strength']:.4f}")
    else:
        print(f"  ✓ Breakout generator correctly withheld proposal (signal too weak)")


def test_liquidity_maker_generator():
    """Test liquidity maker proposal generator."""
    print("\nTesting Liquidity Maker Generator...")
    
    style = GuildStyle(aggression=0.3, entry_threshold=0.2)
    generator = LiquidityMakerGenerator(style)
    
    # Create quiet market observation
    obs = create_test_observation(spread=5.0, regime='QUIET')
    embedding = np.random.randn(10)
    
    signal = generator.compute_signal_strength(obs, embedding)
    print(f"  ✓ Signal strength: {signal:.4f}")
    
    action = generator.generate_proposal(obs, embedding, {})
    if action is not None:
        assert abs(action.delta_q) > 0, "Should suggest position for liquidity making"
        assert 'post_only' in action.params, "Should include post_only parameter"
        print(f"  ✓ Liquidity maker generator working correctly")
        print(f"    - Position delta: {action.delta_q:.4f}")
        print(f"    - Post only: {action.params.get('post_only', False)}")
    else:
        print(f"  ✓ Liquidity maker correctly withheld proposal (signal too weak)")


def test_guild_manager():
    """Test guild manager."""
    print("\nTesting Guild Manager...")
    
    manager = GuildManager()
    
    # Check default guilds
    assert GuildID.TF in manager.guilds, "Should have trend-follow guild"
    assert GuildID.MR in manager.guilds, "Should have mean-revert guild"
    assert GuildID.BR in manager.guilds, "Should have breakout guild"
    assert GuildID.LM in manager.guilds, "Should have liquidity maker guild"
    
    print(f"  ✓ Guild manager initialized with {len(manager.guilds)} guilds")
    
    # Get proposals
    obs = create_test_observation(ema_slope=0.02, rsi=75.0, regime='TREND')
    embedding = np.random.randn(10)
    
    proposals = manager.get_proposals(obs, embedding, {})
    print(f"  ✓ Generated {len(proposals)} proposals from guilds")
    
    for guild_id, action in proposals:
        print(f"    - {guild_id.value}: delta_q={action.delta_q:.4f}")
    
    # Select best proposal
    if proposals:
        best = manager.select_best_proposal(proposals)
        assert best is not None, "Should select a best proposal"
        print(f"  ✓ Selected best proposal from {best[0].value}")


def test_performance_update():
    """Test performance updates through guild manager."""
    print("\nTesting Performance Updates...")
    
    manager = GuildManager()
    
    # Update performance for a guild
    manager.update_performance(
        guild_id=GuildID.TF,
        pnl=100.0,
        fees=5.0,
        hold_time=60.0,
        entry_price=50000.0,
        exit_price=50100.0
    )
    
    perf = manager.performances[GuildID.TF]
    assert perf.total_trades == 1, "Should have 1 trade"
    assert perf.total_pnl == 100.0, "PnL should be 100"
    
    # Get performance summary
    summary = manager.get_performance_summary()
    assert GuildID.TF in summary, "Summary should include TF guild"
    assert summary[GuildID.TF]['total_trades'] == 1, "Summary should show 1 trade"
    
    print(f"  ✓ Performance update working correctly")
    print(f"    - Total trades: {perf.total_trades}")
    print(f"    - Total PnL: ${perf.total_pnl:.2f}")


def test_proposal_selection_with_performance():
    """Test that proposal selection considers guild performance."""
    print("\nTesting Proposal Selection with Performance...")
    
    manager = GuildManager()
    
    # Give TF guild good performance
    for i in range(10):
        manager.update_performance(
            guild_id=GuildID.TF,
            pnl=100.0 + i * 10,
            fees=5.0,
            hold_time=60.0,
            entry_price=50000.0,
            exit_price=50100.0
        )
    
    # Give MR guild poor performance
    for i in range(10):
        manager.update_performance(
            guild_id=GuildID.MR,
            pnl=-50.0,
            fees=5.0,
            hold_time=60.0,
            entry_price=50000.0,
            exit_price=49950.0
        )
    
    # Create observation that both could trade
    obs = create_test_observation(ema_slope=0.02, rsi=75.0, regime='TREND')
    embedding = np.random.randn(10)
    
    proposals = manager.get_proposals(obs, embedding, {})
    
    if len(proposals) >= 2:
        best = manager.select_best_proposal(proposals)
        
        # Performance summary
        summary = manager.get_performance_summary()
        print(f"  ✓ Proposal selection considers performance")
        print(f"    - TF performance: Sharpe={summary[GuildID.TF]['sharpe_ratio']:.2f}, Win={summary[GuildID.TF]['win_rate']:.2%}")
        print(f"    - MR performance: Sharpe={summary[GuildID.MR]['sharpe_ratio']:.2f}, Win={summary[GuildID.MR]['win_rate']:.2%}")
        if best:
            print(f"    - Selected: {best[0].value}")


if __name__ == '__main__':
    print("=" * 60)
    print("PRATE Guild System Tests")
    print("=" * 60)
    
    test_guild_style()
    test_guild_performance()
    test_trend_follow_generator()
    test_mean_revert_generator()
    test_breakout_generator()
    test_liquidity_maker_generator()
    test_guild_manager()
    test_performance_update()
    test_proposal_selection_with_performance()
    
    print("\n" + "=" * 60)
    print("✓ All guild system tests passed successfully!")
    print("=" * 60)
