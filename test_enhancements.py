#!/usr/bin/env python3
"""
Tests for the enhanced features and simulator improvements.
"""

import numpy as np
from prate.features import (
    compute_ema, compute_rsi, compute_atr, compute_bollinger_bands,
    compute_realized_variance, compute_pressure, compute_features,
    regime_scores, RollingBuffers
)
from prate.simulator import Simulator, OrderBook, FillModel, LatencyModel
from prate.types import TradeIntent, Side, RegimeID


def test_technical_indicators():
    """Test technical indicator calculations."""
    print("Testing Technical Indicators...")
    
    # Test EMA
    prices = [100, 102, 101, 103, 105, 104, 106]
    ema = compute_ema(prices, period=5)
    assert ema > 0, "EMA should be positive"
    assert 100 <= ema <= 110, f"EMA should be in reasonable range, got {ema}"
    print(f"  ✓ EMA: {ema:.2f}")
    
    # Test RSI
    prices = list(range(100, 120))  # Uptrend
    rsi = compute_rsi(prices, period=14)
    assert 0 <= rsi <= 100, f"RSI should be 0-100, got {rsi}"
    assert rsi > 50, "RSI should be > 50 for uptrend"
    print(f"  ✓ RSI: {rsi:.2f}")
    
    # Test ATR
    highs = [105, 107, 106, 108, 110]
    lows = [100, 101, 102, 103, 104]
    closes = [103, 104, 105, 106, 107]
    atr = compute_atr(highs, lows, closes, period=3)
    assert atr > 0, "ATR should be positive"
    print(f"  ✓ ATR: {atr:.2f}")
    
    # Test Bollinger Bands
    prices = [100 + np.sin(i/5) * 5 for i in range(30)]
    bb = compute_bollinger_bands(prices, period=20)
    assert bb['bb_upper'] > bb['bb_middle'], "Upper band should be > middle"
    assert bb['bb_middle'] > bb['bb_lower'], "Middle should be > lower band"
    assert bb['bb_width'] >= 0, "BB width should be non-negative"
    print(f"  ✓ Bollinger Bands: middle={bb['bb_middle']:.2f}, width={bb['bb_width']:.4f}")


def test_microstructure_features():
    """Test microstructure feature calculations."""
    print("\nTesting Microstructure Features...")
    
    # Test realized variance
    returns = [0.001, -0.002, 0.0015, -0.001, 0.002]
    rv = compute_realized_variance(returns, period=5)
    assert rv >= 0, "Realized variance should be non-negative"
    print(f"  ✓ Realized Variance: {rv:.6f}")
    
    # Test pressure
    buys = [100, 150, 120, 180, 200]
    sells = [120, 100, 130, 90, 110]
    pressure = compute_pressure(buys, sells, period=5)
    assert -1 <= pressure <= 1, f"Pressure should be -1 to 1, got {pressure}"
    print(f"  ✓ Pressure: {pressure:.3f}")


def test_regime_classification():
    """Test regime classification logic."""
    print("\nTesting Regime Classification...")
    
    # Test trending regime
    features_trend = {
        'rsi': 65.0,
        'ema_slope': 0.02,
        'bb_width': 0.04,
        'realized_var': 0.001,
        'atr': 100.0
    }
    scores = regime_scores(features_trend)
    assert abs(sum(scores.values()) - 1.0) < 0.01, "Scores should sum to 1"
    assert scores[RegimeID.TREND] > 0.2, "Should detect trend regime"
    print(f"  ✓ Trend regime detected: {scores[RegimeID.TREND]:.3f}")
    
    # Test ranging regime
    features_range = {
        'rsi': 50.0,
        'ema_slope': 0.002,
        'bb_width': 0.015,
        'realized_var': 0.0001,
        'atr': 50.0
    }
    scores = regime_scores(features_range)
    assert scores[RegimeID.RANGE] > 0.2, "Should detect range regime"
    print(f"  ✓ Range regime detected: {scores[RegimeID.RANGE]:.3f}")
    
    # Test volatility expansion
    features_volx = {
        'rsi': 55.0,
        'ema_slope': 0.005,
        'bb_width': 0.05,
        'realized_var': 0.002,
        'atr': 150.0
    }
    scores = regime_scores(features_volx)
    assert scores[RegimeID.VOLX] > 0.1, "Should detect volatility expansion"
    print(f"  ✓ VolX regime detected: {scores[RegimeID.VOLX]:.3f}")


def test_feature_engine():
    """Test complete feature engine."""
    print("\nTesting Complete Feature Engine...")
    
    buffers = RollingBuffers({
        'price': 100,
        'high': 100,
        'low': 100,
        'returns': 100,
        'buy_volume': 50,
        'sell_volume': 50,
        'bid_volume': 50,
        'ask_volume': 50
    })
    
    # Add some data
    for i in range(50):
        price = 50000 + i * 10 + np.sin(i/5) * 50
        buffers.update({
            'price': price,
            'high': price + 5,
            'low': price - 5,
            'returns': 0.0001 * np.sin(i/10),
            'buy_volume': 100 + i,
            'sell_volume': 90 + i,
            'bid_volume': 1000,
            'ask_volume': 950
        })
    
    features = compute_features(buffers)
    
    # Check all expected features are present
    expected_features = [
        'ret_1s', 'ema_fast', 'ema_slow', 'ema_slope', 'rsi',
        'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'bb_position', 'realized_var', 'pressure', 'book_imb'
    ]
    
    for feat in expected_features:
        assert feat in features, f"Missing feature: {feat}"
        assert not np.isnan(features[feat]), f"Feature {feat} is NaN"
    
    print(f"  ✓ All {len(expected_features)} features computed successfully")
    print(f"    - RSI: {features['rsi']:.2f}")
    print(f"    - EMA Slope: {features['ema_slope']:.4f}")
    print(f"    - BB Width: {features['bb_width']:.4f}")
    print(f"    - Pressure: {features['pressure']:.3f}")


def test_order_book():
    """Test order book simulation."""
    print("\nTesting Order Book Simulation...")
    
    book = OrderBook(initial_price=50000.0, depth_levels=10)
    
    best_bid = book.get_best_bid()
    best_ask = book.get_best_ask()
    
    assert best_bid < book.mid_price, "Best bid should be < mid price"
    assert best_ask > book.mid_price, "Best ask should be > mid price"
    assert best_bid < best_ask, "Bid should be < ask"
    
    print(f"  ✓ Order book created: bid={best_bid:.2f}, ask={best_ask:.2f}, spread={best_ask-best_bid:.2f}")
    
    # Test update
    book.update(51000.0)
    new_bid = book.get_best_bid()
    assert new_bid > best_bid, "Bid should increase with mid price"
    print(f"  ✓ Order book updated successfully")


def test_fill_model():
    """Test fill probability model."""
    print("\nTesting Fill Model...")
    
    fill_model = FillModel({
        'limit_fill_prob': 0.8,
        'market_fill_prob': 0.95,
        'partial_fill_prob': 0.3
    })
    
    # Test market order
    market_intent = TradeIntent(
        client_id="test_market",
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=1.0,
        price=None,
        tif="GTC",
        post_only=False,
        meta={}
    )
    
    # Run multiple times to check probability
    fills = sum(fill_model.should_fill(market_intent, 50000.0) for _ in range(100))
    assert fills > 80, f"Market orders should fill frequently, got {fills}/100"
    print(f"  ✓ Market order fill rate: {fills}%")
    
    # Test limit order
    limit_intent = TradeIntent(
        client_id="test_limit",
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=1.0,
        price=49000.0,
        tif="GTC",
        post_only=True,
        meta={}
    )
    
    # Should fill when price is touched
    should_fill = fill_model.should_fill(limit_intent, 48900.0)
    print(f"  ✓ Limit order tested (price touched)")


def test_latency_model():
    """Test latency injection."""
    print("\nTesting Latency Model...")
    
    latency_model = LatencyModel({
        'mean_latency_ms': 50,
        'std_latency_ms': 20,
        'min_latency_ms': 10
    })
    
    latencies = [latency_model.get_latency() for _ in range(100)]
    avg_latency = np.mean(latencies)
    min_latency = min(latencies)
    
    assert min_latency >= 10, f"Latency should be >= 10ms, got {min_latency}"
    assert 30 <= avg_latency <= 70, f"Avg latency should be ~50ms, got {avg_latency}"
    print(f"  ✓ Latency model: avg={avg_latency:.1f}ms, min={min_latency}ms, max={max(latencies)}ms")


def test_simulator_enhancements():
    """Test enhanced simulator features."""
    print("\nTesting Enhanced Simulator...")
    
    simulator = Simulator(
        market_data=None,
        fee_schedule={'maker': 0.0002, 'taker': 0.0005},
        slippage_model={'market_bps': 3.0},
        fill_model_config={'limit_fill_prob': 0.9, 'market_fill_prob': 0.99},
        latency_config={'mean_latency_ms': 50},
        funding_config={'base_rate': 0.0001, 'interval_hours': 8}
    )
    
    # Test sending order with latency
    intent = TradeIntent(
        client_id="test_001",
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=0.1,
        price=None,  # Market order
        tif="GTC",
        post_only=False,
        meta={}
    )
    
    cid = simulator.send(intent)
    assert cid == "test_001", "Should return correct client ID"
    print(f"  ✓ Order sent with latency injection")
    
    # Process simulation steps
    for i in range(5):
        simulator.step(1000 * i)
    
    # Check fills
    fills = simulator.poll()
    if fills:
        print(f"  ✓ Received {len(fills)} fill(s)")
        for fill in fills:
            assert 'is_partial' in fill, "Fill should indicate if partial"
            print(f"    - Fill: {fill['qty']:.3f} @ {fill['price']:.2f}, partial={fill['is_partial']}")
    
    # Check account state
    state = simulator.get_account_state()
    assert 'unrealized_pnl' in state, "Should track unrealized PnL"
    assert 'funding_pnl' in state, "Should track funding PnL"
    print(f"  ✓ Account state tracked: equity=${state['equity']:.2f}, position={state['position']:.3f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PRATE Enhancement Tests")
    print("=" * 60)
    
    try:
        test_technical_indicators()
        test_microstructure_features()
        test_regime_classification()
        test_feature_engine()
        test_order_book()
        test_fill_model()
        test_latency_model()
        test_simulator_enhancements()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
