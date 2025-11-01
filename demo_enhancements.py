#!/usr/bin/env python3
"""
Demonstration of the enhanced features and simulator improvements.

This script showcases:
1. Real technical indicators (EMA, RSI, ATR, Bollinger Bands)
2. Microstructure features (pressure, realized variance)
3. Regime classification
4. Realistic order book simulation
5. Fill probability models
6. Latency injection
7. Partial fills
8. Funding rate simulation
"""

import numpy as np
from prate.features import FeatureEngine
from prate.simulator import Simulator
from prate.types import TradeIntent, Side, RegimeID


def generate_sample_price_data(n_points=100, initial_price=50000.0, volatility=0.02):
    """Generate sample price data with trend and noise."""
    prices = [initial_price]
    for i in range(n_points - 1):
        # Add trend + noise
        trend = 0.0001 * i
        noise = np.random.normal(0, volatility * prices[-1])
        new_price = prices[-1] * (1 + trend) + noise
        prices.append(new_price)
    return prices


def demonstrate_feature_engine():
    """Demonstrate the enhanced feature engine."""
    print("=" * 70)
    print("FEATURE ENGINE DEMONSTRATION")
    print("=" * 70)
    
    # Create feature engine
    config = {
        'windows': {
            'price': 100,
            'high': 100,
            'low': 100,
            'returns': 100,
            'buy_volume': 100,
            'sell_volume': 100,
            'bid_volume': 100,
            'ask_volume': 100,
            'volume': 100,
            'bid': 100,
            'ask': 100
        },
        'binning_specs': {
            'ret': {'bins': 20, 'min': -0.01, 'max': 0.01},
            'imb': {'bins': 20, 'min': -1.0, 'max': 1.0},
            'regime': {'bins': 5, 'min': 0, 'max': 5},
            'tod': {'bins': 24, 'min': 0, 'max': 1}
        }
    }
    
    engine = FeatureEngine(config)
    
    # Generate and feed sample data
    prices = generate_sample_price_data(100, 50000.0, 0.005)
    
    print("\nFeeding 100 data points into feature engine...")
    for i, price in enumerate(prices):
        high = price * 1.001
        low = price * 0.999
        ret = (price - prices[i-1]) / prices[i-1] if i > 0 else 0
        
        engine.update({
            'price': price,
            'high': high,
            'low': low,
            'bid': price * 0.9995,
            'ask': price * 1.0005,
            'returns': ret,
            'buy_volume': 100 + np.random.randn() * 20,
            'sell_volume': 95 + np.random.randn() * 20,
            'bid_volume': 1000 + np.random.randn() * 100,
            'ask_volume': 950 + np.random.randn() * 100,
            'volume': 50 + np.random.randn() * 10
        })
    
    # Create snapshot
    ts = 1700000000000
    obs = engine.snapshot(ts, "BTCUSDT")
    
    print("\n" + "-" * 70)
    print("COMPUTED FEATURES:")
    print("-" * 70)
    print(f"Price: ${obs.mid:,.2f}")
    print(f"Spread: ${obs.spread:.2f}")
    print(f"\nTechnical Indicators:")
    print(f"  RSI: {obs.rsi_short:.2f}")
    print(f"  EMA Slope: {obs.ema_slope:.6f}")
    print(f"  ATR: ${obs.atr:.2f}")
    print(f"\nMicrostructure:")
    print(f"  Book Imbalance: {obs.book_imbalance:.4f}")
    print(f"  Pressure: {obs.pressure:.4f}")
    print(f"  Realized Variance: {obs.realized_var:.8f}")
    print(f"\nRegime Classification:")
    for regime, score in obs.regime_soft.items():
        print(f"  {regime.value:8s}: {score:.3f}")
    
    # Determine dominant regime
    dominant_regime = max(obs.regime_soft, key=obs.regime_soft.get)
    print(f"\nDominant Regime: {dominant_regime.value}")


def demonstrate_simulator():
    """Demonstrate the enhanced simulator."""
    print("\n\n" + "=" * 70)
    print("SIMULATOR DEMONSTRATION")
    print("=" * 70)
    
    # Create simulator with all enhancements
    simulator = Simulator(
        market_data=None,
        fee_schedule={'maker': 0.0002, 'taker': 0.0005},
        slippage_model={'market_bps': 3.0},
        fill_model_config={
            'limit_fill_prob': 0.7,
            'market_fill_prob': 0.95,
            'partial_fill_prob': 0.2,
            'min_fill_ratio': 0.6
        },
        latency_config={
            'mean_latency_ms': 50,
            'std_latency_ms': 20,
            'min_latency_ms': 10
        },
        funding_config={
            'base_rate': 0.0001,
            'interval_hours': 8
        }
    )
    
    print("\n✓ Simulator initialized with:")
    print("  - Realistic fill probability models")
    print("  - Order book depth simulation")
    print("  - Latency injection (mean=50ms, std=20ms)")
    print("  - Funding rate simulation (0.01% every 8 hours)")
    print("  - Partial fill support")
    
    # Submit various orders
    print("\n" + "-" * 70)
    print("SUBMITTING ORDERS:")
    print("-" * 70)
    
    orders = [
        TradeIntent(
            client_id="order_001",
            symbol="BTCUSDT",
            side=Side.BUY,
            qty=0.5,
            price=None,  # Market order
            tif="GTC",
            post_only=False,
            meta={'type': 'market'}
        ),
        TradeIntent(
            client_id="order_002",
            symbol="BTCUSDT",
            side=Side.BUY,
            qty=0.3,
            price=49500.0,  # Limit order
            tif="GTC",
            post_only=True,
            meta={'type': 'limit'}
        ),
        TradeIntent(
            client_id="order_003",
            symbol="BTCUSDT",
            side=Side.SELL,
            qty=0.2,
            price=50500.0,  # Limit order
            tif="GTC",
            post_only=True,
            meta={'type': 'limit'}
        )
    ]
    
    for order in orders:
        cid = simulator.send(order)
        order_type = order.meta.get('type', 'unknown')
        price_str = f"@ ${order.price:,.2f}" if order.price else "MARKET"
        print(f"  [{cid}] {order.side.value} {order.qty} {price_str} ({order_type})")
    
    # Simulate time progression
    print("\n" + "-" * 70)
    print("SIMULATION PROGRESS:")
    print("-" * 70)
    
    for step in range(10):
        ts = 1000 * step
        simulator.step(ts)
        
        # Check for fills
        fills = simulator.poll()
        if fills:
            for fill in fills:
                partial_str = " (PARTIAL)" if fill['is_partial'] else ""
                print(f"  [Step {step}] FILL: {fill['side']} {fill['qty']:.3f} @ ${fill['price']:.2f}")
                print(f"            Fee: ${fill['fee']:.4f}, PnL: ${fill['pnl']:.2f}{partial_str}")
    
    # Show final account state
    print("\n" + "-" * 70)
    print("FINAL ACCOUNT STATE:")
    print("-" * 70)
    
    state = simulator.get_account_state()
    print(f"  Equity: ${state['equity']:,.2f}")
    print(f"  Position: {state['position']:.4f} BTC")
    print(f"  Realized PnL: ${state['pnl']:,.2f}")
    print(f"  Unrealized PnL: ${state['unrealized_pnl']:,.2f}")
    print(f"  Funding PnL: ${state['funding_pnl']:,.2f}")
    print(f"  Avg Entry Price: ${state['avg_entry_price']:,.2f}")
    print(f"  Current Price: ${state['price']:,.2f}")
    
    # Calculate total PnL
    total_pnl = state['pnl'] + state['unrealized_pnl']
    print(f"\n  Total PnL: ${total_pnl:,.2f}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "PRATE ENHANCED FEATURES DEMONSTRATION" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    
    demonstrate_feature_engine()
    demonstrate_simulator()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\n✓ All enhanced features demonstrated successfully!")
    print("\nKey Improvements:")
    print("  1. Real technical indicators (EMA, RSI, ATR, Bollinger Bands)")
    print("  2. Microstructure features (pressure, realized variance, book imbalance)")
    print("  3. Heuristic-based regime classification")
    print("  4. Realistic order book simulation with depth")
    print("  5. Probabilistic fill models with partial fill support")
    print("  6. Network latency injection")
    print("  7. Perpetual futures funding rate simulation")
    print("  8. Improved PnL tracking (realized + unrealized)")
    print()


if __name__ == '__main__':
    main()
