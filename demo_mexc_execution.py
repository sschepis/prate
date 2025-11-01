#!/usr/bin/env python3
"""
Example demonstration of PRATE MEXC Futures execution interface.

This script demonstrates:
1. Candle aggregation (1m -> 1s -> custom timeframes)
2. MEXC authentication setup
3. Order creation (simulated - no real execution)
4. Position and balance queries (simulated)

NOTE: This example does NOT execute real trades. It demonstrates the API structure.
"""

import os
import tempfile
from prate import (
    CandleAggregator,
    CandleDatabase,
    Side,
    OrderType,
    MarginMode
)
from prate.mexc_futures import MEXCFuturesAuth


def demonstrate_candle_aggregation():
    """Demonstrate candle aggregation capabilities."""
    print("=" * 70)
    print("CANDLE AGGREGATION DEMONSTRATION")
    print("=" * 70)
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.db')
    db = CandleDatabase(db_path)
    aggregator = CandleAggregator(db)
    
    # Sample 1-minute MEXC candles (OHLCV format)
    candles_1m = [
        {
            'timestamp': 1609459200000,  # 2021-01-01 00:00:00
            'open': 29000.0,
            'high': 29150.0,
            'low': 28900.0,
            'close': 29050.0,
            'volume': 120.5
        },
        {
            'timestamp': 1609459260000,  # 2021-01-01 00:01:00
            'open': 29050.0,
            'high': 29200.0,
            'low': 29000.0,
            'close': 29150.0,
            'volume': 135.8
        },
        {
            'timestamp': 1609459320000,  # 2021-01-01 00:02:00
            'open': 29150.0,
            'high': 29300.0,
            'low': 29100.0,
            'close': 29250.0,
            'volume': 142.3
        }
    ]
    
    print(f"\n1. Converting {len(candles_1m)} x 1-minute candles to 1-second candles...")
    count = aggregator.process_and_store(candles_1m, 'BTC_USDT')
    print(f"   ✓ Stored {count} 1-second candles in database")
    
    print("\n2. Retrieving and aggregating to different timeframes:")
    
    # Get 5-second candles
    start_ts = candles_1m[0]['timestamp']
    end_ts = candles_1m[-1]['timestamp'] + 60000
    
    candles_5s = aggregator.get_aggregated_candles(
        'BTC_USDT', start_ts, end_ts, interval_seconds=5
    )
    print(f"   - 5-second timeframe: {len(candles_5s)} candles")
    
    # Get 10-second candles
    candles_10s = aggregator.get_aggregated_candles(
        'BTC_USDT', start_ts, end_ts, interval_seconds=10
    )
    print(f"   - 10-second timeframe: {len(candles_10s)} candles")
    
    # Get 30-second candles
    candles_30s = aggregator.get_aggregated_candles(
        'BTC_USDT', start_ts, end_ts, interval_seconds=30
    )
    print(f"   - 30-second timeframe: {len(candles_30s)} candles")
    
    # Show sample 5s candle
    if candles_5s:
        sample = candles_5s[0]
        print(f"\n3. Sample 5-second candle:")
        print(f"   - Open:   {sample.open:.2f}")
        print(f"   - High:   {sample.high:.2f}")
        print(f"   - Low:    {sample.low:.2f}")
        print(f"   - Close:  {sample.close:.2f}")
        print(f"   - Volume: {sample.volume:.2f}")
    
    # Cleanup
    db.close()
    os.unlink(db_path)
    
    print(f"\n✓ Candle aggregation demonstration complete")


def demonstrate_mexc_authentication():
    """Demonstrate MEXC authentication (no real connection)."""
    print("\n" + "=" * 70)
    print("MEXC AUTHENTICATION DEMONSTRATION")
    print("=" * 70)
    
    # Use dummy credentials for demonstration
    api_key = "demo_api_key_12345"
    secret_key = "demo_secret_key_67890"
    
    print(f"\n1. Creating MEXC authentication helper...")
    auth = MEXCFuturesAuth(api_key, secret_key)
    print(f"   ✓ Auth helper created")
    
    print(f"\n2. Generating signature for timestamp 1609459200000:")
    timestamp = 1609459200000
    signature = auth.generate_signature(timestamp, "")
    print(f"   - Signature: {signature[:32]}... ({len(signature)} chars)")
    
    print(f"\n3. Generating authenticated headers:")
    headers = auth.get_headers(timestamp, "")
    print(f"   - ApiKey: {headers['ApiKey']}")
    print(f"   - Request-Time: {headers['Request-Time']}")
    print(f"   - Signature: {headers['Signature'][:32]}...")
    print(f"   - Content-Type: {headers['Content-Type']}")
    
    print(f"\n✓ Authentication demonstration complete")


def demonstrate_order_structure():
    """Demonstrate order creation structure (no real execution)."""
    print("\n" + "=" * 70)
    print("ORDER STRUCTURE DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. Market Order Structure (BUY):")
    print("""
    order = exchange.create_order(
        symbol="BTC_USDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=0.001,
        price=50000.0,      # Current market price (required by MEXC)
        leverage=10,
        margin_mode=MarginMode.ISOLATED
    )
    """)
    
    print("\n2. Limit Order Structure (SELL):")
    print("""
    order = exchange.create_order(
        symbol="ETH_USDT",
        side=Side.SELL,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=3500.0,       # Limit price
        leverage=5,
        margin_mode=MarginMode.CROSS,
        client_order_id="my-order-001"
    )
    """)
    
    print("\n3. Querying Positions:")
    print("""
    positions = exchange.get_positions()
    for pos in positions:
        print(f"Symbol: {pos.symbol}")
        print(f"Size: {pos.quantity}")
        print(f"Entry: {pos.entry_price}")
        print(f"PnL: {pos.unrealized_pnl}")
    """)
    
    print("\n4. Checking Balance:")
    print("""
    balance = exchange.get_balance()
    print(f"Available: {balance.available} USDT")
    print(f"Total: {balance.total} USDT")
    """)
    
    print("\n✓ Order structure demonstration complete")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PRATE MEXC FUTURES EXECUTION INTERFACE - DEMO")
    print("=" * 70)
    print("\nThis demonstration shows the capabilities WITHOUT executing real trades.")
    print("\nComponents demonstrated:")
    print("  - Candle aggregation (1m -> 1s -> custom timeframes)")
    print("  - MEXC authentication and signature generation")
    print("  - Order structure and API usage patterns")
    
    demonstrate_candle_aggregation()
    demonstrate_mexc_authentication()
    demonstrate_order_structure()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nTo use with real trading:")
    print("  1. Set environment variables: MEXC_API_KEY and MEXC_SECRET_KEY")
    print("  2. Initialize: exchange = MEXCFuturesExecution(api_key, secret_key)")
    print("  3. Connect: exchange.connect()")
    print("  4. Trade responsibly with proper risk management")
    print("\n⚠️  WARNING: MEXC endpoints are unofficial - implement kill switch!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
