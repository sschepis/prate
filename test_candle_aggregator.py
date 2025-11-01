#!/usr/bin/env python3
"""
Tests for the candle aggregator module.
"""

import os
import tempfile
import numpy as np
from prate.candle_aggregator import CandleAggregator, CandleDatabase, Candle1s


def test_convert_1m_to_1s():
    """Test converting 1-minute candles to 1-second candles."""
    print("Testing 1m to 1s Conversion...")
    
    aggregator = CandleAggregator()
    
    # Create test 1m candles
    candles_1m = [
        {
            'timestamp': 1609459200000,  # 2021-01-01 00:00:00
            'open': 29000.0,
            'high': 29100.0,
            'low': 28900.0,
            'close': 29050.0,
            'volume': 120.0
        },
        {
            'timestamp': 1609459260000,  # 2021-01-01 00:01:00
            'open': 29050.0,
            'high': 29200.0,
            'low': 29000.0,
            'close': 29150.0,
            'volume': 150.0
        }
    ]
    
    candles_1s = aggregator.convert_1m_to_1s(candles_1m, 'BTCUSDT')
    
    # Should produce 60 candles per 1m candle
    assert len(candles_1s) == 120, f"Should produce 120 1s candles, got {len(candles_1s)}"
    
    # Check first candle
    first = candles_1s[0]
    assert first.timestamp == 1609459200000, "First candle timestamp mismatch"
    assert first.symbol == 'BTCUSDT', "Symbol mismatch"
    assert first.source == 'interpolated', "Source should be interpolated"
    
    # Check volume distribution (should be volume/60)
    expected_volume = 120.0 / 60.0
    assert abs(first.volume - expected_volume) < 0.01, \
        f"Volume per second should be ~{expected_volume}, got {first.volume}"
    
    # Check that prices are within bounds
    for i in range(60):
        candle = candles_1s[i]
        assert candle.low <= candle.high, "Low should be <= high"
        assert candle.low <= candle.open <= candle.high, "Open should be within [low, high]"
        assert candle.low <= candle.close <= candle.high, "Close should be within [low, high]"
    
    print(f"  ✓ Converted {len(candles_1m)} 1m candles to {len(candles_1s)} 1s candles")
    print(f"    - First 1s candle: open={first.open:.2f}, close={first.close:.2f}, vol={first.volume:.2f}")


def test_aggregate_1s_to_timeframe():
    """Test aggregating 1s candles to different timeframes."""
    print("\nTesting 1s to Nx Aggregation...")
    
    aggregator = CandleAggregator()
    
    # Create test 1s candles (10 seconds worth)
    candles_1s = []
    base_ts = 1609459200000
    for i in range(10):
        candle = Candle1s(
            timestamp=base_ts + i * 1000,
            symbol='BTCUSDT',
            open=29000.0 + i * 10,
            high=29000.0 + i * 10 + 20,
            low=29000.0 + i * 10 - 10,
            close=29000.0 + i * 10 + 5,
            volume=2.0,
            trades=5,
            source='interpolated'
        )
        candles_1s.append(candle)
    
    # Aggregate to 2s
    candles_2s = aggregator.aggregate_1s_to_timeframe(candles_1s, 2)
    assert len(candles_2s) == 5, f"Should produce 5 2s candles, got {len(candles_2s)}"
    
    # Check first 2s candle
    first_2s = candles_2s[0]
    assert first_2s.open == candles_1s[0].open, "2s open should match first 1s open"
    assert first_2s.close == candles_1s[1].close, "2s close should match second 1s close"
    assert first_2s.volume == 4.0, f"2s volume should be sum of two 1s (4.0), got {first_2s.volume}"
    assert first_2s.trades == 10, f"2s trades should be sum (10), got {first_2s.trades}"
    
    # Aggregate to 5s
    candles_5s = aggregator.aggregate_1s_to_timeframe(candles_1s, 5)
    assert len(candles_5s) == 2, f"Should produce 2 5s candles, got {len(candles_5s)}"
    
    first_5s = candles_5s[0]
    assert first_5s.volume == 10.0, f"5s volume should be sum of five 1s (10.0), got {first_5s.volume}"
    
    print(f"  ✓ Aggregation working correctly")
    print(f"    - 10 x 1s -> {len(candles_2s)} x 2s candles")
    print(f"    - 10 x 1s -> {len(candles_5s)} x 5s candles")
    print(f"    - 2s candle volume: {first_2s.volume:.1f}")
    print(f"    - 5s candle volume: {first_5s.volume:.1f}")


def test_database_storage():
    """Test database storage and retrieval."""
    print("\nTesting Database Storage...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        db = CandleDatabase(db_path)
        
        # Create test candles
        candles = []
        base_ts = 1609459200000
        for i in range(100):
            candle = Candle1s(
                timestamp=base_ts + i * 1000,
                symbol='BTCUSDT',
                open=29000.0 + i,
                high=29000.0 + i + 10,
                low=29000.0 + i - 10,
                close=29000.0 + i + 5,
                volume=1.0,
                trades=3,
                source='interpolated'
            )
            candles.append(candle)
        
        # Insert candles
        count = db.insert_candles(candles)
        assert count == 100, f"Should insert 100 candles, inserted {count}"
        
        # Retrieve candles
        retrieved = db.get_candles('BTCUSDT', base_ts, base_ts + 100 * 1000)
        assert len(retrieved) == 100, f"Should retrieve 100 candles, got {len(retrieved)}"
        
        # Check data integrity
        assert retrieved[0].timestamp == candles[0].timestamp, "Timestamp mismatch"
        assert retrieved[0].open == candles[0].open, "Open price mismatch"
        assert retrieved[0].volume == candles[0].volume, "Volume mismatch"
        
        # Test symbols
        symbols = db.get_symbols()
        assert 'BTCUSDT' in symbols, "BTCUSDT should be in symbols list"
        
        # Test time range
        time_range = db.get_time_range('BTCUSDT')
        assert time_range is not None, "Should have time range"
        assert time_range[0] == base_ts, "Start time mismatch"
        assert time_range[1] == base_ts + 99 * 1000, "End time mismatch"
        
        db.close()
        
        print(f"  ✓ Database storage working correctly")
        print(f"    - Inserted: {count} candles")
        print(f"    - Retrieved: {len(retrieved)} candles")
        print(f"    - Time range: {time_range[0]} - {time_range[1]}")
    
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_process_and_store():
    """Test end-to-end processing and storage."""
    print("\nTesting Process and Store...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        db = CandleDatabase(db_path)
        aggregator = CandleAggregator(db)
        
        # Create 1m candles
        candles_1m = [
            {
                'timestamp': 1609459200000,
                'open': 29000.0,
                'high': 29100.0,
                'low': 28900.0,
                'close': 29050.0,
                'volume': 120.0
            }
        ]
        
        # Process and store
        count = aggregator.process_and_store(candles_1m, 'BTCUSDT')
        assert count == 60, f"Should store 60 1s candles, stored {count}"
        
        # Verify storage
        symbols = db.get_symbols()
        assert 'BTCUSDT' in symbols, "Symbol should be stored"
        
        db.close()
        
        print(f"  ✓ Process and store working correctly")
        print(f"    - Processed 1 x 1m candle")
        print(f"    - Stored {count} x 1s candles")
    
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_get_aggregated_candles():
    """Test retrieving and aggregating candles from database."""
    print("\nTesting Get Aggregated Candles...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        db = CandleDatabase(db_path)
        aggregator = CandleAggregator(db)
        
        # Store some 1s candles
        candles_1s = []
        base_ts = 1609459200000
        for i in range(120):  # 2 minutes worth
            candle = Candle1s(
                timestamp=base_ts + i * 1000,
                symbol='BTCUSDT',
                open=29000.0 + i,
                high=29000.0 + i + 10,
                low=29000.0 + i - 10,
                close=29000.0 + i + 5,
                volume=1.0,
                trades=3,
                source='interpolated'
            )
            candles_1s.append(candle)
        
        db.insert_candles(candles_1s)
        
        # Get aggregated 5s candles
        candles_5s = aggregator.get_aggregated_candles(
            'BTCUSDT',
            base_ts,
            base_ts + 120 * 1000,
            interval_seconds=5
        )
        
        assert len(candles_5s) == 24, f"Should get 24 5s candles, got {len(candles_5s)}"
        
        # Get aggregated 30s candles
        candles_30s = aggregator.get_aggregated_candles(
            'BTCUSDT',
            base_ts,
            base_ts + 120 * 1000,
            interval_seconds=30
        )
        
        assert len(candles_30s) == 4, f"Should get 4 30s candles, got {len(candles_30s)}"
        
        # Test caching
        candles_5s_cached = aggregator.get_aggregated_candles(
            'BTCUSDT',
            base_ts,
            base_ts + 120 * 1000,
            interval_seconds=5,
            use_cache=True
        )
        
        assert len(candles_5s_cached) == len(candles_5s), "Cached result should match"
        
        db.close()
        
        print(f"  ✓ Aggregated retrieval working correctly")
        print(f"    - Retrieved 120 x 1s candles")
        print(f"    - Aggregated to {len(candles_5s)} x 5s candles")
        print(f"    - Aggregated to {len(candles_30s)} x 30s candles")
        print(f"    - Cache hit test: OK")
    
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_multiple_symbols():
    """Test handling multiple trading pairs."""
    print("\nTesting Multiple Symbols...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    try:
        db = CandleDatabase(db_path)
        
        # Insert candles for multiple symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        base_ts = 1609459200000
        
        for symbol in symbols:
            candles = []
            for i in range(60):
                candle = Candle1s(
                    timestamp=base_ts + i * 1000,
                    symbol=symbol,
                    open=1000.0 + i,
                    high=1010.0 + i,
                    low=990.0 + i,
                    close=1005.0 + i,
                    volume=1.0,
                    trades=1,
                    source='interpolated'
                )
                candles.append(candle)
            
            db.insert_candles(candles)
        
        # Verify all symbols stored
        stored_symbols = db.get_symbols()
        assert len(stored_symbols) == 3, f"Should have 3 symbols, got {len(stored_symbols)}"
        
        for symbol in symbols:
            assert symbol in stored_symbols, f"{symbol} should be in database"
            
            # Verify each symbol has correct data
            candles = db.get_candles(symbol, base_ts, base_ts + 60 * 1000)
            assert len(candles) == 60, f"{symbol} should have 60 candles"
        
        db.close()
        
        print(f"  ✓ Multiple symbols working correctly")
        print(f"    - Stored data for {len(symbols)} symbols")
        print(f"    - Symbols: {', '.join(stored_symbols)}")
    
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting Edge Cases...")
    
    aggregator = CandleAggregator()
    
    # Empty candles list
    result = aggregator.convert_1m_to_1s([], 'BTCUSDT')
    assert len(result) == 0, "Empty input should produce empty output"
    
    # Single 1s candle aggregation
    candle = Candle1s(
        timestamp=1609459200000,
        symbol='BTCUSDT',
        open=29000.0,
        high=29010.0,
        low=28990.0,
        close=29005.0,
        volume=1.0,
        trades=1,
        source='actual'
    )
    
    result = aggregator.aggregate_1s_to_timeframe([candle], 1)
    assert len(result) == 1, "1s aggregation should preserve single candle"
    
    # Test invalid interval
    try:
        aggregator.aggregate_1s_to_timeframe([candle], 0)
        assert False, "Should raise ValueError for invalid interval"
    except ValueError:
        pass
    
    print(f"  ✓ Edge cases handled correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("PRATE Candle Aggregator Tests")
    print("=" * 60)
    
    test_convert_1m_to_1s()
    test_aggregate_1s_to_timeframe()
    test_database_storage()
    test_process_and_store()
    test_get_aggregated_candles()
    test_multiple_symbols()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("✓ All candle aggregator tests passed successfully!")
    print("=" * 60)
