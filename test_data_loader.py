#!/usr/bin/env python3
"""
Tests for the data ingestion module.
"""

import numpy as np
import tempfile
import os
from prate.data_loader import DataLoader, OHLCV, Trade, OrderBookSnapshot


def test_csv_candles_basic():
    """Test basic CSV candle loading."""
    print("Testing CSV Candle Loading...")
    
    # Create test CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("1609459200000,29000.0,29500.0,28800.0,29200.0,100.5\n")
        f.write("1609459260000,29200.0,29400.0,29100.0,29300.0,150.3\n")
        f.write("1609459320000,29300.0,29600.0,29250.0,29500.0,200.7\n")
        temp_path = f.name
    
    try:
        loader = DataLoader(symbol='BTCUSDT')
        candles = loader.load_csv_candles(temp_path)
        
        assert len(candles) == 3, f"Expected 3 candles, got {len(candles)}"
        assert candles[0].open == 29000.0, "First candle open price mismatch"
        assert candles[0].close == 29200.0, "First candle close price mismatch"
        assert candles[1].high == 29400.0, "Second candle high price mismatch"
        assert candles[2].volume == 200.7, "Third candle volume mismatch"
        
        print(f"  ✓ Loaded {len(candles)} candles successfully")
        print(f"    - First: {candles[0].open} -> {candles[0].close}")
        print(f"    - Last: {candles[-1].open} -> {candles[-1].close}")
    finally:
        os.unlink(temp_path)


def test_csv_candles_with_gaps():
    """Test CSV candle loading with missing data."""
    print("\nTesting CSV Candles with Gaps...")
    
    # Create test CSV file with gaps
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("1609459200000,29000.0,29500.0,28800.0,29200.0,100.5\n")
        f.write("1609459260000,29200.0,29400.0,29100.0,29300.0,150.3\n")
        # Gap here - missing 1609459320000
        f.write("1609459380000,29400.0,29600.0,29350.0,29500.0,200.7\n")
        temp_path = f.name
    
    try:
        loader = DataLoader(symbol='BTCUSDT', fill_method='forward')
        candles = loader.load_csv_candles(temp_path)
        
        # Should have filled the gap
        assert len(candles) == 4, f"Expected 4 candles (with gap filled), got {len(candles)}"
        
        # Gap-filled candle should use forward fill
        gap_candle = candles[2]
        assert gap_candle.timestamp == 1609459320000, "Gap candle timestamp incorrect"
        assert gap_candle.close == 29300.0, "Gap candle should forward fill from previous"
        
        print(f"  ✓ Gap filled successfully using forward fill")
        print(f"    - Gap candle: {gap_candle.open} -> {gap_candle.close} @ {gap_candle.timestamp}")
    finally:
        os.unlink(temp_path)


def test_csv_trades():
    """Test CSV trade loading."""
    print("\nTesting CSV Trade Loading...")
    
    # Create test CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("timestamp,price,size,side\n")
        f.write("1609459200000,29000.0,0.5,buy\n")
        f.write("1609459201000,29010.0,0.3,sell\n")
        f.write("1609459202000,29005.0,1.2,buy\n")
        temp_path = f.name
    
    try:
        loader = DataLoader(symbol='BTCUSDT')
        trades = loader.load_csv_trades(temp_path)
        
        assert len(trades) == 3, f"Expected 3 trades, got {len(trades)}"
        assert trades[0].price == 29000.0, "First trade price mismatch"
        assert trades[0].side == 'buy', "First trade side mismatch"
        assert trades[1].size == 0.3, "Second trade size mismatch"
        
        print(f"  ✓ Loaded {len(trades)} trades successfully")
        print(f"    - First: {trades[0].side} {trades[0].size} @ {trades[0].price}")
        print(f"    - Last: {trades[-1].side} {trades[-1].size} @ {trades[-1].price}")
    finally:
        os.unlink(temp_path)


def test_csv_orderbook():
    """Test CSV order book loading."""
    print("\nTesting CSV Order Book Loading...")
    
    # Create test CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("timestamp,bid_prices,bid_sizes,ask_prices,ask_sizes\n")
        f.write("1609459200000,28999.0|28998.0,0.5|0.3,29001.0|29002.0,0.4|0.6\n")
        f.write("1609459201000,29000.0|28999.0,0.6|0.4,29002.0|29003.0,0.5|0.7\n")
        temp_path = f.name
    
    try:
        loader = DataLoader(symbol='BTCUSDT')
        snapshots = loader.load_csv_orderbook(temp_path)
        
        assert len(snapshots) == 2, f"Expected 2 snapshots, got {len(snapshots)}"
        assert len(snapshots[0].bids) == 2, "First snapshot should have 2 bid levels"
        assert len(snapshots[0].asks) == 2, "First snapshot should have 2 ask levels"
        assert snapshots[0].bids[0][0] == 28999.0, "Top bid price mismatch"
        assert snapshots[0].asks[0][0] == 29001.0, "Top ask price mismatch"
        
        print(f"  ✓ Loaded {len(snapshots)} order book snapshots successfully")
        print(f"    - First snapshot: {len(snapshots[0].bids)} bids, {len(snapshots[0].asks)} asks")
        print(f"    - Best bid: {snapshots[0].bids[0][0]}, Best ask: {snapshots[0].asks[0][0]}")
    finally:
        os.unlink(temp_path)


def test_time_filtering():
    """Test time range filtering."""
    print("\nTesting Time Range Filtering...")
    
    # Create test CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("1609459200000,29000.0,29500.0,28800.0,29200.0,100.5\n")
        f.write("1609459260000,29200.0,29400.0,29100.0,29300.0,150.3\n")
        f.write("1609459320000,29300.0,29600.0,29250.0,29500.0,200.7\n")
        f.write("1609459380000,29500.0,29700.0,29400.0,29600.0,250.1\n")
        temp_path = f.name
    
    try:
        # Filter to get only middle candles
        loader = DataLoader(
            symbol='BTCUSDT',
            start_ts=1609459260000,
            end_ts=1609459320000
        )
        candles = loader.load_csv_candles(temp_path)
        
        assert len(candles) == 2, f"Expected 2 candles in time range, got {len(candles)}"
        assert candles[0].timestamp == 1609459260000, "First candle timestamp mismatch"
        assert candles[1].timestamp == 1609459320000, "Last candle timestamp mismatch"
        
        print(f"  ✓ Time filtering works correctly")
        print(f"    - Loaded {len(candles)} candles in time range")
    finally:
        os.unlink(temp_path)


def test_normalization():
    """Test price normalization methods."""
    print("\nTesting Price Normalization...")
    
    # Create test data
    candles = [
        OHLCV(1609459200000, 100.0, 105.0, 98.0, 102.0, 50.0),
        OHLCV(1609459260000, 102.0, 108.0, 101.0, 105.0, 60.0),
        OHLCV(1609459320000, 105.0, 110.0, 104.0, 108.0, 70.0),
        OHLCV(1609459380000, 108.0, 112.0, 107.0, 110.0, 80.0),
    ]
    
    loader = DataLoader(symbol='BTCUSDT')
    loader._candles = candles
    
    # Test z-score normalization
    zscore = loader.normalize_prices(method='zscore')
    assert len(zscore) == 4, "Z-score normalization should preserve length"
    assert abs(np.mean(zscore)) < 1e-10, "Z-score mean should be ~0"
    assert abs(np.std(zscore) - 1.0) < 1e-10, "Z-score std should be ~1"
    print(f"  ✓ Z-score normalization: mean={np.mean(zscore):.6f}, std={np.std(zscore):.6f}")
    
    # Test min-max normalization
    minmax = loader.normalize_prices(method='minmax')
    assert len(minmax) == 4, "Min-max normalization should preserve length"
    assert abs(np.min(minmax)) < 1e-10, "Min-max min should be ~0"
    assert abs(np.max(minmax) - 1.0) < 1e-10, "Min-max max should be ~1"
    print(f"  ✓ Min-max normalization: min={np.min(minmax):.6f}, max={np.max(minmax):.6f}")
    
    # Test returns
    returns = loader.normalize_prices(method='returns')
    assert len(returns) == 3, "Returns should have n-1 elements"
    expected_return_0 = (105.0 - 102.0) / 102.0
    assert abs(returns[0] - expected_return_0) < 1e-6, "First return calculation incorrect"
    print(f"  ✓ Returns: first={returns[0]:.6f}, mean={np.mean(returns):.6f}")
    
    # Test log returns
    log_returns = loader.normalize_prices(method='log_returns')
    assert len(log_returns) == 3, "Log returns should have n-1 elements"
    print(f"  ✓ Log returns: first={log_returns[0]:.6f}, mean={np.mean(log_returns):.6f}")


def test_statistics():
    """Test statistics computation."""
    print("\nTesting Statistics Computation...")
    
    candles = [
        OHLCV(1609459200000, 100.0, 105.0, 98.0, 102.0, 50.0),
        OHLCV(1609459260000, 102.0, 108.0, 101.0, 105.0, 60.0),
        OHLCV(1609459320000, 105.0, 110.0, 104.0, 108.0, 70.0),
    ]
    
    loader = DataLoader(symbol='BTCUSDT')
    loader._candles = candles
    
    stats = loader.get_statistics()
    
    assert stats['count'] == 3, "Count mismatch"
    assert stats['start_ts'] == 1609459200000, "Start timestamp mismatch"
    assert stats['end_ts'] == 1609459320000, "End timestamp mismatch"
    assert stats['price_min'] == 102.0, "Min price mismatch"
    assert stats['price_max'] == 108.0, "Max price mismatch"
    assert abs(stats['price_mean'] - 105.0) < 1e-6, "Mean price mismatch"
    assert abs(stats['volume_total'] - 180.0) < 1e-6, "Total volume mismatch"
    
    print(f"  ✓ Statistics computed correctly")
    print(f"    - Count: {stats['count']}")
    print(f"    - Price range: {stats['price_min']:.2f} - {stats['price_max']:.2f}")
    print(f"    - Mean price: {stats['price_mean']:.2f}")
    print(f"    - Total volume: {stats['volume_total']:.2f}")


def test_resampling():
    """Test candle resampling."""
    print("\nTesting Candle Resampling...")
    
    # Create 1-minute candles
    candles = [
        OHLCV(1609459200000, 100.0, 105.0, 98.0, 102.0, 50.0),   # 00:00
        OHLCV(1609459260000, 102.0, 108.0, 101.0, 105.0, 60.0),  # 01:00
        OHLCV(1609459320000, 105.0, 110.0, 104.0, 108.0, 70.0),  # 02:00
        OHLCV(1609459380000, 108.0, 112.0, 107.0, 110.0, 80.0),  # 03:00
    ]
    
    loader = DataLoader(symbol='BTCUSDT')
    loader._candles = candles
    
    # Resample to 2-minute candles
    resampled = loader.resample_candles(target_interval_ms=120000)
    
    assert len(resampled) == 2, f"Expected 2 resampled candles, got {len(resampled)}"
    
    # First resampled candle should combine first two
    assert resampled[0].open == 100.0, "First resampled open should be from first candle"
    assert resampled[0].close == 105.0, "First resampled close should be from second candle"
    assert resampled[0].high == 108.0, "First resampled high should be max of both"
    assert resampled[0].low == 98.0, "First resampled low should be min of both"
    assert resampled[0].volume == 110.0, "First resampled volume should be sum"
    
    # Second resampled candle should combine last two
    assert resampled[1].open == 105.0, "Second resampled open mismatch"
    assert resampled[1].close == 110.0, "Second resampled close mismatch"
    assert resampled[1].volume == 150.0, "Second resampled volume should be sum"
    
    print(f"  ✓ Resampling works correctly")
    print(f"    - Original: {len(candles)} candles")
    print(f"    - Resampled: {len(resampled)} candles")
    print(f"    - First resampled: {resampled[0].open} -> {resampled[0].close}, vol={resampled[0].volume}")


def test_malformed_data():
    """Test handling of malformed CSV data."""
    print("\nTesting Malformed Data Handling...")
    
    # Create test CSV file with some bad lines (no gaps between valid timestamps)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write("timestamp,open,high,low,close,volume\n")
        f.write("1609459200000,29000.0,29500.0,28800.0,29200.0,100.5\n")
        f.write("malformed line\n")  # Bad line
        f.write("1609459260000,29200.0,29400.0,29100.0,29300.0,150.3\n")
        f.write("1609459320000,invalid,29600.0,29250.0,29500.0,200.7\n")  # Bad data
        f.write("1609459380000,29500.0,29700.0,29400.0,29600.0,250.1\n")
        temp_path = f.name
    
    try:
        loader = DataLoader(symbol='BTCUSDT', fill_method='forward')
        candles = loader.load_csv_candles(temp_path)
        
        # Should skip malformed lines and load valid ones (3 valid candles)
        # But may fill gaps, so count valid prices
        valid_candles = [c for c in candles if c.open > 0]
        assert len(valid_candles) >= 3, f"Expected at least 3 valid candles, got {len(valid_candles)}"
        
        print(f"  ✓ Malformed data handled gracefully")
        print(f"    - Loaded {len(candles)} total candles ({len(valid_candles)} with valid data, skipped 2 malformed)")
    finally:
        os.unlink(temp_path)


if __name__ == '__main__':
    print("=" * 60)
    print("PRATE Data Loader Tests")
    print("=" * 60)
    
    test_csv_candles_basic()
    test_csv_candles_with_gaps()
    test_csv_trades()
    test_csv_orderbook()
    test_time_filtering()
    test_normalization()
    test_statistics()
    test_resampling()
    test_malformed_data()
    
    print("\n" + "=" * 60)
    print("✓ All data loader tests passed successfully!")
    print("=" * 60)
