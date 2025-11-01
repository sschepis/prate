"""
Data Ingestion Module - Historical data loading and normalization.

Supports:
- CSV/Parquet file loading
- Data normalization and preprocessing
- Missing data handling
- Time alignment and resampling
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: int  # milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderBookSnapshot:
    """Order book snapshot."""
    timestamp: int  # milliseconds
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]  # [(price, size), ...]


@dataclass
class Trade:
    """Individual trade."""
    timestamp: int  # milliseconds
    price: float
    size: float
    side: str  # 'buy' or 'sell'


class DataLoader:
    """
    Historical data loader with normalization and missing data handling.
    
    Supports loading from CSV and Parquet formats.
    """
    
    def __init__(
        self,
        symbol: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        fill_method: str = 'forward'
    ):
        """
        Initialize data loader.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_ts: Start timestamp in milliseconds (optional)
            end_ts: End timestamp in milliseconds (optional)
            fill_method: Method for filling missing data ('forward', 'backward', 'interpolate', 'zero')
        """
        self.symbol = symbol
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.fill_method = fill_method
        
        # Cached data
        self._candles: List[OHLCV] = []
        self._trades: List[Trade] = []
        self._snapshots: List[OrderBookSnapshot] = []
    
    def load_csv_candles(self, filepath: str) -> List[OHLCV]:
        """
        Load OHLCV candles from CSV file.
        
        Expected CSV format:
        timestamp,open,high,low,close,volume
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of OHLCV candles
        """
        candles = []
        
        try:
            with open(filepath, 'r') as f:
                header = f.readline().strip().split(',')
                
                # Validate header
                expected = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if header != expected:
                    raise ValueError(f"Invalid CSV header. Expected {expected}, got {header}")
                
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 6:
                        continue
                    
                    try:
                        ts = int(parts[0])
                        
                        # Filter by time range if specified
                        if self.start_ts is not None and ts < self.start_ts:
                            continue
                        if self.end_ts is not None and ts > self.end_ts:
                            continue
                        
                        candle = OHLCV(
                            timestamp=ts,
                            open=float(parts[1]),
                            high=float(parts[2]),
                            low=float(parts[3]),
                            close=float(parts[4]),
                            volume=float(parts[5])
                        )
                        candles.append(candle)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Handle missing data
        candles = self._fill_missing_candles(candles)
        
        self._candles = candles
        return candles
    
    def load_csv_trades(self, filepath: str) -> List[Trade]:
        """
        Load individual trades from CSV file.
        
        Expected CSV format:
        timestamp,price,size,side
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of trades
        """
        trades = []
        
        try:
            with open(filepath, 'r') as f:
                header = f.readline().strip().split(',')
                
                # Validate header
                expected = ['timestamp', 'price', 'size', 'side']
                if header != expected:
                    raise ValueError(f"Invalid CSV header. Expected {expected}, got {header}")
                
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 4:
                        continue
                    
                    try:
                        ts = int(parts[0])
                        
                        # Filter by time range if specified
                        if self.start_ts is not None and ts < self.start_ts:
                            continue
                        if self.end_ts is not None and ts > self.end_ts:
                            continue
                        
                        trade = Trade(
                            timestamp=ts,
                            price=float(parts[1]),
                            size=float(parts[2]),
                            side=parts[3].strip().lower()
                        )
                        trades.append(trade)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        self._trades = trades
        return trades
    
    def load_csv_orderbook(self, filepath: str) -> List[OrderBookSnapshot]:
        """
        Load order book snapshots from CSV file.
        
        Expected CSV format:
        timestamp,bid_prices,bid_sizes,ask_prices,ask_sizes
        
        Where prices/sizes are pipe-separated lists:
        1234567890,50000.0|49999.0,0.5|0.3,50001.0|50002.0,0.4|0.6
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of order book snapshots
        """
        snapshots = []
        
        try:
            with open(filepath, 'r') as f:
                header = f.readline().strip().split(',')
                
                # Validate header
                expected = ['timestamp', 'bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes']
                if header != expected:
                    raise ValueError(f"Invalid CSV header. Expected {expected}, got {header}")
                
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 5:
                        continue
                    
                    try:
                        ts = int(parts[0])
                        
                        # Filter by time range if specified
                        if self.start_ts is not None and ts < self.start_ts:
                            continue
                        if self.end_ts is not None and ts > self.end_ts:
                            continue
                        
                        # Parse bid/ask prices and sizes
                        bid_prices = [float(x) for x in parts[1].split('|') if x]
                        bid_sizes = [float(x) for x in parts[2].split('|') if x]
                        ask_prices = [float(x) for x in parts[3].split('|') if x]
                        ask_sizes = [float(x) for x in parts[4].split('|') if x]
                        
                        if len(bid_prices) != len(bid_sizes) or len(ask_prices) != len(ask_sizes):
                            continue
                        
                        snapshot = OrderBookSnapshot(
                            timestamp=ts,
                            bids=list(zip(bid_prices, bid_sizes)),
                            asks=list(zip(ask_prices, ask_sizes))
                        )
                        snapshots.append(snapshot)
                    except (ValueError, IndexError):
                        # Skip malformed lines
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        self._snapshots = snapshots
        return snapshots
    
    def _fill_missing_candles(self, candles: List[OHLCV]) -> List[OHLCV]:
        """
        Fill missing candles using specified fill method.
        
        Args:
            candles: List of candles (may have gaps)
            
        Returns:
            List of candles with gaps filled
        """
        if not candles or len(candles) < 2:
            return candles
        
        # Detect interval (assume uniform, use mode of differences)
        intervals = [candles[i+1].timestamp - candles[i].timestamp 
                    for i in range(min(10, len(candles)-1))]
        # Use minimum interval to avoid using gaps as the interval
        interval = int(min(intervals)) if intervals else 60000
        
        # Find gaps
        filled = [candles[0]]
        for i in range(1, len(candles)):
            prev_ts = filled[-1].timestamp
            curr_ts = candles[i].timestamp
            
            # Check for gap
            if curr_ts - prev_ts > interval * 1.5:
                # Fill gap
                num_missing = int((curr_ts - prev_ts) / interval) - 1
                
                for j in range(1, num_missing + 1):
                    gap_ts = prev_ts + j * interval
                    
                    if self.fill_method == 'forward':
                        # Forward fill
                        gap_candle = OHLCV(
                            timestamp=gap_ts,
                            open=filled[-1].close,
                            high=filled[-1].close,
                            low=filled[-1].close,
                            close=filled[-1].close,
                            volume=0.0
                        )
                    elif self.fill_method == 'backward':
                        # Backward fill
                        gap_candle = OHLCV(
                            timestamp=gap_ts,
                            open=candles[i].open,
                            high=candles[i].open,
                            low=candles[i].open,
                            close=candles[i].open,
                            volume=0.0
                        )
                    elif self.fill_method == 'interpolate':
                        # Linear interpolation
                        ratio = j / (num_missing + 1)
                        interp_price = filled[-1].close + ratio * (candles[i].open - filled[-1].close)
                        gap_candle = OHLCV(
                            timestamp=gap_ts,
                            open=interp_price,
                            high=interp_price,
                            low=interp_price,
                            close=interp_price,
                            volume=0.0
                        )
                    else:  # 'zero' or default
                        # Fill with zeros (not recommended for prices)
                        gap_candle = OHLCV(
                            timestamp=gap_ts,
                            open=0.0,
                            high=0.0,
                            low=0.0,
                            close=0.0,
                            volume=0.0
                        )
                    
                    filled.append(gap_candle)
            
            filled.append(candles[i])
        
        return filled
    
    def normalize_prices(
        self,
        candles: Optional[List[OHLCV]] = None,
        method: str = 'zscore'
    ) -> np.ndarray:
        """
        Normalize price data.
        
        Args:
            candles: List of candles (uses cached if None)
            method: Normalization method ('zscore', 'minmax', 'returns', 'log_returns')
            
        Returns:
            Normalized price array
        """
        if candles is None:
            candles = self._candles
        
        if not candles:
            return np.array([])
        
        prices = np.array([c.close for c in candles])
        
        if method == 'zscore':
            # Z-score normalization
            mean = np.mean(prices)
            std = np.std(prices)
            if std > 0:
                return (prices - mean) / std
            else:
                return prices - mean
        
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_p = np.min(prices)
            max_p = np.max(prices)
            if max_p > min_p:
                return (prices - min_p) / (max_p - min_p)
            else:
                return prices - min_p
        
        elif method == 'returns':
            # Simple returns
            if len(prices) < 2:
                return np.array([0.0])
            return np.diff(prices) / prices[:-1]
        
        elif method == 'log_returns':
            # Log returns
            if len(prices) < 2:
                return np.array([0.0])
            return np.diff(np.log(prices + 1e-10))
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_statistics(self, candles: Optional[List[OHLCV]] = None) -> Dict[str, float]:
        """
        Compute statistics on loaded data.
        
        Args:
            candles: List of candles (uses cached if None)
            
        Returns:
            Dictionary of statistics
        """
        if candles is None:
            candles = self._candles
        
        if not candles:
            return {}
        
        prices = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])
        
        return {
            'count': len(candles),
            'start_ts': candles[0].timestamp,
            'end_ts': candles[-1].timestamp,
            'duration_ms': candles[-1].timestamp - candles[0].timestamp,
            'price_mean': float(np.mean(prices)),
            'price_std': float(np.std(prices)),
            'price_min': float(np.min(prices)),
            'price_max': float(np.max(prices)),
            'volume_mean': float(np.mean(volumes)),
            'volume_std': float(np.std(volumes)),
            'volume_total': float(np.sum(volumes))
        }
    
    def resample_candles(
        self,
        candles: Optional[List[OHLCV]] = None,
        target_interval_ms: int = 60000
    ) -> List[OHLCV]:
        """
        Resample candles to a different time interval.
        
        Args:
            candles: List of candles (uses cached if None)
            target_interval_ms: Target interval in milliseconds
            
        Returns:
            Resampled candles
        """
        if candles is None:
            candles = self._candles
        
        if not candles:
            return []
        
        resampled = []
        current_bucket = []
        bucket_start = candles[0].timestamp - (candles[0].timestamp % target_interval_ms)
        
        for candle in candles:
            candle_bucket = candle.timestamp - (candle.timestamp % target_interval_ms)
            
            if candle_bucket != bucket_start:
                # Finalize current bucket
                if current_bucket:
                    resampled.append(self._aggregate_candles(current_bucket, bucket_start))
                
                # Start new bucket
                current_bucket = [candle]
                bucket_start = candle_bucket
            else:
                current_bucket.append(candle)
        
        # Finalize last bucket
        if current_bucket:
            resampled.append(self._aggregate_candles(current_bucket, bucket_start))
        
        return resampled
    
    def _aggregate_candles(self, candles: List[OHLCV], timestamp: int) -> OHLCV:
        """
        Aggregate multiple candles into one.
        
        Args:
            candles: List of candles to aggregate
            timestamp: Timestamp for aggregated candle
            
        Returns:
            Aggregated OHLCV candle
        """
        if not candles:
            raise ValueError("Cannot aggregate empty candle list")
        
        return OHLCV(
            timestamp=timestamp,
            open=candles[0].open,
            high=max(c.high for c in candles),
            low=min(c.low for c in candles),
            close=candles[-1].close,
            volume=sum(c.volume for c in candles)
        )
