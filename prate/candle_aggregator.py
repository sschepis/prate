"""
Candle Aggregator - Convert and aggregate trading candles at different timeframes.

Supports:
- Converting 1-minute candles to 1-second candles
- Aggregating 1s candles to any desired timeframe (2s-600s)
- Database storage and retrieval
- Multiple trading pairs
- Efficient caching
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json


@dataclass
class Candle1s:
    """1-second candle data."""
    timestamp: int  # Unix timestamp in milliseconds
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int = 0  # Number of trades in this second
    source: str = 'interpolated'  # 'interpolated' or 'actual'


class CandleDatabase:
    """
    SQLite database for storing 1-second candles.
    
    Provides efficient storage and retrieval of high-frequency candle data.
    """
    
    def __init__(self, db_path: str = 'candles.db'):
        """
        Initialize candle database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # 1-second candles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles_1s (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                trades INTEGER DEFAULT 0,
                source TEXT DEFAULT 'interpolated',
                UNIQUE(timestamp, symbol)
            )
        ''')
        
        # Create index for fast lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp_symbol 
            ON candles_1s(timestamp, symbol)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp 
            ON candles_1s(symbol, timestamp)
        ''')
        
        self.conn.commit()
    
    def insert_candles(self, candles: List[Candle1s]) -> int:
        """
        Insert 1-second candles into database.
        
        Args:
            candles: List of 1-second candles
            
        Returns:
            Number of candles inserted
        """
        if not candles:
            return 0
        
        cursor = self.conn.cursor()
        
        # Use INSERT OR REPLACE to handle duplicates
        inserted = 0
        for candle in candles:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO candles_1s 
                    (timestamp, symbol, open, high, low, close, volume, trades, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    candle.timestamp,
                    candle.symbol,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                    candle.trades,
                    candle.source
                ))
                inserted += 1
            except sqlite3.Error as e:
                print(f"Error inserting candle: {e}")
                continue
        
        self.conn.commit()
        return inserted
    
    def get_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int
    ) -> List[Candle1s]:
        """
        Retrieve 1-second candles from database.
        
        Args:
            symbol: Trading pair symbol
            start_ts: Start timestamp (milliseconds)
            end_ts: End timestamp (milliseconds)
            
        Returns:
            List of 1-second candles
        """
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, symbol, open, high, low, close, volume, trades, source
            FROM candles_1s
            WHERE symbol = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp
        ''', (symbol, start_ts, end_ts))
        
        candles = []
        for row in cursor.fetchall():
            candles.append(Candle1s(
                timestamp=row[0],
                symbol=row[1],
                open=row[2],
                high=row[3],
                low=row[4],
                close=row[5],
                volume=row[6],
                trades=row[7],
                source=row[8]
            ))
        
        return candles
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols in database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT DISTINCT symbol FROM candles_1s ORDER BY symbol')
        return [row[0] for row in cursor.fetchall()]
    
    def get_time_range(self, symbol: str) -> Optional[Tuple[int, int]]:
        """
        Get time range for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            (start_ts, end_ts) tuple or None if no data
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MIN(timestamp), MAX(timestamp)
            FROM candles_1s
            WHERE symbol = ?
        ''', (symbol,))
        
        row = cursor.fetchone()
        if row and row[0] is not None:
            return (row[0], row[1])
        return None
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


class CandleAggregator:
    """
    Candle aggregator for converting and aggregating candles.
    
    Main functions:
    - Convert 1m candles to 1s candles (interpolation)
    - Aggregate 1s candles to any timeframe (2s-600s)
    """
    
    def __init__(self, db: Optional[CandleDatabase] = None):
        """
        Initialize candle aggregator.
        
        Args:
            db: Optional database for storage/retrieval
        """
        self.db = db
        self.cache: Dict[str, List[Candle1s]] = {}
    
    def convert_1m_to_1s(
        self,
        candles_1m: List[Dict],
        symbol: str
    ) -> List[Candle1s]:
        """
        Convert 1-minute candles to 1-second candles using interpolation.
        
        Strategy:
        - Linear interpolation for prices
        - Uniform distribution of volume
        - Mark as 'interpolated' source
        
        Args:
            candles_1m: List of 1-minute candles (OHLCV dicts)
            symbol: Trading pair symbol
            
        Returns:
            List of 1-second candles
        """
        if not candles_1m:
            return []
        
        candles_1s = []
        
        for i, candle_1m in enumerate(candles_1m):
            timestamp_1m = candle_1m['timestamp']
            open_price = candle_1m['open']
            high_price = candle_1m['high']
            low_price = candle_1m['low']
            close_price = candle_1m['close']
            volume_1m = candle_1m['volume']
            
            # Distribute volume uniformly across 60 seconds
            volume_per_second = volume_1m / 60.0
            
            # Generate 60 1-second candles
            for sec in range(60):
                ts_1s = timestamp_1m + sec * 1000  # Add seconds in milliseconds
                
                # Interpolate price within the 1m candle
                # Simple linear interpolation from open to close
                ratio = sec / 60.0
                
                # Base interpolated price
                interp_price = open_price + (close_price - open_price) * ratio
                
                # Add some variation based on high/low
                # This is a simplified model - real data would be better
                price_range = high_price - low_price
                
                # Create realistic OHLC for this second
                # Open is the previous close (or interpolated price)
                if sec == 0:
                    sec_open = open_price
                else:
                    sec_open = candles_1s[-1].close
                
                # Close is the interpolated price with small random walk
                sec_close = interp_price
                
                # High/low based on volatility
                volatility_factor = price_range / 60.0
                sec_high = max(sec_open, sec_close) + volatility_factor * 0.3
                sec_low = min(sec_open, sec_close) - volatility_factor * 0.3
                
                # Ensure high/low respect the 1m candle bounds
                sec_high = min(sec_high, high_price)
                sec_low = max(sec_low, low_price)
                
                candle_1s = Candle1s(
                    timestamp=ts_1s,
                    symbol=symbol,
                    open=sec_open,
                    high=sec_high,
                    low=sec_low,
                    close=sec_close,
                    volume=volume_per_second,
                    trades=1,  # Estimated
                    source='interpolated'
                )
                
                candles_1s.append(candle_1s)
        
        return candles_1s
    
    def aggregate_1s_to_timeframe(
        self,
        candles_1s: List[Candle1s],
        interval_seconds: int
    ) -> List[Candle1s]:
        """
        Aggregate 1-second candles to any desired timeframe.
        
        Args:
            candles_1s: List of 1-second candles
            interval_seconds: Target interval in seconds (e.g., 2, 5, 60, 300)
            
        Returns:
            List of aggregated candles
        """
        if not candles_1s:
            return []
        
        if interval_seconds <= 0:
            raise ValueError("Interval must be positive")
        
        aggregated = []
        current_bucket = []
        bucket_start_ts = None
        
        interval_ms = interval_seconds * 1000
        
        for candle in candles_1s:
            # Determine which bucket this candle belongs to
            candle_bucket = (candle.timestamp // interval_ms) * interval_ms
            
            if bucket_start_ts is None:
                bucket_start_ts = candle_bucket
            
            if candle_bucket == bucket_start_ts:
                # Same bucket
                current_bucket.append(candle)
            else:
                # New bucket - aggregate previous bucket
                if current_bucket:
                    agg_candle = self._aggregate_candles(current_bucket, bucket_start_ts)
                    aggregated.append(agg_candle)
                
                # Start new bucket
                current_bucket = [candle]
                bucket_start_ts = candle_bucket
        
        # Aggregate last bucket
        if current_bucket:
            agg_candle = self._aggregate_candles(current_bucket, bucket_start_ts)
            aggregated.append(agg_candle)
        
        return aggregated
    
    def _aggregate_candles(
        self,
        candles: List[Candle1s],
        timestamp: int
    ) -> Candle1s:
        """
        Aggregate multiple candles into one.
        
        Args:
            candles: List of candles to aggregate
            timestamp: Timestamp for aggregated candle
            
        Returns:
            Aggregated candle
        """
        if not candles:
            raise ValueError("Cannot aggregate empty candle list")
        
        symbol = candles[0].symbol
        
        # OHLC aggregation
        open_price = candles[0].open
        high_price = max(c.high for c in candles)
        low_price = min(c.low for c in candles)
        close_price = candles[-1].close
        
        # Volume aggregation
        total_volume = sum(c.volume for c in candles)
        total_trades = sum(c.trades for c in candles)
        
        # Source: actual if any actual data, else interpolated
        source = 'actual' if any(c.source == 'actual' for c in candles) else 'interpolated'
        
        return Candle1s(
            timestamp=timestamp,
            symbol=symbol,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
            trades=total_trades,
            source=source
        )
    
    def process_and_store(
        self,
        candles_1m: List[Dict],
        symbol: str
    ) -> int:
        """
        Convert 1m candles to 1s and store in database.
        
        Args:
            candles_1m: List of 1-minute candles
            symbol: Trading pair symbol
            
        Returns:
            Number of 1s candles stored
        """
        if self.db is None:
            raise ValueError("Database not configured")
        
        # Convert to 1s candles
        candles_1s = self.convert_1m_to_1s(candles_1m, symbol)
        
        # Store in database
        count = self.db.insert_candles(candles_1s)
        
        return count
    
    def get_aggregated_candles(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        interval_seconds: int,
        use_cache: bool = True
    ) -> List[Candle1s]:
        """
        Get aggregated candles from database.
        
        Args:
            symbol: Trading pair symbol
            start_ts: Start timestamp (milliseconds)
            end_ts: End timestamp (milliseconds)
            interval_seconds: Aggregation interval in seconds
            use_cache: Whether to use cache
            
        Returns:
            List of aggregated candles
        """
        if self.db is None:
            raise ValueError("Database not configured")
        
        # Check cache
        cache_key = f"{symbol}_{start_ts}_{end_ts}_1s"
        if use_cache and cache_key in self.cache:
            candles_1s = self.cache[cache_key]
        else:
            # Fetch from database
            candles_1s = self.db.get_candles(symbol, start_ts, end_ts)
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = candles_1s
        
        # Aggregate to desired interval
        if interval_seconds == 1:
            return candles_1s
        else:
            return self.aggregate_1s_to_timeframe(candles_1s, interval_seconds)
    
    def clear_cache(self) -> None:
        """Clear the candle cache."""
        self.cache.clear()
