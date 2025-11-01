"""
Metrics and Monitoring System for PRATE.

Provides comprehensive metrics database, real-time dashboards, trade audit logs,
memory diagnostics, and entropy/coherence visualization.
"""

import sqlite3
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
import numpy as np


@dataclass
class TradeMetric:
    """Record for a single trade."""
    timestamp: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    pnl: float
    cumulative_pnl: float
    guild_id: Optional[str] = None
    basis_id: Optional[int] = None
    metadata: Optional[str] = None  # JSON string


@dataclass
class SystemMetric:
    """Record for system-level metrics."""
    timestamp: str
    metric_name: str
    metric_value: float
    metadata: Optional[str] = None  # JSON string


@dataclass
class MemoryDiagnostic:
    """Record for memory diagnostics."""
    timestamp: str
    memory_dim: int
    memory_norm: float
    retrieval_quality: float
    binding_entropy: float
    metadata: Optional[str] = None  # JSON string


@dataclass
class EntropyMetric:
    """Record for entropy/coherence metrics."""
    timestamp: str
    hilbert_entropy: float
    target_entropy: float
    tau_value: float
    coherence: float
    basis_id: int
    metadata: Optional[str] = None  # JSON string


class MetricsDB:
    """
    SQLite-based metrics database for PRATE.
    
    Features:
    - Trade audit logging
    - System metrics tracking
    - Memory diagnostics
    - Entropy/coherence monitoring
    - Thread-safe operations
    - Query and aggregation methods
    """
    
    def __init__(self, db_path: str = "metrics.db"):
        """
        Initialize metrics database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    pnl REAL NOT NULL,
                    cumulative_pnl REAL NOT NULL,
                    guild_id TEXT,
                    basis_id INTEGER,
                    metadata TEXT
                )
            """)
            
            # System metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Memory diagnostics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_diagnostics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    memory_dim INTEGER NOT NULL,
                    memory_norm REAL NOT NULL,
                    retrieval_quality REAL NOT NULL,
                    binding_entropy REAL NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Entropy metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entropy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    hilbert_entropy REAL NOT NULL,
                    target_entropy REAL NOT NULL,
                    tau_value REAL NOT NULL,
                    coherence REAL NOT NULL,
                    basis_id INTEGER NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_diagnostics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entropy_timestamp ON entropy_metrics(timestamp)")
            
            conn.commit()
    
    def log_trade(self, trade: TradeMetric) -> int:
        """
        Log a trade to the database.
        
        Args:
            trade: Trade metric to log
            
        Returns:
            ID of inserted record
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (timestamp, symbol, side, quantity, price, fee, pnl, 
                                   cumulative_pnl, guild_id, basis_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.timestamp, trade.symbol, trade.side, trade.quantity,
                trade.price, trade.fee, trade.pnl, trade.cumulative_pnl,
                trade.guild_id, trade.basis_id, trade.metadata
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_system_metric(self, metric: SystemMetric) -> int:
        """
        Log a system metric.
        
        Args:
            metric: System metric to log
            
        Returns:
            ID of inserted record
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_metrics (timestamp, metric_name, metric_value, metadata)
                VALUES (?, ?, ?, ?)
            """, (metric.timestamp, metric.metric_name, metric.metric_value, metric.metadata))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_memory_diagnostic(self, diagnostic: MemoryDiagnostic) -> int:
        """
        Log memory diagnostics.
        
        Args:
            diagnostic: Memory diagnostic to log
            
        Returns:
            ID of inserted record
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO memory_diagnostics (timestamp, memory_dim, memory_norm, 
                                                retrieval_quality, binding_entropy, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                diagnostic.timestamp, diagnostic.memory_dim, diagnostic.memory_norm,
                diagnostic.retrieval_quality, diagnostic.binding_entropy, diagnostic.metadata
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_entropy_metric(self, metric: EntropyMetric) -> int:
        """
        Log entropy/coherence metrics.
        
        Args:
            metric: Entropy metric to log
            
        Returns:
            ID of inserted record
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO entropy_metrics (timestamp, hilbert_entropy, target_entropy,
                                            tau_value, coherence, basis_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp, metric.hilbert_entropy, metric.target_entropy,
                metric.tau_value, metric.coherence, metric.basis_id, metric.metadata
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_trades(self, symbol: Optional[str] = None, 
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query trades from database.
        
        Args:
            symbol: Filter by symbol (optional)
            start_time: Filter by start time (ISO format, optional)
            end_time: Filter by end time (ISO format, optional)
            limit: Maximum number of records
            
        Returns:
            List of trade records
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_system_metrics(self, metric_name: Optional[str] = None,
                          start_time: Optional[str] = None,
                          end_time: Optional[str] = None,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query system metrics from database.
        
        Args:
            metric_name: Filter by metric name (optional)
            start_time: Filter by start time (ISO format, optional)
            end_time: Filter by end time (ISO format, optional)
            limit: Maximum number of records
            
        Returns:
            List of system metric records
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM system_metrics WHERE 1=1"
            params = []
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_summary(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            symbol: Filter by symbol (optional)
            
        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_profit,
                    MIN(pnl) as max_loss,
                    SUM(fee) as total_fees,
                    MAX(cumulative_pnl) as peak_pnl,
                    MIN(cumulative_pnl) as trough_pnl
                FROM trades
                WHERE 1=1
            """
            
            params = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            result = dict(row) if row else {}
            
            # Calculate derived metrics
            if result.get('total_trades', 0) > 0:
                result['win_rate'] = result['winning_trades'] / result['total_trades']
                
                # Calculate max drawdown
                result['max_drawdown'] = result['peak_pnl'] - result['trough_pnl'] if result['peak_pnl'] else 0
            else:
                result['win_rate'] = 0
                result['max_drawdown'] = 0
            
            return result
    
    def get_entropy_stats(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Get entropy statistics.
        
        Args:
            limit: Number of recent records to analyze
            
        Returns:
            Dictionary with entropy statistics
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    AVG(hilbert_entropy) as avg_entropy,
                    AVG(ABS(hilbert_entropy - target_entropy)) as avg_error,
                    AVG(tau_value) as avg_tau,
                    AVG(coherence) as avg_coherence
                FROM (
                    SELECT * FROM entropy_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
            """, (limit,))
            
            row = cursor.fetchone()
            return dict(row) if row else {}
    
    def clear_old_data(self, days: int = 30) -> Dict[str, int]:
        """
        Clear data older than specified days.
        
        Args:
            days: Number of days to retain
            
        Returns:
            Dictionary with number of deleted records per table
        """
        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_time = cutoff_time - timedelta(days=days)
        cutoff_str = cutoff_time.isoformat()
        
        deleted = {}
        
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for table in ['trades', 'system_metrics', 'memory_diagnostics', 'entropy_metrics']:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_str,))
                deleted[table] = cursor.rowcount
            
            conn.commit()
        
        return deleted
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class MetricsCollector:
    """
    High-level metrics collector for PRATE system.
    
    Simplifies metrics collection by providing convenient methods
    for common operations.
    """
    
    def __init__(self, db: MetricsDB):
        """
        Initialize metrics collector.
        
        Args:
            db: MetricsDB instance
        """
        self.db = db
        self._cumulative_pnl = 0.0
    
    def record_trade(self, symbol: str, side: str, quantity: float, 
                    price: float, fee: float, pnl: float,
                    guild_id: Optional[str] = None, 
                    basis_id: Optional[int] = None,
                    **metadata) -> None:
        """
        Record a trade.
        
        Args:
            symbol: Trading symbol
            side: Trade side (BUY/SELL)
            quantity: Trade quantity
            price: Execution price
            fee: Trading fee
            pnl: Trade PnL
            guild_id: Guild that generated the trade (optional)
            basis_id: Basis ID used (optional)
            **metadata: Additional metadata
        """
        self._cumulative_pnl += pnl
        
        trade = TradeMetric(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            pnl=pnl,
            cumulative_pnl=self._cumulative_pnl,
            guild_id=guild_id,
            basis_id=basis_id,
            metadata=json.dumps(metadata) if metadata else None
        )
        
        self.db.log_trade(trade)
    
    def record_metric(self, name: str, value: float, **metadata) -> None:
        """
        Record a system metric.
        
        Args:
            name: Metric name
            value: Metric value
            **metadata: Additional metadata
        """
        metric = SystemMetric(
            timestamp=datetime.now().isoformat(),
            metric_name=name,
            metric_value=value,
            metadata=json.dumps(metadata) if metadata else None
        )
        
        self.db.log_system_metric(metric)
    
    def record_memory_state(self, memory_vec: np.ndarray, 
                           retrieval_quality: float = 0.0,
                           **metadata) -> None:
        """
        Record memory diagnostics.
        
        Args:
            memory_vec: Memory vector
            retrieval_quality: Quality of retrieval (0-1)
            **metadata: Additional metadata
        """
        # Calculate diagnostics
        memory_norm = float(np.linalg.norm(memory_vec))
        
        # Calculate binding entropy (simple measure)
        phase = np.angle(memory_vec)
        phase_hist, _ = np.histogram(phase, bins=32, density=True)
        phase_hist = phase_hist + 1e-10  # Avoid log(0)
        binding_entropy = -float(np.sum(phase_hist * np.log(phase_hist)))
        
        diagnostic = MemoryDiagnostic(
            timestamp=datetime.now().isoformat(),
            memory_dim=len(memory_vec),
            memory_norm=memory_norm,
            retrieval_quality=retrieval_quality,
            binding_entropy=binding_entropy,
            metadata=json.dumps(metadata) if metadata else None
        )
        
        self.db.log_memory_diagnostic(diagnostic)
    
    def record_entropy_state(self, hilbert_entropy: float, target_entropy: float,
                            tau_value: float, coherence: float, basis_id: int,
                            **metadata) -> None:
        """
        Record entropy/coherence metrics.
        
        Args:
            hilbert_entropy: Current Hilbert entropy
            target_entropy: Target entropy (H*)
            tau_value: Current tau value
            coherence: Coherence measure
            basis_id: Current basis ID
            **metadata: Additional metadata
        """
        metric = EntropyMetric(
            timestamp=datetime.now().isoformat(),
            hilbert_entropy=hilbert_entropy,
            target_entropy=target_entropy,
            tau_value=tau_value,
            coherence=coherence,
            basis_id=basis_id,
            metadata=json.dumps(metadata) if metadata else None
        )
        
        self.db.log_entropy_metric(metric)
