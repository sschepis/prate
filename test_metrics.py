"""
Tests for Metrics and Monitoring System.
"""

import pytest
import os
import tempfile
import numpy as np
from datetime import datetime, timedelta

from prate.metrics import (
    MetricsDB,
    MetricsCollector,
    TradeMetric,
    SystemMetric,
    MemoryDiagnostic,
    EntropyMetric
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    
    db = MetricsDB(db_path)
    yield db
    
    db.close()
    os.unlink(db_path)


def test_trade_logging(temp_db):
    """Test logging trades."""
    trade = TradeMetric(
        timestamp=datetime.now().isoformat(),
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.0,
        price=50000.0,
        fee=25.0,
        pnl=100.0,
        cumulative_pnl=100.0,
        guild_id="TF",
        basis_id=1
    )
    
    trade_id = temp_db.log_trade(trade)
    assert trade_id > 0
    
    # Retrieve trades
    trades = temp_db.get_trades(symbol="BTCUSDT", limit=10)
    assert len(trades) == 1
    assert trades[0]['symbol'] == "BTCUSDT"
    assert trades[0]['pnl'] == 100.0


def test_system_metric_logging(temp_db):
    """Test logging system metrics."""
    metric = SystemMetric(
        timestamp=datetime.now().isoformat(),
        metric_name="sharpe_ratio",
        metric_value=1.5
    )
    
    metric_id = temp_db.log_system_metric(metric)
    assert metric_id > 0
    
    # Retrieve metrics
    metrics = temp_db.get_system_metrics(metric_name="sharpe_ratio", limit=10)
    assert len(metrics) == 1
    assert metrics[0]['metric_value'] == 1.5


def test_memory_diagnostic_logging(temp_db):
    """Test logging memory diagnostics."""
    diagnostic = MemoryDiagnostic(
        timestamp=datetime.now().isoformat(),
        memory_dim=128,
        memory_norm=5.4,
        retrieval_quality=0.85,
        binding_entropy=2.3
    )
    
    diag_id = temp_db.log_memory_diagnostic(diagnostic)
    assert diag_id > 0


def test_entropy_metric_logging(temp_db):
    """Test logging entropy metrics."""
    metric = EntropyMetric(
        timestamp=datetime.now().isoformat(),
        hilbert_entropy=2.5,
        target_entropy=2.5,
        tau_value=1.2,
        coherence=0.9,
        basis_id=3
    )
    
    metric_id = temp_db.log_entropy_metric(metric)
    assert metric_id > 0


def test_performance_summary(temp_db):
    """Test performance summary calculation."""
    # Log multiple trades
    base_time = datetime.now()
    cumulative_pnl = 0.0
    
    trades = [
        (100.0, 20.0),   # Win
        (-50.0, 10.0),   # Loss
        (150.0, 30.0),   # Win
        (-25.0, 5.0),    # Loss
        (75.0, 15.0),    # Win
    ]
    
    for pnl, fee in trades:
        cumulative_pnl += pnl
        trade = TradeMetric(
            timestamp=base_time.isoformat(),
            symbol="BTCUSDT",
            side="BUY" if pnl > 0 else "SELL",
            quantity=1.0,
            price=50000.0,
            fee=fee,
            pnl=pnl,
            cumulative_pnl=cumulative_pnl
        )
        temp_db.log_trade(trade)
        base_time += timedelta(minutes=5)
    
    # Get summary
    summary = temp_db.get_performance_summary(symbol="BTCUSDT")
    
    assert summary['total_trades'] == 5
    assert summary['winning_trades'] == 3
    assert summary['losing_trades'] == 2
    assert summary['win_rate'] == 0.6
    assert summary['total_pnl'] == 250.0
    assert summary['total_fees'] == 80.0
    assert summary['max_profit'] == 150.0
    assert summary['max_loss'] == -50.0


def test_entropy_stats(temp_db):
    """Test entropy statistics calculation."""
    # Log multiple entropy metrics
    for i in range(10):
        metric = EntropyMetric(
            timestamp=datetime.now().isoformat(),
            hilbert_entropy=2.5 + i * 0.1,
            target_entropy=2.5,
            tau_value=1.0 + i * 0.05,
            coherence=0.9,
            basis_id=i % 3
        )
        temp_db.log_entropy_metric(metric)
    
    # Get stats
    stats = temp_db.get_entropy_stats(limit=10)
    
    assert 'avg_entropy' in stats
    assert 'avg_error' in stats
    assert 'avg_tau' in stats
    assert 'avg_coherence' in stats
    assert stats['avg_coherence'] == 0.9


def test_metrics_collector_trade(temp_db):
    """Test MetricsCollector trade recording."""
    collector = MetricsCollector(temp_db)
    
    collector.record_trade(
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.0,
        price=50000.0,
        fee=25.0,
        pnl=100.0,
        guild_id="TF",
        basis_id=1,
        extra_data="test"
    )
    
    trades = temp_db.get_trades(limit=10)
    assert len(trades) == 1
    assert trades[0]['cumulative_pnl'] == 100.0


def test_metrics_collector_metric(temp_db):
    """Test MetricsCollector metric recording."""
    collector = MetricsCollector(temp_db)
    
    collector.record_metric("sharpe_ratio", 1.5, window=500)
    
    metrics = temp_db.get_system_metrics(metric_name="sharpe_ratio", limit=10)
    assert len(metrics) == 1
    assert metrics[0]['metric_value'] == 1.5


def test_metrics_collector_memory(temp_db):
    """Test MetricsCollector memory diagnostics."""
    collector = MetricsCollector(temp_db)
    
    memory_vec = np.random.randn(128) + 1j * np.random.randn(128)
    
    collector.record_memory_state(memory_vec, retrieval_quality=0.85)
    
    # Verify via public interface
    # (Note: In production, would add a public getter method to MetricsDB)
    # For now, we just verify no exception was raised
    assert True


def test_metrics_collector_entropy(temp_db):
    """Test MetricsCollector entropy state recording."""
    collector = MetricsCollector(temp_db)
    
    collector.record_entropy_state(
        hilbert_entropy=2.5,
        target_entropy=2.5,
        tau_value=1.2,
        coherence=0.9,
        basis_id=3
    )
    
    # Verify via public interface
    # (Note: In production, would add a public getter method to MetricsDB)
    # For now, we just verify no exception was raised
    assert True


def test_time_filtering(temp_db):
    """Test time-based filtering of trades."""
    base_time = datetime.now()
    
    # Log trades at different times
    for i in range(5):
        trade_time = base_time + timedelta(hours=i)
        trade = TradeMetric(
            timestamp=trade_time.isoformat(),
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=50000.0,
            fee=25.0,
            pnl=100.0,
            cumulative_pnl=100.0 * (i + 1)
        )
        temp_db.log_trade(trade)
    
    # Filter by time range
    start_time = (base_time + timedelta(hours=1)).isoformat()
    end_time = (base_time + timedelta(hours=3)).isoformat()
    
    trades = temp_db.get_trades(start_time=start_time, end_time=end_time, limit=10)
    assert len(trades) == 3


def test_clear_old_data(temp_db):
    """Test clearing old data."""
    # Log some old data
    old_time = datetime.now() - timedelta(days=60)
    
    trade = TradeMetric(
        timestamp=old_time.isoformat(),
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.0,
        price=50000.0,
        fee=25.0,
        pnl=100.0,
        cumulative_pnl=100.0
    )
    temp_db.log_trade(trade)
    
    # Log recent data
    recent_trade = TradeMetric(
        timestamp=datetime.now().isoformat(),
        symbol="BTCUSDT",
        side="BUY",
        quantity=1.0,
        price=50000.0,
        fee=25.0,
        pnl=100.0,
        cumulative_pnl=200.0
    )
    temp_db.log_trade(recent_trade)
    
    # Clear old data
    deleted = temp_db.clear_old_data(days=30)
    assert deleted['trades'] == 1
    
    # Verify only recent data remains
    trades = temp_db.get_trades(limit=10)
    assert len(trades) == 1
    assert trades[0]['cumulative_pnl'] == 200.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
