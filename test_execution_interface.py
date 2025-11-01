#!/usr/bin/env python3
"""
Tests for execution interface and MEXC implementation.
"""

import time
from prate.execution_interface import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    Balance,
    Trade,
    Side,
    MarginMode
)
from prate.mexc_futures import MEXCFuturesAuth


def test_order_dataclass():
    """Test Order dataclass."""
    print("Testing Order dataclass...")
    
    order = Order(
        order_id="123456",
        client_order_id="my-order-1",
        symbol="BTC_USDT",
        side=Side.BUY,
        order_type=OrderType.LIMIT,
        price=50000.0,
        quantity=0.1,
        filled_quantity=0.05,
        status=OrderStatus.PARTIALLY_FILLED,
        timestamp=int(time.time() * 1000),
        update_time=int(time.time() * 1000),
        average_price=49900.0
    )
    
    # Test properties
    assert order.remaining_quantity == 0.05, "Remaining quantity should be 0.05"
    assert order.is_active, "Order should be active"
    
    # Test filled order
    filled_order = Order(
        order_id="789",
        client_order_id=None,
        symbol="ETH_USDT",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        price=None,
        quantity=1.0,
        filled_quantity=1.0,
        status=OrderStatus.FILLED,
        timestamp=int(time.time() * 1000),
        update_time=int(time.time() * 1000)
    )
    
    assert filled_order.remaining_quantity == 0.0, "Filled order should have 0 remaining"
    assert not filled_order.is_active, "Filled order should not be active"
    
    print("  ✓ Order dataclass working correctly")


def test_position_dataclass():
    """Test Position dataclass."""
    print("\nTesting Position dataclass...")
    
    position = Position(
        symbol="BTC_USDT",
        side=Side.BUY,
        quantity=0.5,
        entry_price=48000.0,
        mark_price=50000.0,
        liquidation_price=35000.0,
        leverage=10,
        margin_mode=MarginMode.ISOLATED,
        unrealized_pnl=1000.0,
        margin=2400.0,
        timestamp=int(time.time() * 1000)
    )
    
    assert position.symbol == "BTC_USDT", "Symbol mismatch"
    assert position.side == Side.BUY, "Side should be BUY"
    assert position.leverage == 10, "Leverage should be 10"
    assert position.unrealized_pnl == 1000.0, "Unrealized PnL should be 1000.0"
    
    print("  ✓ Position dataclass working correctly")


def test_balance_dataclass():
    """Test Balance dataclass."""
    print("\nTesting Balance dataclass...")
    
    balance = Balance(
        asset="USDT",
        available=10000.0,
        frozen=500.0,
        position_margin=2000.0,
        timestamp=int(time.time() * 1000)
    )
    
    assert balance.total == 12500.0, "Total should be 12500.0"
    assert balance.available == 10000.0, "Available should be 10000.0"
    
    print("  ✓ Balance dataclass working correctly")
    print(f"    - Available: {balance.available}")
    print(f"    - Frozen: {balance.frozen}")
    print(f"    - Position margin: {balance.position_margin}")
    print(f"    - Total: {balance.total}")


def test_mexc_auth_signature():
    """Test MEXC authentication signature generation."""
    print("\nTesting MEXC Auth Signature...")
    
    # Test with dummy credentials
    api_key = "test_api_key"
    secret_key = "test_secret_key"
    
    auth = MEXCFuturesAuth(api_key, secret_key)
    
    # Test signature generation
    timestamp = 1609459200000
    signature = auth.generate_signature(timestamp, "")
    
    assert isinstance(signature, str), "Signature should be a string"
    assert len(signature) == 64, "HMAC-SHA256 should produce 64 hex chars"
    
    # Test consistency
    signature2 = auth.generate_signature(timestamp, "")
    assert signature == signature2, "Same inputs should produce same signature"
    
    # Test with params
    signature3 = auth.generate_signature(timestamp, "symbol=BTC_USDT")
    assert signature3 != signature, "Different params should produce different signature"
    
    print("  ✓ MEXC authentication working correctly")
    print(f"    - Signature length: {len(signature)} chars")
    print(f"    - Signature format: hex string")


def test_mexc_auth_headers():
    """Test MEXC REST API header generation."""
    print("\nTesting MEXC Auth Headers...")
    
    api_key = "test_api_key"
    secret_key = "test_secret_key"
    
    auth = MEXCFuturesAuth(api_key, secret_key)
    
    timestamp = 1609459200000
    headers = auth.get_headers(timestamp, "")
    
    # Check required headers
    assert "ApiKey" in headers, "ApiKey header missing"
    assert "Request-Time" in headers, "Request-Time header missing"
    assert "Signature" in headers, "Signature header missing"
    assert "Content-Type" in headers, "Content-Type header missing"
    
    assert headers["ApiKey"] == api_key, "ApiKey mismatch"
    assert headers["Request-Time"] == str(timestamp), "Request-Time mismatch"
    assert headers["Content-Type"] == "application/json", "Content-Type mismatch"
    
    print("  ✓ MEXC headers generated correctly")
    print(f"    - Headers: {list(headers.keys())}")


def test_order_status_enum():
    """Test OrderStatus enum."""
    print("\nTesting OrderStatus enum...")
    
    statuses = [
        OrderStatus.NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED
    ]
    
    assert len(statuses) == 6, "Should have 6 order statuses"
    assert OrderStatus.NEW.value == "NEW", "NEW status value mismatch"
    
    print("  ✓ OrderStatus enum working correctly")
    print(f"    - Available statuses: {[s.value for s in statuses]}")


def test_order_type_enum():
    """Test OrderType enum."""
    print("\nTesting OrderType enum...")
    
    types = [
        OrderType.LIMIT,
        OrderType.MARKET,
        OrderType.IOC,
        OrderType.FOK
    ]
    
    assert len(types) == 4, "Should have 4 order types"
    assert OrderType.MARKET.value == "MARKET", "MARKET type value mismatch"
    
    print("  ✓ OrderType enum working correctly")
    print(f"    - Available types: {[t.value for t in types]}")


def test_side_enum():
    """Test Side enum."""
    print("\nTesting Side enum...")
    
    assert Side.BUY.value == "BUY", "BUY value mismatch"
    assert Side.SELL.value == "SELL", "SELL value mismatch"
    
    print("  ✓ Side enum working correctly")


def test_margin_mode_enum():
    """Test MarginMode enum."""
    print("\nTesting MarginMode enum...")
    
    assert MarginMode.ISOLATED.value == "ISOLATED", "ISOLATED value mismatch"
    assert MarginMode.CROSS.value == "CROSS", "CROSS value mismatch"
    
    print("  ✓ MarginMode enum working correctly")


def test_trade_dataclass():
    """Test Trade dataclass."""
    print("\nTesting Trade dataclass...")
    
    trade = Trade(
        trade_id="trade123",
        order_id="order456",
        symbol="BTC_USDT",
        side=Side.BUY,
        price=50000.0,
        quantity=0.1,
        fee=5.0,
        fee_asset="USDT",
        timestamp=int(time.time() * 1000),
        is_maker=True
    )
    
    assert trade.symbol == "BTC_USDT", "Symbol mismatch"
    assert trade.is_maker, "Should be maker trade"
    assert trade.fee == 5.0, "Fee mismatch"
    
    print("  ✓ Trade dataclass working correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("PRATE Execution Interface Tests")
    print("=" * 60)
    
    # Test dataclasses
    test_order_dataclass()
    test_position_dataclass()
    test_balance_dataclass()
    test_trade_dataclass()
    
    # Test enums
    test_order_status_enum()
    test_order_type_enum()
    test_side_enum()
    test_margin_mode_enum()
    
    # Test MEXC-specific functionality
    test_mexc_auth_signature()
    test_mexc_auth_headers()
    
    print("\n" + "=" * 60)
    print("✓ All execution interface tests passed successfully!")
    print("=" * 60)
    print("\nNOTE: MEXC live trading tests skipped (require API credentials)")
    print("      These tests validate the interface structure and auth logic.")
