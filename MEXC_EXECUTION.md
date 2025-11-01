# MEXC Futures Execution Interface

This document describes the MEXC Futures execution interface implementation for PRATE.

## Overview

The MEXC Futures execution interface provides programmatic access to MEXC Futures exchange for:
- Order management (create, cancel, query)
- Position tracking
- Balance queries
- Market data streaming (future)

## Architecture

The implementation consists of three main components:

### 1. Abstract Base (`execution_interface.py`)

Defines the contract that all exchange implementations must fulfill:

```python
from prate.execution_interface import ExecutionInterface, Order, Position, Balance

class MyExchange(ExecutionInterface):
    def create_order(self, symbol, side, order_type, quantity, ...):
        # Implementation
        pass
```

**Key Classes:**
- `ExecutionInterface` - Abstract base for execution
- `MarketDataInterface` - Abstract base for market data
- `Order`, `Position`, `Balance`, `Trade` - Data models
- `OrderStatus`, `OrderType`, `Side`, `MarginMode` - Enumerations

### 2. MEXC Implementation (`mexc_futures.py`)

MEXC-specific implementation with three components:

**MEXCFuturesAuth**
- HMAC-SHA256 signature generation
- REST API header construction
- WebSocket authentication (future)

**MEXCFuturesREST**
- REST API client
- Authenticated request handling
- Error handling and retries

**MEXCFuturesExecution**
- Implements `ExecutionInterface`
- Order management
- Position queries
- Balance tracking

### 3. Candle Aggregator (`candle_aggregator.py`)

Handles high-frequency candle data:
- Converts 1m candles to 1s candles (interpolation)
- SQLite storage for 1s candles
- Aggregates to any timeframe (2s-600s)
- Multi-symbol support
- Efficient caching

## Usage Example

### Basic Setup

```python
from prate.mexc_futures import MEXCFuturesExecution
from prate.execution_interface import Side, OrderType, MarginMode

# Initialize with API credentials
exchange = MEXCFuturesExecution(
    api_key="your_api_key",
    secret_key="your_secret_key"
)

# Connect
if exchange.connect():
    print("Connected to MEXC Futures")
else:
    print("Connection failed")
```

### Creating Orders

```python
# Market buy order
order = exchange.create_order(
    symbol="BTC_USDT",
    side=Side.BUY,
    order_type=OrderType.MARKET,
    quantity=0.001,
    leverage=10,
    margin_mode=MarginMode.ISOLATED
)

if order:
    print(f"Order created: {order.order_id}")
    print(f"Status: {order.status}")
```

### Managing Positions

```python
# Get all positions
positions = exchange.get_positions()

for pos in positions:
    print(f"Symbol: {pos.symbol}")
    print(f"Size: {pos.quantity}")
    print(f"Entry: {pos.entry_price}")
    print(f"Mark: {pos.mark_price}")
    print(f"PnL: {pos.unrealized_pnl}")

# Get specific position
btc_position = exchange.get_position("BTC_USDT")
```

### Checking Balance

```python
balance = exchange.get_balance()
if balance:
    print(f"Available: {balance.available} USDT")
    print(f"Frozen: {balance.frozen} USDT")
    print(f"Position Margin: {balance.position_margin} USDT")
    print(f"Total: {balance.total} USDT")
```

### Order Management

```python
# Get open orders
open_orders = exchange.get_open_orders("BTC_USDT")

# Cancel specific order
exchange.cancel_order("BTC_USDT", "order_id_123")

# Cancel all orders for a symbol
count = exchange.cancel_all_orders("BTC_USDT")
print(f"Cancelled {count} orders")

# Cancel all orders on exchange
count = exchange.cancel_all_orders()
```

### Candle Aggregator

```python
from prate.candle_aggregator import CandleAggregator, CandleDatabase

# Initialize database
db = CandleDatabase("candles.db")
aggregator = CandleAggregator(db)

# Convert and store 1m candles
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

count = aggregator.process_and_store(candles_1m, 'BTC_USDT')
print(f"Stored {count} 1s candles")

# Retrieve and aggregate
candles_5s = aggregator.get_aggregated_candles(
    symbol='BTC_USDT',
    start_ts=1609459200000,
    end_ts=1609459260000,
    interval_seconds=5
)
```

## Security Considerations

### ⚠️ Important Warnings

1. **Unofficial Endpoints**: The MEXC order execution endpoints are marked as "under maintenance" in official docs. They may change without notice.

2. **No Support**: MEXC does not officially support these endpoints. Use at your own risk.

3. **API Changes**: Implement monitoring to detect API changes:
   ```python
   # Example kill switch logic
   consecutive_failures = 0
   
   while trading:
       order = exchange.create_order(...)
       if order is None:
           consecutive_failures += 1
           if consecutive_failures >= 5:
               # Kill switch triggered
               exchange.cancel_all_orders()
               notify_operator("API appears broken - halted trading")
               break
       else:
           consecutive_failures = 0
   ```

### Security Best Practices

1. **Store Credentials Securely**
   ```python
   import os
   
   api_key = os.environ['MEXC_API_KEY']
   secret_key = os.environ['MEXC_SECRET_KEY']
   ```

2. **IP Whitelisting**
   - Enable IP whitelisting in MEXC account settings
   - Bind API key to specific server IPs

3. **Minimal Permissions**
   - Only enable "Trade" and "Read" permissions
   - Never enable "Withdraw" for trading bots

4. **Rate Limiting**
   - MEXC allows 100 messages/second per connection
   - Implement backoff logic for errors

5. **Error Handling**
   ```python
   try:
       order = exchange.create_order(...)
       if not order:
           # Log error, don't crash
           logger.error("Order creation failed")
   except Exception as e:
       logger.exception("Unexpected error")
       # Implement recovery logic
   ```

## API Reference

### MEXC Side Codes
- `1` = Open Long
- `2` = Close Short
- `3` = Open Short
- `4` = Close Long

### MEXC Order Types
- `1` = Limit
- `3` = IOC (Immediate or Cancel)
- `4` = FOK (Fill or Kill)
- `5` = Market

### MEXC Margin Modes
- `1` = Isolated
- `2` = Cross

### MEXC Order States
- `1` = Pending/New
- `2` = Partially Filled
- `3` = Filled
- `4` = Cancelled

## Testing

Run the test suite:

```bash
# Test candle aggregator
python test_candle_aggregator.py

# Test execution interface
python test_execution_interface.py
```

Note: Live trading tests are skipped by default (require API credentials).

## Limitations

Current implementation does not include:
- WebSocket market data streaming
- Advanced order types (stop-loss, take-profit via params)
- Trade history retrieval
- Rate limiting enforcement
- Connection pooling
- Automatic reconnection

These features can be added as needed.

## References

- See `mexc.md` for complete MEXC API documentation
- MEXC Futures API: https://contract.mexc.com
- Official Docs: https://mexcdevelop.github.io/apidocs/contract_v1/

## License

See main LICENSE file. Use for educational/research purposes only.
Trading involves substantial risk of loss.
