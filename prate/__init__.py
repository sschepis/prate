"""
PRATE - Prime-Resonant Adaptive Trading Ecology

A sophisticated adaptive trading system using prime-indexed Hilbert space embeddings,
entropy minimization, and holographic memory.
"""

__version__ = "0.2.0"
__author__ = "PRATE Contributors"

from .types import (
    GuildID,
    RegimeID,
    Side,
    Observation,
    Action,
    TradeIntent,
)

from .execution_interface import (
    ExecutionInterface,
    MarketDataInterface,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Balance,
    Trade,
    MarginMode,
)

from .candle_aggregator import (
    CandleAggregator,
    CandleDatabase,
    Candle1s,
)

__all__ = [
    # Core types
    "GuildID",
    "RegimeID",
    "Side",
    "Observation",
    "Action",
    "TradeIntent",
    # Execution interface
    "ExecutionInterface",
    "MarketDataInterface",
    "Order",
    "OrderStatus",
    "OrderType",
    "Position",
    "Balance",
    "Trade",
    "MarginMode",
    # Candle aggregator
    "CandleAggregator",
    "CandleDatabase",
    "Candle1s",
]
