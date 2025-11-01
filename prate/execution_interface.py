"""
Abstract execution interface for exchange connectivity.

This module provides base classes for implementing exchange-specific
execution interfaces. It defines the contract that any exchange adapter
must fulfill.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .types import Side


class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    """Order type enumeration."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class MarginMode(Enum):
    """Margin mode for positions."""
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"


@dataclass
class Order:
    """Represents a trading order."""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: Side
    order_type: OrderType
    price: Optional[float]
    quantity: float
    filled_quantity: float
    status: OrderStatus
    timestamp: int
    update_time: int
    average_price: Optional[float] = None
    
    @property
    def remaining_quantity(self) -> float:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: Side  # LONG = BUY, SHORT = SELL
    quantity: float
    entry_price: float
    mark_price: float
    liquidation_price: Optional[float]
    leverage: int
    margin_mode: MarginMode
    unrealized_pnl: float
    margin: float
    timestamp: int


@dataclass
class Balance:
    """Represents account balance."""
    asset: str  # e.g., "USDT"
    available: float
    frozen: float  # locked in orders
    position_margin: float  # used as margin for positions
    timestamp: int
    
    @property
    def total(self) -> float:
        """Total balance."""
        return self.available + self.frozen + self.position_margin


@dataclass
class Trade:
    """Represents an executed trade."""
    trade_id: str
    order_id: str
    symbol: str
    side: Side
    price: float
    quantity: float
    fee: float
    fee_asset: str
    timestamp: int
    is_maker: bool


class ExecutionInterface(ABC):
    """
    Abstract base class for exchange execution interfaces.
    
    This defines the contract that all exchange-specific implementations
    must fulfill. It provides methods for:
    - Order management (create, cancel, query)
    - Position management
    - Balance queries
    - Market data streaming (via separate methods)
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the exchange.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to exchange.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    # Order Management
    
    @abstractmethod
    def create_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        leverage: Optional[int] = None,
        margin_mode: Optional[MarginMode] = None,
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> Optional[Order]:
        """
        Create a new order.
        
        Args:
            symbol: Trading pair (e.g., "BTC_USDT")
            side: Order side (BUY/SELL)
            order_type: Type of order
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
            leverage: Position leverage
            margin_mode: Isolated or cross margin
            client_order_id: Custom order ID
            **kwargs: Additional exchange-specific parameters
            
        Returns:
            Order object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> bool:
        """
        Cancel an existing order.
        
        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def cancel_all_orders(
        self,
        symbol: Optional[str] = None
    ) -> int:
        """
        Cancel all open orders.
        
        Args:
            symbol: Optional symbol filter (cancels all if None)
            
        Returns:
            Number of orders cancelled
        """
        pass
    
    @abstractmethod
    def get_order(
        self,
        symbol: str,
        order_id: str
    ) -> Optional[Order]:
        """
        Get order details.
        
        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            
        Returns:
            Order object if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open orders
        """
        pass
    
    # Position Management
    
    @abstractmethod
    def get_positions(
        self,
        symbol: Optional[str] = None
    ) -> List[Position]:
        """
        Get open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of open positions
        """
        pass
    
    @abstractmethod
    def get_position(
        self,
        symbol: str
    ) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Position if exists, None otherwise
        """
        pass
    
    @abstractmethod
    def set_leverage(
        self,
        symbol: str,
        leverage: int
    ) -> bool:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading pair
            leverage: Leverage multiplier
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # Account Management
    
    @abstractmethod
    def get_balance(self) -> Optional[Balance]:
        """
        Get account balance.
        
        Returns:
            Balance object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """
        Get trade execution history.
        
        Args:
            symbol: Optional symbol filter
            limit: Maximum number of trades to retrieve
            
        Returns:
            List of executed trades
        """
        pass


class MarketDataInterface(ABC):
    """
    Abstract interface for market data streaming.
    
    This is separate from ExecutionInterface to allow for different
    implementations (e.g., WebSocket vs polling).
    """
    
    @abstractmethod
    def subscribe_ticker(
        self,
        symbol: str,
        callback: callable
    ) -> bool:
        """
        Subscribe to ticker updates.
        
        Args:
            symbol: Trading pair
            callback: Function to call with ticker data
            
        Returns:
            True if subscription successful
        """
        pass
    
    @abstractmethod
    def subscribe_orderbook(
        self,
        symbol: str,
        callback: callable,
        depth: int = 20
    ) -> bool:
        """
        Subscribe to order book updates.
        
        Args:
            symbol: Trading pair
            callback: Function to call with order book data
            depth: Number of price levels
            
        Returns:
            True if subscription successful
        """
        pass
    
    @abstractmethod
    def subscribe_trades(
        self,
        symbol: str,
        callback: callable
    ) -> bool:
        """
        Subscribe to public trades.
        
        Args:
            symbol: Trading pair
            callback: Function to call with trade data
            
        Returns:
            True if subscription successful
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, channel: str) -> bool:
        """
        Unsubscribe from a channel.
        
        Args:
            channel: Channel identifier
            
        Returns:
            True if unsubscribed successfully
        """
        pass
