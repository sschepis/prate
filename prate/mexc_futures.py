"""
MEXC Futures execution interface implementation.

This module implements the execution interface specifically for MEXC Futures,
following the API specifications from mexc.md.

WARNING: This implementation uses unofficial/undocumented endpoints for
order execution (marked as "under maintenance" in official docs). Use with
appropriate risk management and monitoring.
"""

import time
import hmac
import hashlib
import json
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urlencode
import requests

from .execution_interface import (
    ExecutionInterface,
    MarketDataInterface,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Balance,
    Trade,
    Side,
    MarginMode
)


class MEXCFuturesAuth:
    """
    MEXC Futures authentication helper.
    
    Handles signature generation for both REST and WebSocket authentication.
    """
    
    def __init__(self, api_key: str, secret_key: str):
        """
        Initialize authenticator.
        
        Args:
            api_key: MEXC API key (Access Key)
            secret_key: MEXC Secret key
        """
        self.api_key = api_key
        self.secret_key = secret_key
    
    def generate_signature(self, timestamp: int, params: str = "") -> str:
        """
        Generate HMAC-SHA256 signature.
        
        Args:
            timestamp: Unix timestamp in milliseconds
            params: Parameters string (for REST) or empty (for WebSocket)
            
        Returns:
            Hex signature string
        """
        message = f"{self.api_key}{timestamp}{params}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def get_headers(self, timestamp: int, request_body: str = "") -> Dict[str, str]:
        """
        Generate headers for authenticated REST request.
        
        Args:
            timestamp: Request timestamp in milliseconds
            request_body: JSON request body (for POST) or query string (for GET)
            
        Returns:
            Dictionary of headers
        """
        signature = self.generate_signature(timestamp, request_body)
        
        return {
            "ApiKey": self.api_key,
            "Request-Time": str(timestamp),
            "Signature": signature,
            "Content-Type": "application/json"
        }


class MEXCFuturesREST:
    """
    MEXC Futures REST API client.
    
    Handles all REST API interactions including order management,
    position queries, and balance retrieval.
    """
    
    BASE_URL = "https://contract.mexc.com"
    
    def __init__(self, auth: MEXCFuturesAuth, timeout: int = 10):
        """
        Initialize REST client.
        
        Args:
            auth: Authentication helper
            timeout: Request timeout in seconds
        """
        self.auth = auth
        self.timeout = timeout
        self.session = requests.Session()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = True
    ) -> Optional[Dict]:
        """
        Make HTTP request to MEXC API.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            params: Request parameters
            signed: Whether request requires authentication
            
        Returns:
            Response JSON or None on error
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        
        try:
            if method == "GET":
                if signed:
                    timestamp = self._get_timestamp()
                    # Sort params for signature
                    sorted_params = sorted(params.items())
                    query_string = urlencode(sorted_params)
                    headers = self.auth.get_headers(timestamp, query_string)
                    
                    response = self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                else:
                    response = self.session.get(url, params=params, timeout=self.timeout)
            
            elif method == "POST":
                timestamp = self._get_timestamp()
                json_body = json.dumps(params)
                headers = self.auth.get_headers(timestamp, json_body)
                
                response = self.session.post(
                    url,
                    data=json_body,
                    headers=headers,
                    timeout=self.timeout
                )
            
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"MEXC API request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in MEXC API request: {e}")
            return None
    
    def create_order(
        self,
        symbol: str,
        side: int,  # 1=OpenLong, 2=CloseShort, 3=OpenShort, 4=CloseLong
        order_type: int,  # 1=Limit, 3=IOC, 4=FOK, 5=Market
        quantity: float,
        price: float,
        leverage: int,
        open_type: int,  # 1=Isolated, 2=Cross
        external_oid: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create new order via MEXC REST API.
        
        WARNING: Uses unofficial endpoint that may change without notice.
        
        Args:
            symbol: Trading pair (e.g., "BTC_USDT")
            side: 1=OpenLong, 2=CloseShort, 3=OpenShort, 4=CloseLong
            order_type: 1=Limit, 3=IOC, 4=FOK, 5=Market
            quantity: Order quantity
            price: Order price (required even for market orders)
            leverage: Position leverage
            open_type: 1=Isolated, 2=Cross
            external_oid: Optional client order ID
            
        Returns:
            API response or None on error
        """
        params = {
            "symbol": symbol,
            "price": price,
            "vol": quantity,
            "side": side,
            "type": order_type,
            "openType": open_type,
            "leverage": leverage
        }
        
        if external_oid:
            params["externalOid"] = external_oid
        
        return self._request("POST", "/api/v1/private/order/submit", params)
    
    def cancel_order(self, order_ids: List[str]) -> Optional[Dict]:
        """
        Cancel one or more orders.
        
        Args:
            order_ids: List of order IDs to cancel
            
        Returns:
            API response or None on error
        """
        params = {"orderIds": order_ids}
        return self._request("POST", "/api/v1/private/order/cancel", params)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """
        Cancel all open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            API response or None on error
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        return self._request("POST", "/api/v1/private/order/cancel_all", params)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """
        Get open orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            API response or None on error
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        return self._request("GET", "/api/v1/private/order/list/open_orders", params)
    
    def get_open_positions(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """
        Get open positions.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            API response or None on error
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        
        return self._request("GET", "/api/v1/private/position/open_positions", params)
    
    def change_leverage(self, position_id: int, leverage: int) -> Optional[Dict]:
        """
        Change position leverage.
        
        Args:
            position_id: Position ID
            leverage: New leverage value
            
        Returns:
            API response or None on error
        """
        params = {
            "positionId": position_id,
            "leverage": leverage
        }
        return self._request("POST", "/api/v1/private/position/change_leverage", params)
    
    def get_assets(self) -> Optional[Dict]:
        """
        Get account assets/balance.
        
        Returns:
            API response or None on error
        """
        return self._request("GET", "/api/v1/private/account/assets", {})


class MEXCFuturesExecution(ExecutionInterface):
    """
    MEXC Futures execution implementation.
    
    Implements the ExecutionInterface for MEXC Futures trading.
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = False
    ):
        """
        Initialize MEXC Futures execution interface.
        
        Args:
            api_key: MEXC API key
            secret_key: MEXC secret key
            testnet: Use testnet (if available)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        
        self.auth = MEXCFuturesAuth(api_key, secret_key)
        self.rest_client = MEXCFuturesREST(self.auth)
        self._connected = False
    
    def connect(self) -> bool:
        """Establish connection (verify credentials)."""
        try:
            # Test connection by getting balance
            result = self.rest_client.get_assets()
            if result is not None:
                self._connected = True
                return True
            return False
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from exchange."""
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    def _convert_side_to_mexc(
        self,
        side: Side,
        is_close: bool = False
    ) -> int:
        """
        Convert Side enum to MEXC side integer.
        
        Args:
            side: Side enum (BUY/SELL)
            is_close: Whether this is closing a position
            
        Returns:
            MEXC side integer (1=OpenLong, 2=CloseShort, 3=OpenShort, 4=CloseLong)
        """
        if side == Side.BUY:
            return 4 if is_close else 1  # CloseLong : OpenLong
        else:
            return 2 if is_close else 3  # CloseShort : OpenShort
    
    def _convert_order_type_to_mexc(self, order_type: OrderType) -> int:
        """
        Convert OrderType enum to MEXC type integer.
        
        Args:
            order_type: OrderType enum
            
        Returns:
            MEXC type integer (1=Limit, 3=IOC, 4=FOK, 5=Market)
        """
        mapping = {
            OrderType.LIMIT: 1,
            OrderType.MARKET: 5,
            OrderType.IOC: 3,
            OrderType.FOK: 4
        }
        return mapping.get(order_type, 1)
    
    def _convert_margin_mode_to_mexc(self, margin_mode: MarginMode) -> int:
        """Convert MarginMode to MEXC openType."""
        return 1 if margin_mode == MarginMode.ISOLATED else 2
    
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
        """Create a new order on MEXC."""
        if not self._connected:
            print("Not connected to MEXC")
            return None
        
        # Set defaults
        leverage = leverage or 10
        margin_mode = margin_mode or MarginMode.ISOLATED
        
        # For market orders, use current market price as placeholder
        if order_type == OrderType.MARKET and price is None:
            # In production, should fetch current market price
            # For now, using a placeholder
            price = 50000.0  # This should be fetched from market data
        
        mexc_side = self._convert_side_to_mexc(side, kwargs.get('is_close', False))
        mexc_type = self._convert_order_type_to_mexc(order_type)
        mexc_open_type = self._convert_margin_mode_to_mexc(margin_mode)
        
        result = self.rest_client.create_order(
            symbol=symbol,
            side=mexc_side,
            order_type=mexc_type,
            quantity=quantity,
            price=price,
            leverage=leverage,
            open_type=mexc_open_type,
            external_oid=client_order_id
        )
        
        if result and result.get('success'):
            data = result.get('data', {})
            return Order(
                order_id=str(data.get('orderId', '')),
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                filled_quantity=0.0,
                status=OrderStatus.NEW,
                timestamp=int(time.time() * 1000),
                update_time=int(time.time() * 1000)
            )
        
        return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        if not self._connected:
            return False
        
        result = self.rest_client.cancel_order([order_id])
        return result is not None and result.get('success', False)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        if not self._connected:
            return 0
        
        result = self.rest_client.cancel_all_orders(symbol)
        if result and result.get('success'):
            # Return count of cancelled orders
            return result.get('data', {}).get('cancelledCount', 0)
        return 0
    
    def get_order(self, symbol: str, order_id: str) -> Optional[Order]:
        """Get order details."""
        # MEXC doesn't have a direct get_order endpoint
        # We need to get all open orders and filter
        orders = self.get_open_orders(symbol)
        for order in orders:
            if order.order_id == order_id:
                return order
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        if not self._connected:
            return []
        
        result = self.rest_client.get_open_orders(symbol)
        if not result or not result.get('success'):
            return []
        
        orders = []
        data = result.get('data', [])
        
        for order_data in data:
            # Parse MEXC order response
            orders.append(Order(
                order_id=str(order_data.get('orderId', '')),
                client_order_id=order_data.get('externalOid'),
                symbol=order_data.get('symbol', ''),
                side=Side.BUY if order_data.get('side') in [1, 4] else Side.SELL,
                order_type=self._parse_order_type(order_data.get('type')),
                price=float(order_data.get('price', 0)),
                quantity=float(order_data.get('vol', 0)),
                filled_quantity=float(order_data.get('dealVol', 0)),
                status=self._parse_order_status(order_data.get('state')),
                timestamp=int(order_data.get('createTime', 0)),
                update_time=int(order_data.get('updateTime', 0)),
                average_price=float(order_data.get('dealAvgPrice', 0)) if order_data.get('dealAvgPrice') else None
            ))
        
        return orders
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get open positions."""
        if not self._connected:
            return []
        
        result = self.rest_client.get_open_positions(symbol)
        if not result or not result.get('success'):
            return []
        
        positions = []
        data = result.get('data', [])
        
        for pos_data in data:
            positions.append(Position(
                symbol=pos_data.get('symbol', ''),
                side=Side.BUY if pos_data.get('positionType') == 1 else Side.SELL,
                quantity=abs(float(pos_data.get('holdVol', 0))),
                entry_price=float(pos_data.get('openAvgPrice', 0)),
                mark_price=float(pos_data.get('fairPrice', 0)),
                liquidation_price=float(pos_data.get('liqPrice')) if pos_data.get('liqPrice') else None,
                leverage=int(pos_data.get('leverage', 1)),
                margin_mode=MarginMode.ISOLATED if pos_data.get('openType') == 1 else MarginMode.CROSS,
                unrealized_pnl=float(pos_data.get('unrealisedPnl', 0)),
                margin=float(pos_data.get('im', 0)),
                timestamp=int(time.time() * 1000)
            ))
        
        return positions
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions(symbol)
        return positions[0] if positions else None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if not self._connected:
            return False
        
        # Need to get position_id first
        position = self.get_position(symbol)
        if not position:
            print(f"No position found for {symbol}")
            return False
        
        # This is a simplified implementation
        # In reality, need to extract position_id from MEXC response
        # For now, returning False as we need position_id
        print("set_leverage requires position_id from MEXC - not fully implemented")
        return False
    
    def get_balance(self) -> Optional[Balance]:
        """Get account balance."""
        if not self._connected:
            return None
        
        result = self.rest_client.get_assets()
        if not result or not result.get('success'):
            return None
        
        data = result.get('data', {})
        
        return Balance(
            asset="USDT",  # MEXC futures typically use USDT
            available=float(data.get('availableBalance', 0)),
            frozen=float(data.get('frozenBalance', 0)),
            position_margin=float(data.get('positionMargin', 0)),
            timestamp=int(time.time() * 1000)
        )
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get trade execution history."""
        # This would require additional MEXC API endpoints
        # Not implemented in this basic version
        print("get_trade_history not implemented for MEXC")
        return []
    
    def _parse_order_type(self, mexc_type: int) -> OrderType:
        """Parse MEXC order type to OrderType enum."""
        mapping = {
            1: OrderType.LIMIT,
            3: OrderType.IOC,
            4: OrderType.FOK,
            5: OrderType.MARKET
        }
        return mapping.get(mexc_type, OrderType.LIMIT)
    
    def _parse_order_status(self, mexc_state: int) -> OrderStatus:
        """Parse MEXC order state to OrderStatus enum."""
        # MEXC states: 1=Pending, 2=PartialFilled, 3=Filled, 4=Cancelled
        mapping = {
            1: OrderStatus.NEW,
            2: OrderStatus.PARTIALLY_FILLED,
            3: OrderStatus.FILLED,
            4: OrderStatus.CANCELLED
        }
        return mapping.get(mexc_state, OrderStatus.NEW)
