"""
Simulator for backtesting with fill models.
"""

import random
from typing import Dict, List, Optional, Any
from .types import TradeIntent


class OrderBook:
    """Simulated order book with depth."""
    
    def __init__(self, initial_price: float = 50000.0, depth_levels: int = 10):
        self.mid_price = initial_price
        self.depth_levels = depth_levels
        self.bid_depth = {}  # price -> volume
        self.ask_depth = {}  # price -> volume
        self._generate_depth()
    
    def _generate_depth(self) -> None:
        """Generate realistic order book depth."""
        tick_size = self.mid_price * 0.0001
        
        # Generate bid side
        self.bid_depth = {}
        for i in range(self.depth_levels):
            price = self.mid_price - (i + 1) * tick_size
            # Volume decreases with distance from mid
            volume = 10.0 / (i + 1)
            self.bid_depth[price] = volume
        
        # Generate ask side
        self.ask_depth = {}
        for i in range(self.depth_levels):
            price = self.mid_price + (i + 1) * tick_size
            volume = 10.0 / (i + 1)
            self.ask_depth[price] = volume
    
    def update(self, new_mid: float) -> None:
        """Update book with new mid price."""
        self.mid_price = new_mid
        self._generate_depth()
    
    def get_best_bid(self) -> float:
        """Get best bid price."""
        return max(self.bid_depth.keys()) if self.bid_depth else self.mid_price * 0.9995
    
    def get_best_ask(self) -> float:
        """Get best ask price."""
        return min(self.ask_depth.keys()) if self.ask_depth else self.mid_price * 1.0005
    
    def get_available_volume(self, side: str, price: float) -> float:
        """Get available volume at price level."""
        if side == 'BUY':
            return self.ask_depth.get(price, 0.0)
        else:
            return self.bid_depth.get(price, 0.0)


class FillModel:
    """Realistic fill probability model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.limit_fill_prob = config.get('limit_fill_prob', 0.7)
        self.market_fill_prob = config.get('market_fill_prob', 0.95)
        self.partial_fill_prob = config.get('partial_fill_prob', 0.3)
        self.min_fill_ratio = config.get('min_fill_ratio', 0.5)
    
    def should_fill(self, intent: TradeIntent, current_price: float) -> bool:
        """Determine if order should fill."""
        if intent.price is None:
            # Market order
            return random.random() < self.market_fill_prob
        
        # Limit order - check if price is touched
        if intent.side.value == 'BUY':
            price_touched = current_price <= intent.price
        else:
            price_touched = current_price >= intent.price
        
        if not price_touched:
            return False
        
        return random.random() < self.limit_fill_prob
    
    def get_fill_ratio(self, intent: TradeIntent, available_volume: float) -> float:
        """Determine what fraction of order fills."""
        if intent.qty <= available_volume:
            return 1.0
        
        # Partial fill scenario
        if random.random() < self.partial_fill_prob:
            return max(self.min_fill_ratio, available_volume / intent.qty)
        
        return 1.0


class LatencyModel:
    """Simulates network and exchange latency."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.mean_latency_ms = config.get('mean_latency_ms', 50)
        self.std_latency_ms = config.get('std_latency_ms', 20)
        self.min_latency_ms = config.get('min_latency_ms', 10)
    
    def get_latency(self) -> int:
        """Get random latency in milliseconds."""
        latency = max(
            self.min_latency_ms,
            int(random.gauss(self.mean_latency_ms, self.std_latency_ms))
        )
        return latency


class Simulator:
    """
    Backtester/Simulator with realistic fill models.
    
    Simulates order execution with:
    - Realistic fill probability models
    - Order book depth simulation
    - Perpetual futures funding rates
    - Latency injection
    - Partial fill support
    """
    
    def __init__(
        self,
        market_data: Any,
        fee_schedule: Dict[str, float],
        slippage_model: Optional[Dict[str, Any]] = None,
        fill_model_config: Optional[Dict[str, Any]] = None,
        latency_config: Optional[Dict[str, Any]] = None,
        funding_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize simulator.
        
        Args:
            market_data: Market data replay object
            fee_schedule: Fee rates (maker, taker)
            slippage_model: Slippage model parameters
            fill_model_config: Fill probability model config
            latency_config: Latency model config
            funding_config: Funding rate config
        """
        self.market_data = market_data
        self.fee_schedule = fee_schedule
        self.slippage_model = slippage_model or {}
        
        self.current_ts = 0
        self.position = 0.0
        self.equity = 100000.0
        self.pnl = 0.0
        self.avg_entry_price = 0.0
        
        self.pending_orders: Dict[str, Dict[str, Any]] = {}  # cid -> {intent, submit_ts}
        self.fills: List[Dict[str, Any]] = []
        
        # Enhanced models
        self.order_book = OrderBook()
        self.fill_model = FillModel(fill_model_config)
        self.latency_model = LatencyModel(latency_config)
        
        # Funding rate simulation
        funding_config = funding_config or {}
        self.funding_rate = funding_config.get('base_rate', 0.0001)
        self.funding_interval_hours = funding_config.get('interval_hours', 8)
        self.last_funding_ts = 0
        self.funding_pnl = 0.0
    
    def send(self, intent: TradeIntent) -> str:
        """
        Send trade intent (add to pending orders with latency).
        
        Args:
            intent: Trade intent
            
        Returns:
            Client order ID
        """
        # Add latency to order submission
        latency = self.latency_model.get_latency()
        submit_ts = self.current_ts + latency
        
        self.pending_orders[intent.client_id] = {
            'intent': intent,
            'submit_ts': submit_ts,
            'partial_filled_qty': 0.0
        }
        return intent.client_id
    
    def cancel(self, client_id: str) -> None:
        """
        Cancel pending order.
        
        Args:
            client_id: Client order ID
        """
        if client_id in self.pending_orders:
            del self.pending_orders[client_id]
    
    def poll(self) -> List[Dict[str, Any]]:
        """
        Poll for fills.
        
        Returns:
            List of fill events
        """
        fills = self.fills.copy()
        self.fills.clear()
        return fills
    
    def step(self, ts: int) -> None:
        """
        Process events up to timestamp.
        
        Args:
            ts: Target timestamp
        """
        self.current_ts = ts
        current_price = self._get_current_price()
        
        # Update order book
        self.order_book.update(current_price)
        
        # Apply funding if interval passed
        self._apply_funding(ts)
        
        # Process pending orders that have reached their submit time
        for cid, order_data in list(self.pending_orders.items()):
            intent = order_data['intent']
            submit_ts = order_data['submit_ts']
            partial_filled_qty = order_data['partial_filled_qty']
            
            # Skip if order hasn't been submitted yet (latency)
            if submit_ts > ts:
                continue
            
            # Check if order should fill
            if not self.fill_model.should_fill(intent, current_price):
                continue
            
            # Determine fill price
            if intent.price:
                fill_price = intent.price
            else:
                # Market order with slippage
                fill_price = current_price
                slippage_bps = self.slippage_model.get('market_bps', 5.0)
                slippage = fill_price * slippage_bps / 10000.0
                if intent.side.value == 'BUY':
                    fill_price += slippage
                else:
                    fill_price -= slippage
            
            # Determine available volume
            available_volume = self.order_book.get_available_volume(intent.side.value, fill_price)
            
            # Determine fill quantity (support partial fills)
            remaining_qty = intent.qty - partial_filled_qty
            fill_ratio = self.fill_model.get_fill_ratio(intent, available_volume)
            fill_qty = remaining_qty * fill_ratio
            
            if fill_qty <= 0:
                continue
            
            # Calculate fee
            fee_rate = self.fee_schedule.get('taker', 0.001)
            if intent.post_only:
                fee_rate = self.fee_schedule.get('maker', 0.0005)
            
            fee = abs(fill_qty * fill_price * fee_rate)
            
            # Update position
            qty_signed = fill_qty if intent.side.value == 'BUY' else -fill_qty
            old_position = self.position
            new_position = self.position + qty_signed
            
            # Calculate PnL
            if old_position * new_position < 0:
                # Position flip - close old and open new
                close_qty = -old_position
                open_qty = new_position
                
                # PnL from closing
                if old_position != 0:
                    close_pnl = close_qty * (fill_price - self.avg_entry_price)
                else:
                    close_pnl = 0.0
                
                pnl = close_pnl - fee
                self.avg_entry_price = fill_price if open_qty != 0 else 0.0
            elif abs(new_position) > abs(old_position):
                # Increasing position
                if old_position != 0:
                    total_value = old_position * self.avg_entry_price + qty_signed * fill_price
                    self.avg_entry_price = total_value / new_position if new_position != 0 else 0.0
                else:
                    self.avg_entry_price = fill_price
                pnl = -fee
            else:
                # Decreasing position (taking profit/loss)
                if old_position != 0:
                    pnl = qty_signed * (fill_price - self.avg_entry_price) - fee
                else:
                    pnl = -fee
            
            self.position = new_position
            self.pnl += pnl
            self.equity += pnl
            
            # Update partial fill tracking
            order_data['partial_filled_qty'] += fill_qty
            
            # Record fill
            fill = {
                'client_id': cid,
                'symbol': intent.symbol,
                'side': intent.side.value,
                'qty': fill_qty,
                'price': fill_price,
                'fee': fee,
                'pnl': pnl,
                'ts': ts,
                'is_partial': order_data['partial_filled_qty'] < intent.qty
            }
            self.fills.append(fill)
            
            # Remove from pending if fully filled
            if order_data['partial_filled_qty'] >= intent.qty:
                del self.pending_orders[cid]
    
    def _apply_funding(self, ts: int) -> None:
        """Apply perpetual futures funding rate."""
        if self.last_funding_ts == 0:
            self.last_funding_ts = ts
            return
        
        # Check if funding interval has passed
        interval_ms = self.funding_interval_hours * 60 * 60 * 1000
        if ts - self.last_funding_ts < interval_ms:
            return
        
        # Apply funding
        if self.position != 0:
            current_price = self._get_current_price()
            position_value = abs(self.position) * current_price
            funding = position_value * self.funding_rate
            
            # Long pays funding (negative), short receives (positive)
            if self.position > 0:
                funding = -funding
            
            self.funding_pnl += funding
            self.pnl += funding
            self.equity += funding
        
        self.last_funding_ts = ts
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        # Placeholder: return fixed price
        # In real implementation, query market_data
        return 50000.0
    
    def get_account_state(self) -> Dict[str, float]:
        """Get current account state."""
        current_price = self._get_current_price()
        unrealized_pnl = 0.0
        
        if self.position != 0 and self.avg_entry_price != 0:
            unrealized_pnl = self.position * (current_price - self.avg_entry_price)
        
        return {
            'equity': self.equity,
            'position': self.position,
            'pnl': self.pnl,
            'price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'avg_entry_price': self.avg_entry_price,
            'funding_pnl': self.funding_pnl
        }
