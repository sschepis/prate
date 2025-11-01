"""
Simulator for backtesting with fill models.
"""

from typing import Dict, List, Optional, Any
from .types import TradeIntent


class Simulator:
    """
    Backtester/Simulator with simple fill models.
    
    Simulates order execution with:
    - Limit order fills based on price touches
    - Market order fills with slippage
    - Fee application
    - Perpetual funding (optional)
    """
    
    def __init__(
        self,
        market_data: Any,
        fee_schedule: Dict[str, float],
        slippage_model: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize simulator.
        
        Args:
            market_data: Market data replay object
            fee_schedule: Fee rates (maker, taker)
            slippage_model: Slippage model parameters
        """
        self.market_data = market_data
        self.fee_schedule = fee_schedule
        self.slippage_model = slippage_model or {}
        
        self.current_ts = 0
        self.position = 0.0
        self.equity = 100000.0
        self.pnl = 0.0
        
        self.pending_orders: Dict[str, TradeIntent] = {}
        self.fills: List[Dict[str, Any]] = []
    
    def send(self, intent: TradeIntent) -> str:
        """
        Send trade intent (add to pending orders).
        
        Args:
            intent: Trade intent
            
        Returns:
            Client order ID
        """
        self.pending_orders[intent.client_id] = intent
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
        
        # Simple fill logic: assume all limit orders fill at current price
        # In a real implementation, check if price touched limit price
        
        current_price = self._get_current_price()
        
        for cid, intent in list(self.pending_orders.items()):
            # Simplified: fill all orders
            fill_price = intent.price if intent.price else current_price
            
            # Apply slippage for market orders
            if not intent.price:
                slippage_bps = self.slippage_model.get('market_bps', 5.0)
                slippage = fill_price * slippage_bps / 10000.0
                if intent.side.value == 'BUY':
                    fill_price += slippage
                else:
                    fill_price -= slippage
            
            # Calculate fee
            fee_rate = self.fee_schedule.get('taker', 0.001)
            if intent.post_only:
                fee_rate = self.fee_schedule.get('maker', 0.0005)
            
            fee = abs(intent.qty * fill_price * fee_rate)
            
            # Update position
            qty_signed = intent.qty if intent.side.value == 'BUY' else -intent.qty
            old_position = self.position
            self.position += qty_signed
            
            # Calculate PnL for position change
            if old_position != 0:
                # Simplified PnL calculation
                pnl = -qty_signed * fill_price  # Negative because buying costs money
            else:
                pnl = 0.0
            
            pnl -= fee
            self.pnl += pnl
            self.equity += pnl
            
            # Record fill
            fill = {
                'client_id': cid,
                'symbol': intent.symbol,
                'side': intent.side.value,
                'qty': intent.qty,
                'price': fill_price,
                'fee': fee,
                'pnl': pnl,
                'ts': ts
            }
            self.fills.append(fill)
            
            # Remove from pending
            del self.pending_orders[cid]
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        # Placeholder: return fixed price
        # In real implementation, query market_data
        return 50000.0
    
    def get_account_state(self) -> Dict[str, float]:
        """Get current account state."""
        return {
            'equity': self.equity,
            'position': self.position,
            'pnl': self.pnl,
            'price': self._get_current_price()
        }
