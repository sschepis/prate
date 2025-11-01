"""
Type definitions for PRATE system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import numpy as np


class GuildID(Enum):
    """Guild identifiers for different trading strategies."""
    TF = "TF"    # Trend-follow
    MR = "MR"    # Mean-revert
    BR = "BR"    # Breakout
    LM = "LM"    # Liquidity make
    FA = "FA"    # Funding carry
    OBS = "OBS"  # Observation / explore


class RegimeID(Enum):
    """Market regime identifiers."""
    TREND = "TREND"
    RANGE = "RANGE"
    VOLX = "VOLX"      # Volatility expansion
    QUIET = "QUIET"
    UNKNOWN = "UNKNOWN"


class Side(Enum):
    """Trade side."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Observation:
    """Market observation with features."""
    ts: int  # timestamp in milliseconds
    symbol: str
    mid: float
    bid: float
    ask: float
    spread: float
    last_px: float
    last_qty: float
    vol_1s: float
    vol_1m: float
    book_imbalance: float
    pressure: float
    realized_var: float
    atr: float
    rsi_short: float
    ema_slope: float
    inventory: float
    equity: float
    unrealized_pnl: float
    funding_rate: float
    time_of_day_bucket: int
    regime_soft: Dict[RegimeID, float]
    features_vec: np.ndarray  # continuous features
    features_disc: Dict[str, int]  # discretized features for prime embedding


@dataclass
class Action:
    """Trading action."""
    style: GuildID
    delta_q: float  # target position delta
    params: Dict[str, float]  # style-specific parameters


@dataclass
class TradeIntent:
    """Trade intent to be executed."""
    symbol: str
    side: Side
    qty: float
    price: Optional[float]
    tif: str  # time in force
    post_only: bool
    client_id: str
    meta: Dict[str, any]


@dataclass
class Basis:
    """Basis subset of primes."""
    id: str
    primes: list[int]
