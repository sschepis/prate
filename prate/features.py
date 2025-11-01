"""
Feature Engine for computing technical and microstructure features.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any
from .types import Observation, RegimeID


class RollingBuffers:
    """Maintains rolling windows for feature computation."""
    
    def __init__(self, windows: Dict[str, int]):
        self.windows = windows
        self.buffers = {name: [] for name in windows}
    
    def update(self, tick: Dict[str, Any]) -> None:
        """Update buffers with new tick data."""
        for name, max_len in self.windows.items():
            if name in tick:
                self.buffers[name].append(tick[name])
                if len(self.buffers[name]) > max_len:
                    self.buffers[name].pop(0)
    
    def get(self, name: str) -> list:
        """Get buffer by name."""
        return self.buffers.get(name, [])


class Discretizer:
    """Discretizes continuous values into integer buckets."""
    
    def __init__(self, bins: int, min_val: float, max_val: float):
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
    
    def bin(self, value: float) -> int:
        """Convert value to bin index."""
        if value <= self.min_val:
            return 0
        if value >= self.max_val:
            return self.bins - 1
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        return int(normalized * self.bins)


def make_discretizers(binning_specs: Dict[str, Dict[str, Any]]) -> Dict[str, Discretizer]:
    """Create discretizers from specifications."""
    discretizers = {}
    for name, spec in binning_specs.items():
        discretizers[name] = Discretizer(
            bins=spec['bins'],
            min_val=spec['min'],
            max_val=spec['max']
        )
    return discretizers


def compute_ema(values: list, period: int) -> float:
    """Compute Exponential Moving Average."""
    if len(values) < period:
        # For insufficient data, return simple moving average as EMA approximation
        return np.mean(values) if values else np.nan
    
    alpha = 2.0 / (period + 1)
    # Initialize EMA with SMA of first 'period' values
    ema = np.mean(values[:period])
    for val in values[period:]:
        ema = alpha * val + (1 - alpha) * ema
    return ema


def compute_rsi(prices: list, period: int = 14) -> float:
    """Compute Relative Strength Index."""
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI
    
    changes = np.diff(prices[-period-1:])
    gains = np.maximum(changes, 0)
    losses = np.maximum(-changes, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """Compute Average True Range."""
    if len(closes) < period + 1:
        if highs and lows:
            return np.mean([h - l for h, l in zip(highs, lows)])
        return 0.0
    
    true_ranges = []
    for i in range(1, len(closes)):
        high = highs[i] if i < len(highs) else closes[i]
        low = lows[i] if i < len(lows) else closes[i]
        prev_close = closes[i-1]
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)
    
    return np.mean(true_ranges[-period:]) if true_ranges else 0.0


def compute_bollinger_bands(prices: list, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """Compute Bollinger Bands."""
    if len(prices) < period:
        price = prices[-1] if prices else 0.0
        return {'bb_upper': price, 'bb_middle': price, 'bb_lower': price, 'bb_width': 0.0}
    
    recent_prices = prices[-period:]
    middle = np.mean(recent_prices)
    std = np.std(recent_prices)
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle if middle != 0 else 0.0
    
    return {
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_width': width
    }


def compute_realized_variance(returns: list, period: int = 20) -> float:
    """Compute realized variance from returns."""
    if len(returns) < 2:
        return 0.0
    
    recent_returns = returns[-period:] if len(returns) > period else returns
    return np.var(recent_returns)


def compute_pressure(buys: list, sells: list, period: int = 10) -> float:
    """Compute buying/selling pressure from volume data."""
    if not buys or not sells:
        return 0.0
    
    recent_buys = buys[-period:] if len(buys) > period else buys
    recent_sells = sells[-period:] if len(sells) > period else sells
    
    total_buy = sum(recent_buys)
    total_sell = sum(recent_sells)
    total = total_buy + total_sell
    
    if total == 0:
        return 0.0
    
    return (total_buy - total_sell) / total


def compute_features(buffers: RollingBuffers) -> Dict[str, float]:
    """Compute features from rolling buffers."""
    features = {}
    
    # Basic return
    prices = buffers.get('price')
    if len(prices) > 1:
        features['ret_1s'] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0.0
    else:
        features['ret_1s'] = 0.0
    
    # EMA indicators
    features['ema_fast'] = compute_ema(prices, period=12)
    features['ema_slow'] = compute_ema(prices, period=26)
    features['ema_slope'] = (features['ema_fast'] - features['ema_slow']) / features['ema_slow'] if features['ema_slow'] != 0 else 0.0
    
    # RSI
    features['rsi'] = compute_rsi(prices, period=14)
    
    # ATR
    highs = buffers.get('high')
    lows = buffers.get('low')
    features['atr'] = compute_atr(highs, lows, prices, period=14)
    
    # Bollinger Bands
    bb = compute_bollinger_bands(prices, period=20)
    features.update(bb)
    
    # Bollinger Band position
    if prices and bb['bb_upper'] != bb['bb_lower']:
        features['bb_position'] = (prices[-1] - bb['bb_lower']) / (bb['bb_upper'] - bb['bb_lower'])
    else:
        features['bb_position'] = 0.5
    
    # Realized variance
    returns = buffers.get('returns')
    features['realized_var'] = compute_realized_variance(returns, period=20)
    
    # Pressure from volume
    buy_volumes = buffers.get('buy_volume')
    sell_volumes = buffers.get('sell_volume')
    features['pressure'] = compute_pressure(buy_volumes, sell_volumes, period=10)
    
    # Order book imbalance
    bid_volumes = buffers.get('bid_volume')
    ask_volumes = buffers.get('ask_volume')
    if bid_volumes and ask_volumes and len(bid_volumes) > 0 and len(ask_volumes) > 0:
        total_bid = bid_volumes[-1]
        total_ask = ask_volumes[-1]
        total = total_bid + total_ask
        features['book_imb'] = (total_bid - total_ask) / total if total > 0 else 0.0
    else:
        features['book_imb'] = 0.0
    
    # Time of day normalized [0, 1]
    features['tod'] = 0.0  # Placeholder - computed from timestamp in snapshot
    
    return features


def regime_scores(features: Dict[str, float]) -> Dict[RegimeID, float]:
    """Compute soft regime scores from features using heuristics."""
    # Initialize scores
    scores = {
        RegimeID.TREND: 0.0,
        RegimeID.RANGE: 0.0,
        RegimeID.VOLX: 0.0,
        RegimeID.QUIET: 0.0,
        RegimeID.UNKNOWN: 0.0
    }
    
    # Extract features
    rsi = features.get('rsi', 50.0)
    ema_slope = features.get('ema_slope', 0.0)
    bb_width = features.get('bb_width', 0.0)
    realized_var = features.get('realized_var', 0.0)
    atr = features.get('atr', 0.0)
    
    # TREND: Strong directional movement with RSI extremes and EMA slope
    trend_score = 0.0
    if abs(ema_slope) > 0.01:  # Strong EMA divergence
        trend_score += 0.4
    if rsi > 60 or rsi < 40:  # RSI trending
        trend_score += 0.3
    if bb_width > 0.03:  # Wide bands indicate trending
        trend_score += 0.3
    scores[RegimeID.TREND] = min(trend_score, 1.0)
    
    # RANGE: Bounded movement with mean reversion characteristics
    range_score = 0.0
    if 45 <= rsi <= 55:  # Neutral RSI
        range_score += 0.4
    if abs(ema_slope) < 0.005:  # Flat EMAs
        range_score += 0.4
    if bb_width < 0.02:  # Narrow bands
        range_score += 0.2
    scores[RegimeID.RANGE] = min(range_score, 1.0)
    
    # VOLX: High volatility expansion
    volx_score = 0.0
    if bb_width > 0.04:  # Very wide bands
        volx_score += 0.4
    if realized_var > 0.001:  # High realized variance
        volx_score += 0.4
    if atr > 0:  # Consider ATR relative measure
        volx_score += 0.2
    scores[RegimeID.VOLX] = min(volx_score, 1.0)
    
    # QUIET: Low volatility
    quiet_score = 0.0
    if bb_width < 0.015:  # Very narrow bands
        quiet_score += 0.4
    if realized_var < 0.0001:  # Low variance
        quiet_score += 0.4
    if abs(ema_slope) < 0.003:  # Very flat
        quiet_score += 0.2
    scores[RegimeID.QUIET] = min(quiet_score, 1.0)
    
    # Normalize scores to sum to 1
    total = sum(scores.values())
    if total > 0:
        for regime in scores:
            scores[regime] = scores[regime] / total
    else:
        # Default to unknown if no clear regime
        scores[RegimeID.UNKNOWN] = 1.0
    
    return scores


class FeatureEngine:
    """
    Feature Engine for computing technical and microstructure features.
    
    Maintains rolling windows for microstructure and technicals, producing both
    continuous features (features_vec) and discretized integers (features_disc)
    for prime embedding.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        self.buffers = RollingBuffers(cfg.get('windows', {}))
        self.discretizers = make_discretizers(cfg.get('binning_specs', {}))
    
    def update(self, tick: Dict[str, Any]) -> None:
        """Update internal buffers with new tick data."""
        self.buffers.update(tick)
    
    def snapshot(self, ts: int, symbol: str) -> Observation:
        """
        Create an observation snapshot from current buffer state.
        
        Args:
            ts: Timestamp in milliseconds
            symbol: Trading symbol
            
        Returns:
            Observation object with all features
        """
        # Compute raw features
        feats = compute_features(self.buffers)
        regime_soft = regime_scores(feats)
        
        # Extract current market data from buffers
        prices = self.buffers.get('price')
        bids = self.buffers.get('bid')
        asks = self.buffers.get('ask')
        volumes = self.buffers.get('volume')
        
        current_price = prices[-1] if prices else 0.0
        current_bid = bids[-1] if bids else current_price * 0.9995
        current_ask = asks[-1] if asks else current_price * 1.0005
        current_volume = volumes[-1] if volumes else 0.0
        
        # Time of day bucket (0-23)
        dt = datetime.fromtimestamp(ts / 1000.0)
        tod_bucket = dt.hour
        feats['tod'] = tod_bucket / 24.0  # Normalize to [0, 1]
        
        # Discretize features
        features_disc = {}
        if 'ret' in self.discretizers:
            features_disc['ret_bucket'] = self.discretizers['ret'].bin(feats.get('ret_1s', 0.0))
        if 'imb' in self.discretizers:
            features_disc['imb_bucket'] = self.discretizers['imb'].bin(feats.get('book_imb', 0.0))
        if 'regime' in self.discretizers:
            top_regime = max(regime_soft, key=regime_soft.get)
            features_disc['regime_id'] = self.discretizers['regime'].bin(list(RegimeID).index(top_regime))
        if 'tod' in self.discretizers:
            features_disc['tod_bucket'] = self.discretizers['tod'].bin(feats.get('tod', 0.0))
        
        # Pack features into vector
        features_vec = np.array([
            feats.get('ret_1s', 0.0),
            feats.get('book_imb', 0.0),
            feats.get('tod', 0.0),
            feats.get('rsi', 50.0),
            feats.get('ema_slope', 0.0),
            feats.get('atr', 0.0),
            feats.get('bb_width', 0.0),
            feats.get('pressure', 0.0),
            feats.get('realized_var', 0.0),
        ])
        
        # Create observation with computed features
        return Observation(
            ts=ts,
            symbol=symbol,
            mid=current_price,
            bid=current_bid,
            ask=current_ask,
            spread=current_ask - current_bid,
            last_px=current_price,
            last_qty=current_volume,
            vol_1s=current_volume,
            vol_1m=sum(volumes[-60:]) if len(volumes) >= 60 else sum(volumes),
            book_imbalance=feats.get('book_imb', 0.0),
            pressure=feats.get('pressure', 0.0),
            realized_var=feats.get('realized_var', 0.0),
            atr=feats.get('atr', 0.0),
            rsi_short=feats.get('rsi', 50.0),
            ema_slope=feats.get('ema_slope', 0.0),
            inventory=0.0,  # Filled by caller
            equity=0.0,  # Filled by caller
            unrealized_pnl=0.0,  # Filled by caller
            funding_rate=0.0,  # Filled by caller or from buffer
            time_of_day_bucket=tod_bucket,
            regime_soft=regime_soft,
            features_vec=features_vec,
            features_disc=features_disc
        )
