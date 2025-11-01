"""
Feature Engine for computing technical and microstructure features.
"""

import numpy as np
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


def compute_features(buffers: RollingBuffers) -> Dict[str, float]:
    """Compute features from rolling buffers."""
    features = {}
    
    # Example feature computations (simplified)
    prices = buffers.get('price')
    if len(prices) > 1:
        features['ret_1s'] = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0.0
    else:
        features['ret_1s'] = 0.0
    
    # Add more feature computations as needed
    features['book_imb'] = 0.0  # Placeholder
    features['tod'] = 0.0  # Time of day normalized
    
    return features


def regime_scores(features: Dict[str, float]) -> Dict[RegimeID, float]:
    """Compute soft regime scores from features."""
    # Simplified regime classification
    scores = {
        RegimeID.TREND: 0.25,
        RegimeID.RANGE: 0.25,
        RegimeID.VOLX: 0.25,
        RegimeID.QUIET: 0.25,
        RegimeID.UNKNOWN: 0.0
    }
    
    # In a real implementation, use a trained classifier
    # For now, return uniform distribution
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
        ])
        
        # Create observation (with placeholder values for now)
        return Observation(
            ts=ts,
            symbol=symbol,
            mid=0.0,
            bid=0.0,
            ask=0.0,
            spread=0.0,
            last_px=0.0,
            last_qty=0.0,
            vol_1s=0.0,
            vol_1m=0.0,
            book_imbalance=feats.get('book_imb', 0.0),
            pressure=0.0,
            realized_var=0.0,
            atr=0.0,
            rsi_short=0.0,
            ema_slope=0.0,
            inventory=0.0,
            equity=0.0,
            unrealized_pnl=0.0,
            funding_rate=0.0,
            time_of_day_bucket=features_disc.get('tod_bucket', 0),
            regime_soft=regime_soft,
            features_vec=features_vec,
            features_disc=features_disc
        )
