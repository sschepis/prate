"""
Residue feature computation for phase learning.
"""

import numpy as np
from typing import Dict, List
from .types import Observation


def _mix_ints(values: List[int], seed: int = 1469598103) -> int:
    """Mix integers into a single hash code."""
    h = seed
    for v in values:
        h = (h ^ (v + 0x9e3779b97f4a7c15)) & 0xffffffffffffffff
        h = (h * 1099511628211) & 0xffffffffffffffff
    return int(h & 0xffffffff)


def residue_features(
    obs: Observation,
    params: Dict[str, any],
    P: List[int],
    topk: int = 12,
    mix: str = 'state+action',
    lambdas: tuple = (1.0, 1.0)
) -> Dict[int, float]:
    """
    Compute sparse residue features for phase learning.
    
    Args:
        obs: Observation with discretized features
        params: Action parameters
        P: List of primes
        topk: Number of top features to keep
        mix: Feature mixing mode
        lambdas: (state_weight, action_weight)
        
    Returns:
        Dictionary mapping prime index to feature value
    """
    # Build state code
    state_ints = []
    for k, v in sorted(obs.features_disc.items()):
        state_ints.append(int(v))
    x_s = _mix_ints(state_ints)
    
    # Build action code
    style_id = int(params.get('style_id', 0))
    dq_bucket = int(np.clip(abs(params.get('delta_q', 0.0)) * 1000, 0, 999))
    knobs = []
    
    for name in sorted(params.keys()):
        if name in ('style_id', 'delta_q'):
            continue
        val = params[name]
        if isinstance(val, (int, np.integer)):
            knobs.append(int(val))
        elif isinstance(val, float):
            knobs.append(int(np.clip(round(val * 1000), -999, 999)) & 0xffff)
    
    x_a = _mix_ints([style_id, dq_bucket] + knobs, seed=2166136261)
    
    lam_s, lam_a = lambdas
    feats = []
    
    for j, p in enumerate(P):
        rs = x_s % p
        ra = x_a % p
        val = lam_s * np.cos(2 * np.pi * rs / p) + lam_a * np.sin(2 * np.pi * ra / p)
        feats.append((j, float(val)))
    
    # Keep top-k by magnitude
    feats.sort(key=lambda t: abs(t[1]), reverse=True)
    sparse = {j: v for (j, v) in feats[:topk] if abs(v) > 1e-6}
    
    return sparse
