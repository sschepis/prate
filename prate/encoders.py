"""
Encoding functions for holographic memory keys and values.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from .types import Observation, Basis


def _seeded_phase(seed_bytes: bytes, p: int, two_pi: float = 2 * np.pi) -> float:
    """Generate stable seeded phase for a prime."""
    h = np.frombuffer(
        np.uint64(np.abs(hash(seed_bytes)) & ((1 << 63) - 1)).tobytes(),
        dtype=np.uint64
    )[0]
    # Use modulo to prevent overflow
    mix_val = int(p) * 1469598103934665603
    h_mixed = int(h) ^ (mix_val & 0xFFFFFFFFFFFFFFFF)
    return two_pi * ((h_mixed % (10 ** 6)) / 1_000_000.0)


def encode_key(
    obs: Observation,
    cfg: Dict,
    P: List[int],
    guild_id: Optional[str] = None
) -> np.ndarray:
    """
    Encode observation context as complex key vector.
    
    Args:
        obs: Observation with discretized features
        cfg: Configuration with weights and mask
        P: List of primes
        guild_id: Optional guild identifier
        
    Returns:
        K: Complex unit vector of length M
    """
    M = len(P)
    K = np.zeros(M, dtype=np.complex128)
    
    codes = obs.features_disc.copy()
    
    # Add regime if not present
    if 'regime_id' not in codes and hasattr(obs, 'regime_soft'):
        codes['regime_id'] = int(max(obs.regime_soft, key=obs.regime_soft.get))
    
    # Seeding for stable phase offsets
    symbol_bytes = obs.symbol.encode('utf-8')
    guild_bytes = (str(guild_id) if guild_id else 'NA').encode('utf-8')
    
    weights = cfg.get('weights', {})
    mask = set(cfg.get('mask', [])) if cfg.get('mask') else None
    
    for j, p in enumerate(P):
        alpha_p = _seeded_phase(symbol_bytes + b'|' + guild_bytes, p)
        acc = 0 + 0j
        
        for name, x in codes.items():
            w = float(weights.get(name, 1.0))
            r = x % p
            theta = (2 * np.pi * r / p) + alpha_p
            acc += w * np.exp(1j * theta)
        
        # Bias tag
        acc += 0.25 * np.exp(1j * (alpha_p + 0.37))
        
        # Prime masking
        if mask and p not in mask:
            acc *= 0.5
        
        # Unit normalize
        if acc == 0:
            acc = 1 + 0j
        K[j] = acc / np.abs(acc)
    
    return K


def encode_value(
    B: Basis,
    dphi: np.ndarray,
    dtau: float,
    hints: Dict[str, int],
    cfg: Dict,
    P: List[int]
) -> np.ndarray:
    """
    Encode strategy deltas as complex value vector.
    
    Args:
        B: Active basis
        dphi: Phase nudges (length M)
        dtau: Tau shift
        hints: Strategy hints
        cfg: Configuration with beta weights
        P: List of primes
        
    Returns:
        V: Complex unit vector of length M
    """
    M = len(P)
    V = np.zeros(M, dtype=np.complex128)
    
    beta_basis = cfg.get('beta_basis', 0.8)
    beta_phi = cfg.get('beta_phi', 0.6)
    beta_tau = cfg.get('beta_tau', 0.4)
    beta_hint = cfg.get('beta_hint', 0.3)
    k_phi = cfg.get('k_phi', 1.0)
    k_tau = cfg.get('k_tau', 1.0)
    
    # Handle dphi as dict or array
    if isinstance(dphi, dict):
        dphi_vec = np.zeros(M)
        for idx, d in dphi.items():
            dphi_vec[idx] = d
    else:
        dphi_vec = np.asarray(dphi)
    
    basis_set = set(B.primes)
    
    for j, p in enumerate(P):
        alpha_p = _seeded_phase(f"basis|{p}".encode(), p)
        v = 0 + 0j
        
        # Basis tag
        if p in basis_set:
            v += beta_basis * np.exp(1j * alpha_p)
        
        # Phase delta
        v += beta_phi * np.exp(1j * (k_phi * dphi_vec[j]))
        
        # Tau bias
        v += beta_tau * np.exp(1j * (k_tau * dtau))
        
        # Hints
        if hints:
            for name, hv in hints.items():
                r = int(hv) % p
                theta = (2 * np.pi * r / p) + 0.11
                v += beta_hint * np.exp(1j * theta)
        
        # Unit normalize
        if v == 0:
            v = 1 + 0j
        V[j] = v / np.abs(v)
    
    return V


def decode_value(
    Vhat: np.ndarray,
    P: List[int],
    basis_catalog: Dict[str, List[int]],
    cfg_value: Dict,
    known_hint_names: Optional[List[str]] = None,
    delta_offset_hint: float = 0.11,
    shrink_phi: float = 0.30,
    dtau_clip: float = 2.0,
    return_confidence: bool = True
) -> Tuple[Dict[str, float], np.ndarray, float, Dict[str, int], Optional[Dict]]:
    """
    Decode retrieved value vector into priors and deltas.
    
    Args:
        Vhat: Retrieved complex vector
        P: List of primes
        basis_catalog: Dictionary of basis_id -> primes
        cfg_value: Value encoding configuration
        known_hint_names: Names of hints to decode
        delta_offset_hint: Phase offset for hints
        shrink_phi: Shrinkage factor for phi
        dtau_clip: Clipping bound for dtau
        return_confidence: Whether to return confidence scores
        
    Returns:
        (B_prior, dphi, dtau, hints, conf)
    """
    M = len(P)
    angles = np.angle(Vhat)
    mags = np.abs(Vhat)
    
    k_phi = cfg_value.get('k_phi', 1.0)
    k_tau = cfg_value.get('k_tau', 1.0)
    
    # Basis prior
    alpha_basis = np.array([_seeded_phase(f"basis|{p}".encode(), p) for p in P])
    B_scores = {}
    
    for bid, B_primes in basis_catalog.items():
        mask = np.array([1 if p in set(B_primes) else 0 for p in P], dtype=np.float64)
        proj = np.real(Vhat * np.exp(-1j * alpha_basis))
        denom = mask.sum() if mask.sum() > 0 else 1.0
        score = (proj * mask).sum() / denom
        B_scores[bid] = float(score)
    
    # Normalize to probabilities
    if B_scores:
        scores_arr = np.array(list(B_scores.values()))
        temp = 1.0
        exps = np.exp((scores_arr - scores_arr.max()) / max(1e-6, temp))
        probs = exps / (exps.sum() + 1e-12)
        B_prior = {bid: float(p) for bid, p in zip(B_scores.keys(), probs)}
        best_B = max(B_prior.items(), key=lambda kv: kv[1])[0]
        basis_conf = float(B_prior[best_B])
    else:
        B_prior = {}
        best_B = None
        basis_conf = 0.0
    
    # dphi
    dphi = ((angles + np.pi) % (2 * np.pi) - np.pi) / max(k_phi, 1e-6)
    dphi *= (1.0 - shrink_phi)
    
    # dtau
    if best_B is not None:
        mask_best = np.array([1 if p in set(basis_catalog[best_B]) else 0 for p in P])
    else:
        mask_best = np.zeros(M)
    
    V_clean = Vhat.copy()
    idxs = np.where(mask_best > 0)[0]
    if idxs.size > 0:
        V_clean[idxs] = V_clean[idxs] * np.exp(-1j * alpha_basis[idxs])
    
    mean_vec = V_clean.mean()
    dtau = ((np.angle(mean_vec) + np.pi) % (2 * np.pi) - np.pi) / max(k_tau, 1e-6)
    dtau = float(np.clip(dtau, -dtau_clip, dtau_clip))
    tau_conf = float(np.abs(mean_vec))
    
    # Hints (simplified)
    hints = {}
    if known_hint_names:
        for hname in known_hint_names:
            # Take first prime's residue as hint
            if len(P) > 0:
                p = P[0]
                r_est = int(np.round((p / (2 * np.pi)) * ((angles[0] - delta_offset_hint + np.pi) % (2 * np.pi) - np.pi))) % p
                hints[hname] = r_est
    
    # Confidence
    conf = None
    if return_confidence:
        phi_snr = float(1.0 / (np.std((angles + np.pi) % (2 * np.pi) - np.pi) + 1e-6))
        conf = {"basis_conf": basis_conf, "phi_snr": phi_snr, "tau_conf": tau_conf}
    
    return B_prior, dphi, dtau, hints, conf
