"""
Prime-Hilbert embedding for market observations.
"""

import numpy as np
from typing import Dict, Tuple, List
from .types import Observation


def p_index(p: int, P: List[int]) -> int:
    """Get index of prime p in prime list P."""
    return P.index(p)


def wrap_angle(x: float) -> float:
    """Wrap angle to [-π, π)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def wrap_angle_arr(x: np.ndarray) -> np.ndarray:
    """Wrap array of angles to [-π, π)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


class PrimeEmbedder:
    """
    Prime-Hilbert Embedding.
    
    Maps discretized integer features to prime-indexed amplitudes and phases.
    """
    
    def __init__(self, primes: List[int], M: int):
        """
        Initialize embedder with first M primes.
        
        Args:
            primes: List of prime numbers
            M: Number of primes to use
        """
        self.P = primes[:M]
        self.M = M
    
    def embed(
        self, 
        o: Observation, 
        phi: np.ndarray
    ) -> Dict[int, Tuple[float, float]]:
        """
        Embed observation into prime-indexed Hilbert space.
        
        Args:
            o: Observation with discretized features
            phi: Phase vector (length M)
            
        Returns:
            Ψ: Dictionary mapping prime p to (amplitude, phase) tuple
        """
        # Extract integer features from discretized observation
        ints = [o.features_disc[k] for k in sorted(o.features_disc.keys())]
        
        if not ints:
            # No discretized features, return zeros
            return {p: (0.0, 0.0) for p in self.P}
        
        Ψ = {}
        for j, p in enumerate(self.P):
            a_sum = 0.0
            φ_sum = 0.0
            
            for x in ints:
                r = x % p
                # Amplitude: a_p(x) = 1 - (x mod p)/p
                a_sum += 1.0 - r / p
                # Phase: φ_p(x) = 2π(x mod p + φ_p)/p
                φ_sum += 2.0 * np.pi * (r + phi[j]) / p
            
            # Average over features
            a_avg = a_sum / len(ints)
            φ_avg = wrap_angle(φ_sum / len(ints))
            
            Ψ[p] = (a_avg, φ_avg)
        
        return Ψ


def hilbert_entropy(Ψ: Dict[int, Tuple[float, float]]) -> float:
    """
    Compute entropy of Hilbert embedding.
    
    Args:
        Ψ: Prime-indexed amplitudes and phases
        
    Returns:
        Entropy H(Ψ)
    """
    if not Ψ:
        return 0.0
    
    # Extract amplitudes
    amplitudes = np.array([a for (a, _) in Ψ.values()])
    
    # Normalize to probability distribution
    total = amplitudes.sum()
    if total == 0:
        return 0.0
    
    probs = amplitudes / total
    
    # Shannon entropy
    # Avoid log(0) by filtering out zero probabilities
    probs_nonzero = probs[probs > 1e-12]
    if len(probs_nonzero) == 0:
        return 0.0
    
    entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
    
    return float(entropy)
