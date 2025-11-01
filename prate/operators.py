"""
Hilbert operators: Projection (Π), Entropy Collapse (E_τ), Measurement (M), and Refinement.
"""

import numpy as np
from typing import Dict, Tuple, Any
from .types import Basis
from .embedding import hilbert_entropy


class Operators:
    """
    Hilbert space operators for PRATE.
    
    Implements:
    - Π: Projection onto basis subset
    - E_τ: Entropy collapse (soft top-k)
    - M: Measurement (deterministic hash)
    - R: Refinement (composite operation)
    """
    
    def project(
        self, 
        Ψ: Dict[int, Tuple[float, float]], 
        B: Basis
    ) -> Dict[int, Tuple[float, float]]:
        """
        Project Ψ onto basis B.
        
        Args:
            Ψ: Full prime embedding
            B: Basis subset
            
        Returns:
            Ψ_B: Projection onto basis primes
        """
        return {p: Ψ[p] for p in B.primes if p in Ψ}
    
    def collapse(
        self, 
        ΨB: Dict[int, Tuple[float, float]], 
        tau: float
    ) -> Dict[int, Tuple[float, float]]:
        """
        Entropy collapse: keep top-k primes until H(Ψ) ≤ τ.
        
        Args:
            ΨB: Projected embedding
            tau: Target entropy threshold
            
        Returns:
            Collapsed embedding
        """
        # Sort by amplitude (descending)
        ranked = sorted(ΨB.items(), key=lambda kv: kv[1][0], reverse=True)
        
        kept = {}
        for p, (a, φ) in ranked:
            kept[p] = (a, φ)
            H = hilbert_entropy(kept)
            if H <= tau:
                break
        
        return kept
    
    def measure(self, ΨB: Dict[int, Tuple[float, float]]) -> int:
        """
        Measurement operator: map Ψ_B to deterministic 32-bit integer.
        
        Uses FNV-1a hash variant mixing amplitudes and phases.
        
        Args:
            ΨB: Collapsed embedding
            
        Returns:
            32-bit measurement value
        """
        # FNV-1a hash
        h = 2166136261  # FNV offset basis
        
        for p, (a, φ) in sorted(ΨB.items()):
            # Mix amplitude and phase with prime
            val = int(a * 1e6) + int(φ * 1e6) + p
            h ^= val
            h *= 16777619  # FNV prime
            h &= 0xffffffff  # Keep 32 bits
        
        return h
    
    def refine(
        self, 
        n0: Dict[str, Any], 
        Ψ: Dict[int, Tuple[float, float]], 
        B: Basis, 
        tau: float
    ) -> Dict[str, Any]:
        """
        Refinement operator: composite Π ∘ E_τ ∘ M.
        
        Args:
            n0: Base proposal (initial parameters)
            Ψ: Full embedding
            B: Basis
            tau: Entropy threshold
            
        Returns:
            Refined parameters
        """
        # Apply operators in sequence
        ΨB = self.project(Ψ, B)
        ΨB_collapsed = self.collapse(ΨB, tau)
        m = self.measure(ΨB_collapsed)
        
        # Fuse measurement with base proposal
        return self._fuse(n0, m, ΨB_collapsed)
    
    def _fuse(
        self, 
        n0: Dict[str, Any], 
        m: int, 
        ΨB: Dict[int, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Fuse base proposal with measurement.
        
        Style-specific mapping of measurement to parameter adjustments.
        
        Args:
            n0: Base proposal
            m: Measurement value
            ΨB: Collapsed embedding
            
        Returns:
            Fused parameters
        """
        # Simple fusion: use measurement to perturb parameters
        params = n0.copy()
        
        # Extract some bits for different parameters
        seed = m
        rng = np.random.RandomState(seed)
        
        # Example: adjust position size based on measurement
        if 'delta_q' in params:
            # Add small perturbation
            perturbation = rng.uniform(-0.1, 0.1)
            params['delta_q'] = params['delta_q'] * (1.0 + perturbation)
        
        # Add measurement-derived parameters
        params['measurement'] = m
        params['entropy'] = hilbert_entropy(ΨB)
        
        return params
