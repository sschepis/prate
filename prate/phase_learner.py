"""
Phase learner for online φ updates based on reward feedback.
"""

import numpy as np
from typing import Dict, Set
from .embedding import wrap_angle_arr


class PhaseLearner:
    """
    Online phase learning with reward-based updates.
    
    Updates phase vector φ based on reward advantages and residue features.
    """
    
    def __init__(
        self, 
        P: list[int], 
        eta0: float, 
        protected: Set[int] = None
    ):
        """
        Initialize phase learner.
        
        Args:
            P: List of primes
            eta0: Initial learning rate
            protected: Set of prime indices that should not be updated
        """
        self.P = P
        self.M = len(P)
        self.eta0 = eta0
        self.protected = protected if protected is not None else set()
        
        self.phi = np.zeros(self.M)
        self.t = 0
    
    def step(
        self, 
        reward: float, 
        baseline: float, 
        residue_feats: Dict[int, float]
    ) -> np.ndarray:
        """
        Perform one phase learning update.
        
        Args:
            reward: Observed reward
            baseline: Baseline (running average) reward
            residue_feats: Sparse residue features {prime_index: value}
            
        Returns:
            Updated phase vector φ
        """
        self.t += 1
        
        # Decaying learning rate
        eta_t = self.eta0 / np.sqrt(self.t)
        
        # Advantage
        adv = reward - baseline
        
        # Update phases
        dphi = np.zeros(self.M)
        for p_idx, v in residue_feats.items():
            if p_idx in self.protected:
                continue
            if 0 <= p_idx < self.M:
                dphi[p_idx] += eta_t * adv * v
        
        # Apply update and wrap angles
        self.phi = wrap_angle_arr(self.phi + dphi)
        
        return self.phi
    
    def get_phi(self) -> np.ndarray:
        """Get current phase vector."""
        return self.phi.copy()
    
    def reset(self) -> None:
        """Reset phase vector to zeros."""
        self.phi = np.zeros(self.M)
        self.t = 0


class Baseline:
    """Exponential moving average baseline for reward."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize baseline.
        
        Args:
            alpha: EMA smoothing factor
        """
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False
    
    def update(self, reward: float) -> float:
        """
        Update baseline with new reward.
        
        Args:
            reward: New reward observation
            
        Returns:
            Current baseline value (before update)
        """
        old_value = self.value
        
        if not self.initialized:
            self.value = reward
            self.initialized = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * reward
        
        return old_value
    
    def get(self) -> float:
        """Get current baseline value."""
        return self.value
