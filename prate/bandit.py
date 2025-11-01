"""
Bandit for basis/style selection using Thompson sampling or UCB.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from .types import Basis


class BasisBandit:
    """
    Multi-armed bandit for basis/style selection.
    
    Supports Thompson sampling and UCB algorithms.
    """
    
    def __init__(
        self, 
        candidates: List[Basis], 
        algo: str = "thompson"
    ):
        """
        Initialize bandit.
        
        Args:
            candidates: List of candidate bases
            algo: Algorithm ("thompson" or "ucb")
        """
        self.candidates = candidates
        self.algo = algo
        
        # Statistics: (mean reward, count)
        self.stats = {B.id: (0.0, 0) for B in candidates}
        
        # Posteriors: (mean, std)
        self.posteriors = {B.id: (0.0, 1.0) for B in candidates}
        
        self.t = 0  # Total trials
    
    def sample_with_prior(
        self, 
        prior: Optional[Dict[str, float]] = None
    ) -> Basis:
        """
        Sample a basis using the bandit algorithm, optionally with priors.
        
        Args:
            prior: Optional dictionary of basis_id -> prior score
            
        Returns:
            Selected basis
        """
        self.t += 1
        
        scores = []
        for B in self.candidates:
            μ, σ = self.posteriors[B.id]
            
            if self.algo == "thompson":
                # Thompson sampling: draw from posterior
                draw = np.random.normal(μ, max(σ, 1e-6))
            else:  # UCB
                _, n = self.stats[B.id]
                if n == 0:
                    draw = float('inf')  # Explore unvisited arms
                else:
                    # UCB formula: μ + β√(2ln(t)/n)
                    beta = 1.0
                    draw = μ + beta * np.sqrt(2 * np.log(self.t) / n)
            
            # Add prior if provided
            if prior and B.id in prior:
                draw += prior[B.id]
            
            scores.append((draw, B))
        
        # Select basis with highest score
        return max(scores, key=lambda x: x[0])[1]
    
    def update(self, basis_id: str, reward: float) -> None:
        """
        Update bandit statistics with observed reward.
        
        Args:
            basis_id: ID of selected basis
            reward: Observed reward
        """
        μ, n = self.stats[basis_id]
        
        # Update statistics
        n_new = n + 1
        μ_new = μ + (reward - μ) / n_new
        
        self.stats[basis_id] = (μ_new, n_new)
        
        # Update posterior (simple Bayesian update)
        # Assume Gaussian posterior with decreasing variance
        σ_new = 1.0 / np.sqrt(n_new + 1)
        self.posteriors[basis_id] = (μ_new, σ_new)
    
    def get_stats(self) -> Dict[str, Tuple[float, int]]:
        """Get current statistics."""
        return self.stats.copy()
    
    def get_best_basis(self) -> str:
        """Get basis with highest mean reward."""
        return max(self.stats.items(), key=lambda kv: kv[1][0])[0]
