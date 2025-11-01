"""
Entropy thermostat (τ controller) using PI control.
"""

import numpy as np


class TauController:
    """
    Entropy thermostat using PI (Proportional-Integral) control.
    
    Maintains H(Ψ) ≈ H* by adjusting τ.
    """
    
    def __init__(
        self, 
        H_star: float, 
        kP: float, 
        kI: float, 
        bounds: tuple[float, float],
        initial_tau: float = None
    ):
        """
        Initialize τ controller.
        
        Args:
            H_star: Target entropy setpoint
            kP: Proportional gain
            kI: Integral gain
            bounds: (min_tau, max_tau) bounds
            initial_tau: Initial τ value (default: mid-point of bounds)
        """
        self.H_star = H_star
        self.kP = kP
        self.kI = kI
        self.bounds = bounds
        
        if initial_tau is None:
            self.tau = (bounds[0] + bounds[1]) / 2.0
        else:
            self.tau = np.clip(initial_tau, bounds[0], bounds[1])
        
        self.e_int = 0.0  # Integral error
    
    def step(self, H_now: float) -> float:
        """
        Update τ based on current entropy.
        
        Args:
            H_now: Current entropy H(Ψ)
            
        Returns:
            Updated τ value
        """
        # Compute error
        e = H_now - self.H_star
        
        # Update integral
        self.e_int += e
        
        # PI control law
        tau_new = self.tau - self.kP * e - self.kI * self.e_int
        
        # Apply bounds
        self.tau = np.clip(tau_new, self.bounds[0], self.bounds[1])
        
        return self.tau
    
    def reset(self) -> None:
        """Reset controller state."""
        self.tau = (self.bounds[0] + self.bounds[1]) / 2.0
        self.e_int = 0.0
    
    def get_error(self, H_now: float) -> float:
        """Get current error."""
        return H_now - self.H_star
