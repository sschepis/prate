"""
Holographic memory using complex-valued Holographic Reduced Representations (HRR).
"""

import numpy as np
from typing import Tuple, List


def fft_circ_conv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT."""
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(y))


def fft_corr(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Correlation via FFT."""
    return np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(H))


class HoloMemory:
    """
    Holographic Memory using complex-valued HRR.
    
    Implements associative memory via circular convolution binding
    and correlation-based retrieval in the frequency domain.
    """
    
    def __init__(self, P: List[int], gamma: float, eta: float):
        """
        Initialize holographic memory.
        
        Args:
            P: List of primes (determines dimensionality)
            gamma: Decay factor for memory trace
            eta: Learning rate for new memories
        """
        self.P = P
        self.M = len(P)
        self.gamma = gamma
        self.eta = eta
        
        # Memory trace (complex-valued vector)
        self.H = np.zeros(self.M, dtype=np.complex128)
    
    def bind(self, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Bind key and value vectors.
        
        Args:
            K: Key vector (complex, length M)
            V: Value vector (complex, length M)
            
        Returns:
            Bound memory trace M = K ⊛ V
        """
        return fft_circ_conv(K, V)
    
    def correlate(self, Kq: np.ndarray) -> np.ndarray:
        """
        Correlate query key with memory.
        
        Args:
            Kq: Query key vector (complex, length M)
            
        Returns:
            Retrieved value vector
        """
        return fft_corr(Kq, self.H)
    
    def write(self, K: np.ndarray, V: np.ndarray, gain: float = 1.0) -> None:
        """
        Write key-value pair to memory.
        
        Args:
            K: Key vector (complex, length M)
            V: Value vector (complex, length M)
            gain: Scaling factor for this write
        """
        M_new = self.bind(K, V)
        self.H = self.gamma * self.H + self.eta * gain * M_new
    
    def read(self, Kq: np.ndarray) -> np.ndarray:
        """
        Read from memory using query key.
        
        Args:
            Kq: Query key vector (complex, length M)
            
        Returns:
            Retrieved value vector
        """
        return self.correlate(Kq)
    
    def reset(self) -> None:
        """Reset memory to zeros."""
        self.H = np.zeros(self.M, dtype=np.complex128)
    
    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        return {
            'magnitude': float(np.linalg.norm(self.H)),
            'phase_variance': float(np.var(np.angle(self.H))),
            'sparsity': float(np.sum(np.abs(self.H) < 1e-6) / self.M)
        }
