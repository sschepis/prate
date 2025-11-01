"""
RL Integration Module - Reinforcement Learning components for PRATE.

Implements:
- Actor-critic policy (PPO/SAC compatible)
- State packing utilities
- Experience replay buffer
- Continuous parameter optimization
- Gradient computation
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Experience:
    """Single experience tuple for replay."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class ReplayBuffer:
    """
    Experience replay buffer for RL training.
    
    Supports both on-policy (PPO) and off-policy (SAC) algorithms.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
    
    def push(self, experience: Experience) -> None:
        """
        Add experience to buffer.
        
        Args:
            experience: Experience tuple
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experiences
        """
        if len(self.buffer) < batch_size:
            return self.buffer.copy()
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_trajectory(self, trajectory_length: Optional[int] = None) -> List[Experience]:
        """
        Sample a complete trajectory (for on-policy algorithms).
        
        Args:
            trajectory_length: Length of trajectory (uses all if None)
            
        Returns:
            List of experiences in temporal order
        """
        if trajectory_length is None:
            return self.buffer.copy()
        
        start_idx = max(0, len(self.buffer) - trajectory_length)
        return self.buffer[start_idx:]
    
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()
        self.position = 0
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class StatePacker:
    """
    Packs observations and context into RL state vectors.
    
    Handles normalization and feature engineering for RL input.
    """
    
    def __init__(
        self,
        state_dim: int,
        normalize: bool = True,
        history_length: int = 10
    ):
        """
        Initialize state packer.
        
        Args:
            state_dim: Dimension of packed state vector
            normalize: Whether to normalize features
            history_length: Number of historical observations to include
        """
        self.state_dim = state_dim
        self.normalize = normalize
        self.history_length = history_length
        
        # Running statistics for normalization
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.count = 0
        
        # History buffer
        self.history: List[np.ndarray] = []
    
    def pack(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Pack observation and context into state vector.
        
        Args:
            observation: Market observation dictionary
            context: Additional context (positions, PnL, etc.)
            
        Returns:
            Packed state vector
        """
        features = []
        
        # Extract price features
        if 'mid' in observation:
            features.append(observation['mid'])
        if 'spread' in observation:
            features.append(observation['spread'])
        
        # Extract technical indicators
        for key in ['rsi_short', 'ema_slope', 'atr', 'realized_var', 'pressure', 'book_imbalance']:
            if key in observation:
                features.append(observation[key])
        
        # Extract regime probabilities
        if 'regime_soft' in observation:
            for regime, prob in observation['regime_soft'].items():
                features.append(prob)
        
        # Extract position/account info
        if context:
            for key in ['inventory', 'equity', 'unrealized_pnl']:
                if key in context:
                    features.append(context[key])
        
        # Pad or truncate to state_dim
        state = np.array(features, dtype=np.float32)
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]
        
        # Add to history
        self.history.append(state.copy())
        if len(self.history) > self.history_length:
            self.history.pop(0)
        
        # Update running statistics
        if self.normalize:
            self._update_stats(state)
            state = self._normalize(state)
        
        return state
    
    def pack_with_history(
        self,
        observation: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Pack observation with historical context.
        
        Args:
            observation: Market observation dictionary
            context: Additional context
            
        Returns:
            Packed state vector with history (flattened)
        """
        current_state = self.pack(observation, context)
        
        # Concatenate history
        if len(self.history) < self.history_length:
            # Pad with zeros if not enough history
            padding = [np.zeros(self.state_dim) for _ in range(self.history_length - len(self.history))]
            full_history = padding + self.history
        else:
            full_history = self.history[-self.history_length:]
        
        # Flatten history
        return np.concatenate(full_history)
    
    def _update_stats(self, state: np.ndarray) -> None:
        """Update running mean and std."""
        self.count += 1
        delta = state - self.mean
        self.mean += delta / self.count
        delta2 = state - self.mean
        self.std = np.sqrt((self.std ** 2 * (self.count - 1) + delta * delta2) / self.count)
    
    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using running statistics."""
        # Avoid division by zero
        std_safe = np.where(self.std > 1e-8, self.std, 1.0)
        return (state - self.mean) / std_safe


class ActorCritic:
    """
    Actor-Critic policy network (simplified implementation).
    
    For production, this should be replaced with PyTorch/JAX implementation.
    This version uses simple linear approximation for demonstration.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 3e-4
    ):
        """
        Initialize actor-critic network.
        
        Args:
            state_dim: Dimension of state vector
            action_dim: Dimension of action vector
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # Initialize weights (simple linear approximation)
        # For production, replace with proper neural network
        self.actor_weights = np.random.randn(state_dim, action_dim) * 0.01
        self.actor_bias = np.zeros(action_dim)
        
        self.critic_weights = np.random.randn(state_dim, 1) * 0.01
        self.critic_bias = np.zeros(1)
        
        # Action std (learnable for continuous actions)
        self.log_std = np.zeros(action_dim)
    
    def forward_actor(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through actor network.
        
        Args:
            state: State vector
            
        Returns:
            (mean, std) of action distribution
        """
        mean = np.dot(state, self.actor_weights) + self.actor_bias
        std = np.exp(self.log_std)
        return mean, std
    
    def forward_critic(self, state: np.ndarray) -> float:
        """
        Forward pass through critic network.
        
        Args:
            state: State vector
            
        Returns:
            State value estimate
        """
        value = np.dot(state, self.critic_weights) + self.critic_bias
        return float(value[0])
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Select action from policy.
        
        Args:
            state: State vector
            deterministic: If True, return mean action (no sampling)
            
        Returns:
            (action, log_prob) tuple
        """
        mean, std = self.forward_actor(state)
        
        if deterministic:
            action = mean
            log_prob = 0.0
        else:
            # Sample from Gaussian
            action = mean + std * np.random.randn(self.action_dim)
            
            # Compute log probability
            log_prob = -0.5 * np.sum(
                ((action - mean) / (std + 1e-8)) ** 2 + 
                2 * self.log_std + 
                np.log(2 * np.pi)
            )
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate actions for given states.
        
        Args:
            states: Batch of states (N x state_dim)
            actions: Batch of actions (N x action_dim)
            
        Returns:
            (values, log_probs, entropy) tuple
        """
        values = []
        log_probs = []
        entropies = []
        
        for state, action in zip(states, actions):
            # Value
            value = self.forward_critic(state)
            values.append(value)
            
            # Log prob
            mean, std = self.forward_actor(state)
            log_prob = -0.5 * np.sum(
                ((action - mean) / (std + 1e-8)) ** 2 + 
                2 * self.log_std + 
                np.log(2 * np.pi)
            )
            log_probs.append(log_prob)
            
            # Entropy
            entropy = 0.5 * np.sum(self.log_std + np.log(2 * np.pi * np.e))
            entropies.append(entropy)
        
        return (
            np.array(values),
            np.array(log_probs),
            np.array(entropies)
        )
    
    def compute_gradients(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_probs: np.ndarray,
        clip_epsilon: float = 0.2
    ) -> Dict[str, np.ndarray]:
        """
        Compute PPO gradients.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            advantages: Advantage estimates
            returns: Discounted returns
            old_log_probs: Log probs from old policy
            clip_epsilon: PPO clipping parameter
            
        Returns:
            Dictionary of gradients
        """
        # Evaluate current policy
        values, log_probs, entropies = self.evaluate_actions(states, actions)
        
        # Compute ratio for PPO
        ratio = np.exp(log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        
        # Policy loss (negative for gradient ascent)
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        
        # Value loss
        value_loss = np.mean((returns - values) ** 2)
        
        # Entropy bonus (for exploration)
        entropy_loss = -np.mean(entropies)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        # Compute gradients (simplified - in practice use autograd)
        # This is a placeholder - real implementation would use PyTorch/JAX
        gradients = {
            'actor_weights': np.zeros_like(self.actor_weights),
            'actor_bias': np.zeros_like(self.actor_bias),
            'critic_weights': np.zeros_like(self.critic_weights),
            'critic_bias': np.zeros_like(self.critic_bias),
            'log_std': np.zeros_like(self.log_std),
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': np.mean(entropies)
        }
        
        return gradients
    
    def update(self, gradients: Dict[str, np.ndarray]) -> None:
        """
        Update policy parameters.
        
        Args:
            gradients: Dictionary of gradients
        """
        # Update actor
        if 'actor_weights' in gradients:
            self.actor_weights -= self.lr * gradients['actor_weights']
        if 'actor_bias' in gradients:
            self.actor_bias -= self.lr * gradients['actor_bias']
        
        # Update critic
        if 'critic_weights' in gradients:
            self.critic_weights -= self.lr * gradients['critic_weights']
        if 'critic_bias' in gradients:
            self.critic_bias -= self.lr * gradients['critic_bias']
        
        # Update log std
        if 'log_std' in gradients:
            self.log_std -= self.lr * gradients['log_std']


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        dones: Array of done flags
        gamma: Discount factor
        lambda_: GAE lambda parameter
        
    Returns:
        (advantages, returns) tuple
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lambda_ * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    
    return advantages, returns


def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """
    Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: Advantage estimates
        
    Returns:
        Normalized advantages
    """
    return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
