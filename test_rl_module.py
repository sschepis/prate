#!/usr/bin/env python3
"""
Tests for the RL integration module.
"""

import numpy as np
from prate.rl_module import (
    ReplayBuffer, Experience, StatePacker, ActorCritic,
    compute_gae, normalize_advantages
)


def test_replay_buffer():
    """Test replay buffer functionality."""
    print("Testing Replay Buffer...")
    
    buffer = ReplayBuffer(capacity=5)
    
    # Add experiences
    for i in range(3):
        exp = Experience(
            state=np.array([float(i)]),
            action=np.array([float(i * 2)]),
            reward=float(i),
            next_state=np.array([float(i + 1)]),
            done=False,
            log_prob=0.0,
            value=0.0
        )
        buffer.push(exp)
    
    assert len(buffer) == 3, f"Buffer should have 3 experiences, got {len(buffer)}"
    
    # Sample
    batch = buffer.sample(2)
    assert len(batch) == 2, f"Should sample 2 experiences, got {len(batch)}"
    
    # Test capacity limit
    for i in range(10):
        exp = Experience(
            state=np.array([float(i)]),
            action=np.array([float(i)]),
            reward=float(i),
            next_state=np.array([float(i)]),
            done=False,
            log_prob=0.0,
            value=0.0
        )
        buffer.push(exp)
    
    assert len(buffer) == 5, f"Buffer should cap at capacity 5, got {len(buffer)}"
    
    # Test clear
    buffer.clear()
    assert len(buffer) == 0, "Buffer should be empty after clear"
    
    print(f"  ✓ Replay buffer working correctly")
    print(f"    - Capacity management: OK")
    print(f"    - Sampling: OK")
    print(f"    - Clear: OK")


def test_state_packer():
    """Test state packing functionality."""
    print("\nTesting State Packer...")
    
    packer = StatePacker(state_dim=10, normalize=False, history_length=3)
    
    # Create observation
    observation = {
        'mid': 50000.0,
        'spread': 10.0,
        'rsi_short': 65.0,
        'ema_slope': 0.01,
        'atr': 500.0,
        'realized_var': 0.0001,
        'pressure': 0.5,
        'book_imbalance': 0.2,
        'regime_soft': {'TREND': 0.7, 'RANGE': 0.2, 'VOLX': 0.1}
    }
    
    context = {
        'inventory': 0.5,
        'equity': 10000.0,
        'unrealized_pnl': 50.0
    }
    
    state = packer.pack(observation, context)
    
    assert state.shape == (10,), f"State should have shape (10,), got {state.shape}"
    assert state[0] == 50000.0, "First feature should be mid price"
    
    print(f"  ✓ State packing working correctly")
    print(f"    - State shape: {state.shape}")
    print(f"    - First few features: {state[:3]}")
    
    # Test with history
    for _ in range(2):
        state = packer.pack(observation, context)
    
    state_with_history = packer.pack_with_history(observation, context)
    expected_history_dim = 10 * 3  # state_dim * history_length
    assert state_with_history.shape == (expected_history_dim,), \
        f"State with history should have shape ({expected_history_dim},), got {state_with_history.shape}"
    
    print(f"  ✓ State with history working correctly")
    print(f"    - History shape: {state_with_history.shape}")


def test_state_normalization():
    """Test state normalization."""
    print("\nTesting State Normalization...")
    
    packer = StatePacker(state_dim=5, normalize=True, history_length=1)
    
    observation = {
        'mid': 50000.0,
        'spread': 10.0,
        'rsi_short': 65.0,
        'ema_slope': 0.01,
        'atr': 500.0
    }
    
    # Pack multiple states to build statistics
    states = []
    for i in range(100):
        obs = observation.copy()
        obs['mid'] = 50000.0 + i * 10
        state = packer.pack(obs)
        states.append(state)
    
    # Check that normalization is working
    final_state = states[-1]
    assert not np.any(np.isnan(final_state)), "Normalized state should not have NaN"
    assert not np.any(np.isinf(final_state)), "Normalized state should not have inf"
    
    print(f"  ✓ State normalization working correctly")
    print(f"    - Mean: {packer.mean[:3]}")
    print(f"    - Std: {packer.std[:3]}")


def test_actor_critic():
    """Test actor-critic policy."""
    print("\nTesting Actor-Critic Policy...")
    
    state_dim = 10
    action_dim = 3
    
    policy = ActorCritic(state_dim, action_dim)
    
    # Test forward passes
    state = np.random.randn(state_dim)
    
    mean, std = policy.forward_actor(state)
    assert mean.shape == (action_dim,), f"Actor mean should have shape ({action_dim},)"
    assert std.shape == (action_dim,), f"Actor std should have shape ({action_dim},)"
    
    value = policy.forward_critic(state)
    assert isinstance(value, float), "Critic should return a float"
    
    print(f"  ✓ Actor-Critic forward passes working")
    print(f"    - Actor mean shape: {mean.shape}")
    print(f"    - Critic value: {value:.4f}")


def test_action_selection():
    """Test action selection."""
    print("\nTesting Action Selection...")
    
    state_dim = 10
    action_dim = 3
    
    policy = ActorCritic(state_dim, action_dim)
    state = np.random.randn(state_dim)
    
    # Test stochastic action
    action, log_prob = policy.select_action(state, deterministic=False)
    assert action.shape == (action_dim,), f"Action should have shape ({action_dim},)"
    assert isinstance(log_prob, (float, np.floating)), "Log prob should be a float"
    
    # Test deterministic action
    action_det, log_prob_det = policy.select_action(state, deterministic=True)
    assert action_det.shape == (action_dim,), f"Deterministic action should have shape ({action_dim},)"
    assert log_prob_det == 0.0, "Deterministic action should have log_prob = 0"
    
    print(f"  ✓ Action selection working correctly")
    print(f"    - Stochastic action: {action}")
    print(f"    - Log prob: {log_prob:.4f}")
    print(f"    - Deterministic action: {action_det}")


def test_evaluate_actions():
    """Test action evaluation."""
    print("\nTesting Action Evaluation...")
    
    state_dim = 5
    action_dim = 2
    batch_size = 10
    
    policy = ActorCritic(state_dim, action_dim)
    
    states = np.random.randn(batch_size, state_dim)
    actions = np.random.randn(batch_size, action_dim)
    
    values, log_probs, entropies = policy.evaluate_actions(states, actions)
    
    assert values.shape == (batch_size,), f"Values should have shape ({batch_size},)"
    assert log_probs.shape == (batch_size,), f"Log probs should have shape ({batch_size},)"
    assert entropies.shape == (batch_size,), f"Entropies should have shape ({batch_size},)"
    
    print(f"  ✓ Action evaluation working correctly")
    print(f"    - Values shape: {values.shape}")
    print(f"    - Mean value: {np.mean(values):.4f}")
    print(f"    - Mean entropy: {np.mean(entropies):.4f}")


def test_compute_gradients():
    """Test gradient computation."""
    print("\nTesting Gradient Computation...")
    
    state_dim = 5
    action_dim = 2
    batch_size = 10
    
    policy = ActorCritic(state_dim, action_dim)
    
    states = np.random.randn(batch_size, state_dim)
    actions = np.random.randn(batch_size, action_dim)
    advantages = np.random.randn(batch_size)
    returns = np.random.randn(batch_size)
    old_log_probs = np.random.randn(batch_size)
    
    gradients = policy.compute_gradients(
        states, actions, advantages, returns, old_log_probs
    )
    
    assert 'policy_loss' in gradients, "Should compute policy loss"
    assert 'value_loss' in gradients, "Should compute value loss"
    assert 'entropy' in gradients, "Should compute entropy"
    
    assert isinstance(gradients['policy_loss'], (float, np.floating)), \
        "Policy loss should be a scalar"
    assert isinstance(gradients['value_loss'], (float, np.floating)), \
        "Value loss should be a scalar"
    
    print(f"  ✓ Gradient computation working correctly")
    print(f"    - Policy loss: {gradients['policy_loss']:.4f}")
    print(f"    - Value loss: {gradients['value_loss']:.4f}")
    print(f"    - Entropy: {gradients['entropy']:.4f}")


def test_gae():
    """Test Generalized Advantage Estimation."""
    print("\nTesting GAE Computation...")
    
    # Create sample trajectory
    rewards = np.array([1.0, 0.5, 0.8, 1.2, 0.0])
    values = np.array([0.9, 0.6, 0.7, 1.0, 0.1])
    dones = np.array([0, 0, 0, 0, 1])
    
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95)
    
    assert advantages.shape == rewards.shape, "Advantages should have same shape as rewards"
    assert returns.shape == rewards.shape, "Returns should have same shape as rewards"
    
    # Check that advantages + values = returns
    np.testing.assert_array_almost_equal(advantages + values, returns)
    
    print(f"  ✓ GAE computation working correctly")
    print(f"    - Advantages: {advantages}")
    print(f"    - Returns: {returns}")


def test_normalize_advantages():
    """Test advantage normalization."""
    print("\nTesting Advantage Normalization...")
    
    advantages = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = normalize_advantages(advantages)
    
    # Check mean ~0 and std ~1
    assert abs(np.mean(normalized)) < 1e-6, "Normalized advantages should have mean ~0"
    assert abs(np.std(normalized) - 1.0) < 1e-6, "Normalized advantages should have std ~1"
    
    print(f"  ✓ Advantage normalization working correctly")
    print(f"    - Original: {advantages}")
    print(f"    - Normalized: {normalized}")
    print(f"    - Mean: {np.mean(normalized):.6f}, Std: {np.std(normalized):.6f}")


def test_full_rl_loop():
    """Test a complete RL training loop."""
    print("\nTesting Complete RL Loop...")
    
    state_dim = 10
    action_dim = 3
    
    # Initialize components
    policy = ActorCritic(state_dim, action_dim, learning_rate=1e-3)
    buffer = ReplayBuffer(capacity=100)
    packer = StatePacker(state_dim, normalize=False)
    
    # Simulate trajectory
    for step in range(20):
        observation = {
            'mid': 50000.0 + step * 10,
            'spread': 10.0,
            'rsi_short': 50.0 + step,
            'ema_slope': 0.01,
            'atr': 500.0
        }
        
        state = packer.pack(observation)
        action, log_prob = policy.select_action(state)
        value = policy.forward_critic(state)
        
        reward = np.random.randn()  # Simulated reward
        next_observation = {
            'mid': 50000.0 + (step + 1) * 10,
            'spread': 10.0,
            'rsi_short': 50.0 + step + 1,
            'ema_slope': 0.01,
            'atr': 500.0
        }
        next_state = packer.pack(next_observation)
        done = (step == 19)
        
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        buffer.push(exp)
    
    # Extract trajectory for training
    trajectory = buffer.sample_trajectory()
    assert len(trajectory) == 20, f"Should have 20 experiences, got {len(trajectory)}"
    
    # Prepare batch
    states = np.array([exp.state for exp in trajectory])
    actions = np.array([exp.action for exp in trajectory])
    rewards = np.array([exp.reward for exp in trajectory])
    values = np.array([exp.value for exp in trajectory])
    dones = np.array([exp.done for exp in trajectory])
    old_log_probs = np.array([exp.log_prob for exp in trajectory])
    
    # Compute advantages
    advantages, returns = compute_gae(rewards, values, dones)
    advantages = normalize_advantages(advantages)
    
    # Compute gradients
    gradients = policy.compute_gradients(
        states, actions, advantages, returns, old_log_probs
    )
    
    # Update policy
    policy.update(gradients)
    
    print(f"  ✓ Complete RL loop working correctly")
    print(f"    - Collected {len(trajectory)} experiences")
    print(f"    - Policy loss: {gradients['policy_loss']:.4f}")
    print(f"    - Value loss: {gradients['value_loss']:.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("PRATE RL Module Tests")
    print("=" * 60)
    
    test_replay_buffer()
    test_state_packer()
    test_state_normalization()
    test_actor_critic()
    test_action_selection()
    test_evaluate_actions()
    test_compute_gradients()
    test_gae()
    test_normalize_advantages()
    test_full_rl_loop()
    
    print("\n" + "=" * 60)
    print("✓ All RL module tests passed successfully!")
    print("=" * 60)
