#!/usr/bin/env python3
"""
Example usage of the PRATE system.

This demonstrates basic initialization and a simple simulation loop.
"""

import numpy as np
from prate.ecology import Ecology
from prate.simulator import Simulator
from prate.types import Observation, RegimeID

# Configuration for the ecology
config = {
    'primes': {
        'M': 37  # Use first 37 primes
    },
    'guilds': [
        {
            'id': 'TF',  # Trend-follow
            'bases': [
                [2, 3, 5, 7, 11, 13, 17],
                [2, 3, 5, 11, 19, 23]
            ]
        },
        {
            'id': 'MR',  # Mean-revert
            'bases': [
                [37, 41, 43, 47, 53],
                [59, 61, 67, 71, 73]
            ]
        }
    ],
    'bandit': {
        'algo': 'thompson',
        'prior_weight': 0.3
    },
    'tau_controller': {
        'H_star': 2.0,
        'kP': 0.15,
        'kI': 0.02
    },
    'phase': {
        'eta0': 0.02,
        'protected': []
    },
    'holographic': {
        'gamma': 0.995,
        'eta': 0.05
    },
    'risk': {
        'max_trade_risk_pct': 0.5,
        'daily_dd_pct': 2.5,
        'var_max': 0.03,
        'leverage_cap': 3.0,
        'max_position': 10.0
    },
    'training': {
        'reward_weights': {
            'fees': 1.0,
            'turnover': 0.2,
            'variance': 0.1,
            'drawdown': 2.0
        },
        'write_threshold': 0.0
    },
    'key': {
        'weights': {'ret_bucket': 1.0, 'imb_bucket': 0.8},
        'mask': None
    },
    'value': {
        'beta_basis': 0.8,
        'beta_phi': 0.6,
        'beta_tau': 0.4,
        'beta_hint': 0.3,
        'k_phi': 1.0,
        'k_tau': 1.0
    }
}

# Create simulator
simulator = Simulator(
    market_data=None,
    fee_schedule={'maker': 0.0005, 'taker': 0.001},
    slippage_model={'market_bps': 5.0}
)

# Create ecology
print("Initializing PRATE Ecology...")
ecology = Ecology(config, exec_interface=simulator)
print(f"✓ Ecology initialized with {len(ecology.P)} primes and {len(ecology.bases)} bases\n")

# Create sample observations
def create_sample_observation(ts: int, symbol: str = "BTCUSDT") -> Observation:
    """Create a sample observation for testing."""
    return Observation(
        ts=ts,
        symbol=symbol,
        mid=50000.0,
        bid=49995.0,
        ask=50005.0,
        spread=10.0,
        last_px=50000.0,
        last_qty=0.1,
        vol_1s=100.0,
        vol_1m=6000.0,
        book_imbalance=0.05,
        pressure=0.02,
        realized_var=0.001,
        atr=500.0,
        rsi_short=55.0,
        ema_slope=0.001,
        inventory=0.0,
        equity=100000.0,
        unrealized_pnl=0.0,
        funding_rate=0.0001,
        time_of_day_bucket=12,
        regime_soft={
            RegimeID.TREND: 0.4,
            RegimeID.RANGE: 0.3,
            RegimeID.VOLX: 0.2,
            RegimeID.QUIET: 0.1,
            RegimeID.UNKNOWN: 0.0
        },
        features_vec=np.array([0.001, 0.05, 0.5]),
        features_disc={
            'ret_bucket': 5,
            'imb_bucket': 3,
            'regime_id': 0,
            'tod_bucket': 12
        }
    )

# Run simulation loop
print("Running simulation...")
print("-" * 60)

num_steps = 10
for i in range(num_steps):
    ts = 1700000000000 + i * 1000  # Simulated timestamps
    obs = create_sample_observation(ts)
    
    # Execute ecology step
    ecology.step(obs)
    
    # Print progress
    if (i + 1) % 5 == 0:
        metrics = ecology.get_metrics()
        print(f"Step {i+1}/{num_steps}")
        print(f"  Trades: {metrics['trades']}")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        if metrics['entropy_history']:
            print(f"  Current Entropy: {metrics['entropy_history'][-1]:.3f}")
        print()

print("-" * 60)
print("Simulation complete!")

# Final metrics
final_metrics = ecology.get_metrics()
print("\nFinal Metrics:")
print(f"  Total Trades: {final_metrics['trades']}")
print(f"  Total PnL: ${final_metrics['total_pnl']:.2f}")
print(f"  Avg Entropy: {np.mean(final_metrics['entropy_history']):.3f}")

# Bandit statistics
print("\nBandit Statistics:")
for basis_id, (mean_reward, count) in ecology.bandit.get_stats().items():
    print(f"  {basis_id}: mean={mean_reward:.4f}, n={count}")

print("\n✓ Example completed successfully!")
