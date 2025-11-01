# PRATE - Prime-Resonant Adaptive Trading Ecology

A sophisticated adaptive trading system that models trading as an entropy-minimizing observer ecology operating in a financial environment.

## Overview

PRATE is an advanced algorithmic trading framework that combines:
- **Prime-Hilbert Embedding**: Market observations embedded in prime-indexed Hilbert space
- **Entropy-Based Control**: Continuous adaptation through entropy minimization
- **Holographic Memory**: Complex-phase associative memory for context-strategy recall
- **Multi-Guild Architecture**: Specialized trading strategies (trend-follow, mean-revert, breakout, etc.)
- **Reinforcement Learning**: Optional RL integration for continuous parameter optimization

## Key Features

- **Mathematical Foundation**: Grounded in rigorous mathematical operators (Π projection, E_τ entropy collapse, M measurement-to-action)
- **Adaptive Learning**: Phase-learning updates and entropy thermostats for continuous improvement
- **Risk Management**: KAM protection with leverage, exposure, drawdown, and latency guards
- **Multi-Strategy**: Guild-based architecture supporting different trading archetypes
- **Memory System**: Holographic memory for few-shot regime adaptation
- **Backtesting**: Comprehensive simulation and validation framework

## Architecture

The system consists of several key layers:

1. **Data Ingestion**: WebSocket or CSV replay for candles, order-book, funding, trades
2. **Feature Engine**: Technical and microstructure feature computation with integer discretization
3. **Ecology Core**: Manages guilds, phase vectors, τ controllers, Π-bandits, and HilbertRefine loop
4. **RL Module**: Optional PPO/SAC head for continuous parameter optimization
5. **Holographic Memory**: Complex-phase associative store for context→strategy recall
6. **Risk Kernel**: KAM protection for safe execution
7. **Execution Interface**: Abstract API for order management
8. **Persistence & Dashboard**: Metrics DB, entropy/coherence graphs, trade audit, PnL tracking

## Guild Structure

| Guild | Strategy Archetype | Primary Primes | Reward Bias |
|-------|-------------------|----------------|-------------|
| G_TF  | Trend-follow      | small (2–31)   | momentum    |
| G_MR  | Mean-revert       | mid (37–97)    | reversion   |
| G_BR  | Breakout          | high primes    | volatility expansion |
| G_LM  | Liquidity make    | mixed          | fee rebates |
| G_FA  | Funding carry     | period-based   | carry yield |
| G_OBS | Observation       | random         | information gain |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation Mode

```python
from prate.ecology import Ecology
from prate.simulator import Simulator

# Initialize components
simulator = Simulator(market_data, fee_schedule, slippage_model)
ecology = Ecology(config)

# Run simulation
for observation in data_stream:
    ecology.step(observation)
```

### Paper Trading Mode

```python
from prate.ecology import Ecology
from prate.execution import LiveExecInterface

# Connect to exchange (paper mode)
exec_interface = LiveExecInterface(exchange='binance', paper_mode=True)
ecology = Ecology(config, exec_interface=exec_interface)

# Run live with paper trading
ecology.run()
```

## Performance Metrics

The system tracks:

**Financial Metrics:**
- Net PnL, Sharpe/Sortino ratios
- Maximum drawdown, hit rate, expectancy

**Structural Metrics:**
- Entropy H(Ψ), coherence, τ-error
- Phase drift, bandit regret

**Memory Metrics:**
- Retrieval lift, interference ratio, novelty index

## Security & Compliance

- API key isolation and encryption
- Exchange rate limit compliance
- Independent risk limits enforced before any live order
- All exchange connectivity is external to the model core
- Trade intents (side, size, price, validity) are validated before execution

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Core / RL | Python (NumPy, PyTorch, JAX) |
| Feature Engine | Rust or C++ (bindings to Python) |
| Holographic Memory | Complex tensor ops on GPU (cuFFT) |
| Dashboard | FastAPI + React/Vue (optional) |
| Persistence | PostgreSQL / Parquet |
| Backtester | Vectorized Python, Cython core |

## Validation Plan

1. **Unit Tests**: Operators Π, E, M, τ-controller stability
2. **Backtest Validation**: Walk-forward splits on historical data
3. **Monte-Carlo Robustness**: Parameter perturbations, latency, fee noise
4. **Paper-Trading Trial**: Live data with simulated execution
5. **Audit**: Risk invariants and entropy bounds confirmation

## Expected Outcomes

- Adaptive trading system shifting between regimes through prime-phase resonance
- Continuous online improvement from entropy feedback
- Holographic recall of profitable contexts for few-shot regime adaptation
- Stable long-term behavior via KAM protection and entropy thermostats

## Development Status

See [STATUS.md](STATUS.md) for current implementation status.

## Documentation

See [DESIGN.md](DESIGN.md) for comprehensive design documentation.

## License

Copyright (c) 2024. All rights reserved.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software. Always validate in simulation before considering any live trading, and consult with financial professionals.
