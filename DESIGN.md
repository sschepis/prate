# PRATE Design Document

## Prime-Resonant Adaptive Trading Ecology (PRATE)

## 1. Overview

The PRATE architecture models trading as an entropy-minimizing observer ecology operating in a financial environment. Each observer is an integer-program agent embedded in a prime-indexed Hilbert space; its dynamics follow the operators Π (projection), E_τ (entropy collapse), and M (measurement-to-action). Continuous adaptation, phase learning, and holographic memory supply non-stop improvement from market feedback.

```
O_i : ℤ^k → ℤ,     n_i = O_i(s_i,θ_i,ξ_i)
```

The ecology evolves toward coherent low-entropy policies that maximize long-term expected utility (profit adjusted for risk).

---

## 2. Mathematical Core

### 2.1 Prime-Hilbert Embedding

Let P={p_1,...,p_M} be the first M primes. Each discretized market observation x_t∈ℤ is embedded as:

```
Ψ(x_t) = {p ↦ (a_p(x_t), φ_p(x_t))}_{p∈P}
```

with:

```
a_p(x_t) = 1 - (x_t mod p)/p
φ_p(x_t) = 2π(x_t mod p + φ_p)/p
```

### 2.2 Operators

```
Π_B Ψ = {(a_p,φ_p) | p∈B}
E_τ(Ψ) = soft-top-k(Ψ, H(Ψ)≤τ)
M(Ψ) = mix(a_p,φ_p) ↦ m
```

The composite refinement is:
```
R(n_0) = f(n_0, M(E_τ ∘ Π_B Ψ))
```

### 2.3 Reward Functional

For a trade or action a_t:

```
r_t = ΔPnL_t 
      - λ_f c_t 
      - λ_s |Δq_t| 
      - λ_v Var(PnL)_{t:t+T} 
      - λ_d 𝟙_drawdown
```

Expected discounted return J = 𝔼[Σ_t γ^t r_t] defines the learning objective.

### 2.4 Phase-Learning Update

```
φ_p ← φ_p + η_t(r_t - r̄_t) Residue_p(a_t)
```

### 2.5 Entropy Thermostat

Maintain H(Ψ_t) ≈ H* via:

```
τ_{t+1} = τ_t - k_P(H(Ψ_t) - H*) - k_I Σ_{u≤t}(H(Ψ_u) - H*)
```

### 2.6 Bandit-Style Basis Selection

Each candidate basis B_j⊂P has posterior mean reward R̂_j; select by Thompson or UCB sampling:

```
B_t = argmax_j(R̂_j + β√(2ln t / n_j))
```

### 2.7 Holographic Memory

Memory tensor ℋ_B∈ℂ^{|P|} accumulates bound key-value pairs:

```
M_t = K_t ⊛ V_t
ℋ_B ← γℋ_B + ηM_t
V̂_t = corr(K_q, ℋ_B)
```

Keys encode context (regime, symbol, features); values encode profitable parameter deltas.

---

## 3. System Architecture

| Layer | Function |
|-------|----------|
| Data Ingestion | Websocket or CSV replay; candles, order-book, funding, trades |
| Feature Engine | Computes technical and microstructure features, discretizes to integers for embedding |
| Ecology Core | Manages guilds, phase vectors, τ controllers, Π-bandits, HilbertRefine loop |
| RL Module | PPO/SAC head for continuous param optimization inside each style |
| Holographic Memory | Complex-phase associative store for context→strategy recall |
| Risk Kernel | KAM protection: leverage, exposure, drawdown, latency guards |
| Execution Interface | Abstract API to send/cancel orders (stubbed in simulation) |
| Persistence & Dashboard | Metrics DB, entropy/coherence graphs, trade audit, PnL, memory diagnostics |

---

## 4. Guild Structure

| Guild | Strategy Archetype | Primary Primes B | Reward Bias |
|-------|-------------------|------------------|-------------|
| G_TF | Trend-follow | small primes (2–31) | momentum |
| G_MR | Mean-revert | mid primes (37–97) | reversion |
| G_BR | Breakout | high primes | volatility expansion |
| G_LM | Liquidity make | mixed | fee rebates |
| G_FA | Funding carry | selected by period of funding cycle | carry yield |
| G_OBS | Observation / explore | random | information gain |

Guilds share telemetry of top-k supports, coherence, and τ deviations every T_sync seconds.

---

## 5. RL Integration

State s_t=(o_t,Ψ_t,φ_B,H(Ψ_t),B), action a_t=(Δq,style params), reward r_t as above.

Actor–critic optimizes continuous parameters while the ecology handles discrete style selection and phase evolution.

Gradient step:
```
∇_θ J = 𝔼_t[∇_θ log π_θ(a_t|s_t)A_t]
```

The advantage A_t = r_t + γV(s_{t+1}) - V(s_t) is entropy-regularized by E_τ.

---

## 6. Risk and Capital Formalism

```
Position size f_i = min(f_max, κμ_i/σ_i²)
VaR_99 < R_max
DD_daily < D_max
```

Stops = α · ATR; time-outs = T_max

Entropy thermostat freezes adaptation when risk metrics breach limits.

---

## 7. Algorithmic Loop (Sim / Paper Mode)

1. Observe o_t, embed Ψ_t = PrimeEmbed(o_t, P, φ_B)
2. Retrieve V_prior from ℋ_B
3. Select basis B_t via bandit; adjust τ_t by controller
4. Generate base proposal n_0 from policy; refine n=R(n_0;Ψ_t)
5. Evaluate trade in simulator; obtain r_t
6. Update φ_B, τ_t, bandit posteriors, holographic memory
7. Apply KAM protection and risk constraints

---

## 8. Performance Metrics

**Financial:**
- Net PnL, Sharpe/Sortino, max drawdown, hit rate, expectancy

**Structural:**
- H(Ψ), coherence, τ-error, φ-drift, bandit regret

**Memory:**
- Retrieval lift, interference ratio, novelty index

---

## 9. Security and Compliance

All exchange connectivity is external to the model core. The PRATE system outputs trade intents (side, size, price, validity).

Execution and custody components must implement:
- API key isolation and encryption
- Compliance with exchange rate limits
- Independent risk limits enforced before any live order

---

## 10. Implementation Stack

| Component | Suggested Tech |
|-----------|---------------|
| Core / RL | Python (NumPy, PyTorch, JAX) |
| Feature Engine | Rust or C++ for speed; bindings to Python |
| Holographic Memory | Complex tensor ops on GPU (cuFFT) |
| Dashboard | FastAPI + React/Vue (optional) |
| Persistence | PostgreSQL / Parquet for metrics |
| Backtester | Vectorized Python, Cython core for fills |

---

## 11. Validation Plan

1. **Unit tests** for operators Π, E, M, τ-controller stability
2. **Backtest validation** on 1-s or 1-m bars; walk-forward splits
3. **Monte-Carlo robustness**: parameter perturbations, random latency, fee noise
4. **Paper-trading trial**: live data, simulated execution
5. **Audit**: confirm risk invariants and entropy bounds before any real trade interface

---

## 12. Expected Outcomes

- Adaptive trading system that shifts between regimes through prime-phase resonance
- Continuous online improvement from entropy feedback
- Holographic recall of profitable contexts enabling few-shot regime adaptation
- Stable long-term behavior due to KAM protection and entropy thermostats

---

## 13. Next Steps

1. Implement the simulation/backtest core with all mathematical modules above
2. Train RL + ecology on historical data
3. Validate metrics offline
4. Only after successful audits, connect to your exchange layer under separate, human-approved risk controls

---

## Module Interfaces & Implementation Details

### Global Types & Conventions

```python
Time           = int | datetime           # exchange/server epoch milliseconds
Symbol         = str                      # e.g., "BTCUSDT"
Side           = enum{BUY, SELL}
Venue          = str                      # e.g., "MEXC"
PrimeIndex     = int                      # p ∈ P (first M primes)
BasisID        = str                      # name/uid of a basis subset B⊆P
GuildID        = enum{TF, MR, BR, LM, FA, OBS}
RegimeID       = enum{TREND, RANGE, VOLX, QUIET, UNKNOWN}
```

### Observation Structure

```python
class Observation:
    ts: Time
    symbol: Symbol
    mid, bid, ask: float
    spread: float
    last_px, last_qty: float
    vol_1s, vol_1m: float
    book_imbalance, pressure, realized_var, atr, rsi_short, ema_slope: float
    inventory, equity, unrealized_pnl: float
    funding_rate, time_of_day_bucket: float|int
    regime_soft: dict[RegimeID, float]     # soft scores ∈ [0,1]
    features_vec: np.ndarray[float, F]     # packed numeric features (continuous)
    features_disc: dict[str,int]           # discretized feature IDs for prime embed
```

### Action Structure

```python
class Action:
    style: GuildID
    delta_q: float        # target position delta
    params: dict[str, float|int]  # style-specific knobs
```

---

## Detailed Component Specifications

See sections below for detailed pseudocode and implementation guidance for each major component:

1. **Feature Engine**: Maintains rolling windows and produces discretized features
2. **Prime-Hilbert Embedding**: Maps integer features to prime-indexed amplitudes and phases
3. **Hilbert Operators**: Projection, collapse, measurement, and refinement operations
4. **Bandit for Basis Selection**: Thompson sampling or UCB for basis/style selection
5. **Entropy Thermostat**: PI controller for maintaining target entropy
6. **Phase Learner**: Online φ updates based on reward feedback
7. **Holographic Memory**: Complex HRR for context-strategy associations
8. **RL Bridge**: Optional continuous parameter optimization
9. **Risk Kernel**: KAM protection and safety checks
10. **Execution Interface**: Abstract API for order management
11. **Simulator/Backtester**: Fill models and market replay
12. **Ecology Core**: Main coordination loop

Each component is designed to be modular and testable independently.

---

## Configuration Schema

```yaml
primes:
  M: 73
  list: [2,3,5,...]

guilds:
  - id: TF
    bases: [[2,3,5,7,11,13,17], [2,3,5,11,19,23]]
  - id: MR
    bases: [[37,41,43,47,53], [59,61,67,71,73]]

tau_controller:
  H_star: 2.0
  kP: 0.15
  kI: 0.02

bandit:
  algo: thompson
  prior_weight: 0.3

phase:
  eta0: 0.02
  protected: []

holographic:
  gamma: 0.995
  eta: 0.05

risk:
  max_trade_risk_pct: 0.5
  daily_dd_pct: 2.5
  var_max: 0.03
  leverage_cap: 3.0

training:
  reward_weights:
    fees: 1.0
    turnover: 0.2
    variance: 0.1
    drawdown: 2.0
  write_threshold: 0.0
```

---

## Testing & Validation

### Unit Tests
- hilbert_entropy, project/collapse/measure
- τ controller step response (setpoint tracking, no overshoot)
- bandit convergence on synthetic stationary rewards
- holo bind/correlate orthogonality and interference bounds

### Property Tests
- risk kernel never emits intents that breach limits
- phase learner bounded update under adversarial rewards

### Backtest Protocols
- walk-forward split: train 60d, test 30d; rotate 2023–2025
- fee/slip sensitivity ±50%
- latency injection and partial fill randomness

### Go/No-Go Criteria
- Sharpe ≥ 1.0 after fees over ≥ 500 trades
- Max DD ≤ configured D_max
- Stable τ error |H(Ψ)−H*| median < 0.3
- Positive memory-hit lift vs. disabled baseline

---

## Build Order (Modular Development)

1. **FeatureEngine** + **PrimeEmbedder** + **Operators**
2. **TauController** + **Bandit** + **PhaseLearner**
3. **HoloMemory** enc/dec + residue features
4. **Simulator** + **RiskKernel** + **Ecology**
5. **Dashboard** + **Metrics**
6. **(Optional)** RLPolicy head

---

## Mathematical Notation & Conventions

- Primes P={p_1,...,p_M} with an index map idx(p_j)=j
- Complex vectors live in ℂ^M (NumPy complex128 arrays)
- wrap(x) wraps a real angle to [-π,π)
- unit(e^{iθ}) means magnitude 1 (unit phasor)
- Stable seeded hash h(·) → 64-bit integer
- Temperature/scale controls (τ_phase, τ_amp) tune sharpness and salience
