"""
Ecology Core - Main coordination loop for PRATE system.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from .types import Observation, Action, TradeIntent, Side, Basis, GuildID
from .embedding import PrimeEmbedder, hilbert_entropy
from .operators import Operators
from .bandit import BasisBandit
from .tau_controller import TauController
from .phase_learner import PhaseLearner, Baseline
from .holo_memory import HoloMemory
from .encoders import encode_key, encode_value, decode_value
from .residue import residue_features
from .risk import RiskKernel


class Ecology:
    """
    Ecology Core - coordinates all PRATE components.
    
    Main loop:
    1. Observe and embed
    2. Read holographic memory
    3. Select basis via bandit
    4. Generate and refine proposal
    5. Execute trade
    6. Update learning components
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        exec_interface: Optional[Any] = None
    ):
        """
        Initialize ecology.
        
        Args:
            config: Configuration dictionary
            exec_interface: Execution interface (Simulator or LiveExec)
        """
        self.cfg = config
        self.exec = exec_interface
        
        # Initialize primes
        self.P = self._generate_primes(config['primes']['M'])
        
        # Initialize components
        self.embedder = PrimeEmbedder(self.P, len(self.P))
        self.operators = Operators()
        
        # Create bases from config
        self.bases = self._create_bases(config['guilds'])
        self.bandit = BasisBandit(self.bases, algo=config['bandit']['algo'])
        
        # Tau controller
        tau_cfg = config['tau_controller']
        self.tau_ctl = TauController(
            H_star=tau_cfg['H_star'],
            kP=tau_cfg['kP'],
            kI=tau_cfg['kI'],
            bounds=(0.5, 5.0)
        )
        
        # Phase learners (one per guild)
        phase_cfg = config['phase']
        self.phase_learners = {}
        for guild_cfg in config['guilds']:
            guild_id = guild_cfg['id']
            self.phase_learners[guild_id] = PhaseLearner(
                self.P,
                eta0=phase_cfg['eta0'],
                protected=set(phase_cfg.get('protected', []))
            )
        
        # Current phase vector (shared)
        self.phi_vec = np.zeros(len(self.P))
        
        # Holographic memory
        holo_cfg = config['holographic']
        self.holo = HoloMemory(
            self.P,
            gamma=holo_cfg['gamma'],
            eta=holo_cfg['eta']
        )
        
        # Risk kernel
        self.risk = RiskKernel(config['risk'])
        
        # Baseline for rewards
        self.baseline = Baseline(alpha=0.1)
        
        # Metrics
        self.metrics = {
            'trades': 0,
            'total_pnl': 0.0,
            'entropy_history': []
        }
        
        self.state = "ACTIVE"
    
    def _generate_primes(self, M: int) -> List[int]:
        """Generate first M prime numbers."""
        primes = []
        candidate = 2
        while len(primes) < M:
            is_prime = True
            for p in primes:
                if p * p > candidate:
                    break
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes
    
    def _create_bases(self, guilds_config: List[Dict]) -> List[Basis]:
        """Create basis objects from config."""
        bases = []
        for guild_cfg in guilds_config:
            guild_id = guild_cfg['id']
            for i, prime_list in enumerate(guild_cfg['bases']):
                basis = Basis(
                    id=f"{guild_id}_{i}",
                    primes=prime_list
                )
                bases.append(basis)
        return bases
    
    def step(self, o: Observation) -> None:
        """
        Execute one ecology step.
        
        Args:
            o: Market observation
        """
        if self.state != "ACTIVE":
            return
        
        # 1) Prime embed and entropy
        Ψ = self.embedder.embed(o, self.phi_vec)
        HΨ = hilbert_entropy(Ψ)
        self.metrics['entropy_history'].append(HΨ)
        
        # 2) Holo read (priors)
        Kq = encode_key(o, self.cfg.get('key', {}), self.P, guild_id=None)
        Vhat = self.holo.read(Kq)
        
        # Decode priors
        basis_catalog = {b.id: b.primes for b in self.bases}
        B_prior, dphi_prior, dtau_prior, hints, conf = decode_value(
            Vhat, self.P, basis_catalog, self.cfg.get('value', {}),
            known_hint_names=[]
        )
        
        # 3) Sample basis/style with bandit
        B = self.bandit.sample_with_prior(B_prior)
        tau = self.tau_ctl.step(HΨ + dtau_prior)
        
        # 4) Base proposal (simple rule-based for now)
        n0 = self._base_proposal(B, o, hints)
        
        # 5) Hilbert refinement → action params
        params = self.operators.refine(n0, Ψ, B, tau)
        
        # 6) Risk wrap → trade intent
        intent = self._params_to_intent(o.symbol, params)
        
        if not self.exec:
            return  # No execution interface
        
        account_state = self.exec.get_account_state() if hasattr(self.exec, 'get_account_state') else {}
        intent_vetted = self.risk.vet_intent(intent, account_state)
        
        if not intent_vetted:
            return
        
        # 7) Execute (sim or abstract)
        cid = self.exec.send(intent_vetted)
        
        # Step simulator if applicable
        if hasattr(self.exec, 'step'):
            self.exec.step(o.ts)
        
        # 8) Evaluate + reward
        fills = self.exec.poll()
        r_t = self._reward_from_fills(fills, o)
        
        # 9) Bandit, φ, Holo write, Baseline
        self.bandit.update(B.id, r_t)
        
        # Compute residue features
        res_feats = residue_features(o, params, self.P, topk=12)
        
        # Update phase learner for this guild
        guild_id = self._basis_to_guild(B.id)
        baseline_val = self.baseline.update(r_t)
        self.phi_vec = self.phase_learners[guild_id].step(r_t, baseline_val, res_feats)
        
        # Write to holographic memory if reward above threshold
        if r_t > self.cfg['training'].get('write_threshold', 0.0):
            K = encode_key(o, self.cfg.get('key', {}), self.P, guild_id=guild_id)
            V = encode_value(
                B,
                dphi=self.phi_vec,
                dtau=0.0,
                hints={},
                cfg=self.cfg.get('value', {}),
                P=self.P
            )
            gain = 1.0 + max(0, r_t)  # Simple gain function
            self.holo.write(K, V, gain=gain)
        
        # 10) Safety & logging
        self.metrics['trades'] += len(fills)
        self.metrics['total_pnl'] += sum(f.get('pnl', 0.0) for f in fills)
        
        current_metrics = {
            'daily_dd': -100.0 * self.risk.daily_pnl / self.risk.initial_equity,
            'var_99': 0.0,  # Placeholder
            'entropy_diverged': abs(HΨ - self.tau_ctl.H_star) > 2.0
        }
        
        if self.risk.should_halt(current_metrics):
            self.state = "HALT"
            print(f"Trading halted: {current_metrics}")
    
    def _base_proposal(
        self, 
        B: Basis, 
        o: Observation, 
        hints: Dict
    ) -> Dict[str, Any]:
        """Generate base proposal."""
        # Simple proposal: small position change
        return {
            'delta_q': 0.01,  # Small position change
            'style_id': 0,
        }
    
    def _params_to_intent(self, symbol: str, params: Dict) -> TradeIntent:
        """Convert parameters to trade intent."""
        delta_q = params.get('delta_q', 0.0)
        
        return TradeIntent(
            symbol=symbol,
            side=Side.BUY if delta_q > 0 else Side.SELL,
            qty=abs(delta_q),
            price=None,  # Market order
            tif='GTC',
            post_only=False,
            client_id=f"prate_{np.random.randint(1000000)}",
            meta=params
        )
    
    def _reward_from_fills(
        self, 
        fills: List[Dict], 
        o: Observation
    ) -> float:
        """Compute reward from fills."""
        if not fills:
            return 0.0
        
        weights = self.cfg['training']['reward_weights']
        
        total_reward = 0.0
        for fill in fills:
            pnl = fill.get('pnl', 0.0)
            fee = fill.get('fee', 0.0)
            
            # Reward = PnL - weighted fees
            reward = pnl - weights.get('fees', 1.0) * fee
            total_reward += reward
        
        return total_reward
    
    def _basis_to_guild(self, basis_id: str) -> str:
        """Map basis ID to guild ID."""
        # Extract guild from basis_id (format: "GUILD_INDEX")
        return basis_id.split('_')[0]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
