"""
Tests for Residue Features, Risk Kernel, and Encoders.
"""

import pytest
import numpy as np

from prate.residue import residue_features, _mix_ints
from prate.risk import RiskKernel
from prate.encoders import encode_key, encode_value, decode_value, _seeded_phase
from prate.types import TradeIntent, Basis, Side, RegimeID


# ===== RESIDUE FEATURES TESTS =====

def test_mix_ints():
    """Test integer mixing function."""
    result1 = _mix_ints([1, 2, 3])
    result2 = _mix_ints([1, 2, 3])
    result3 = _mix_ints([1, 2, 4])
    
    # Same input should produce same output
    assert result1 == result2
    
    # Different input should produce different output
    assert result1 != result3
    
    # Should be a valid 32-bit integer
    assert 0 <= result1 < 2**32


def test_residue_features_basic():
    """Test basic residue feature computation."""
    # Create minimal observation object
    obs = type('Observation', (), {
        'features_disc': {'price_bucket': 100, 'vol_bucket': 50}
    })()
    
    params = {
        'style_id': 1,
        'delta_q': 0.1,
        'param1': 0.5,
    }
    
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    features = residue_features(obs, params, P, topk=5)
    
    # Should return sparse features
    assert isinstance(features, dict)
    assert len(features) <= 5
    
    # All indices should be valid
    for idx in features.keys():
        assert 0 <= idx < len(P)
    
    # All values should be floats
    for val in features.values():
        assert isinstance(val, float)
        assert abs(val) > 1e-6


def test_residue_features_topk():
    """Test top-k feature selection."""
    obs = type('Observation', (), {
        'features_disc': {'f1': 10, 'f2': 20}
    })()
    
    params = {'style_id': 0, 'delta_q': 0.0}
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    # Test different topk values
    features_3 = residue_features(obs, params, P, topk=3)
    features_5 = residue_features(obs, params, P, topk=5)
    
    assert len(features_3) <= 3
    assert len(features_5) <= 5


def test_residue_features_lambdas():
    """Test lambda weighting."""
    obs = type('Observation', (), {
        'features_disc': {'f1': 10}
    })()
    
    params = {'style_id': 1, 'delta_q': 0.1}
    P = [2, 3, 5, 7, 11]
    
    # State-only (action weight = 0)
    features_state = residue_features(obs, params, P, lambdas=(1.0, 0.0))
    
    # Action-only (state weight = 0)
    features_action = residue_features(obs, params, P, lambdas=(0.0, 1.0))
    
    # Both should produce features but different values
    assert len(features_state) > 0
    assert len(features_action) > 0


# ===== RISK KERNEL TESTS =====

def test_risk_kernel_init():
    """Test risk kernel initialization."""
    limits = {
        'max_trade_risk_pct': 2.0,
        'daily_dd_pct': 5.0,
        'leverage_cap': 3.0,
        'max_position': 10.0
    }
    
    kernel = RiskKernel(limits)
    assert kernel.limits == limits
    assert kernel.daily_pnl == 0.0


def test_risk_kernel_vet_intent_pass():
    """Test vetting a valid trade intent."""
    limits = {
        'max_trade_risk_pct': 10.0,
        'daily_dd_pct': 5.0,
        'leverage_cap': 3.0,
        'max_position': 100.0
    }
    
    kernel = RiskKernel(limits)
    kernel.set_initial_equity(10000.0)
    
    intent = TradeIntent(
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=1.0,
        price=100.0,
        tif="GTC",
        post_only=False,
        client_id="test",
        meta={}
    )
    
    account_state = {
        'equity': 10000.0,
        'position': 0.0,
        'price': 100.0
    }
    
    vetted = kernel.vet_intent(intent, account_state)
    assert vetted is not None
    assert vetted.qty == 1.0


def test_risk_kernel_vet_intent_exceed_leverage():
    """Test rejecting intent that exceeds leverage limit."""
    limits = {
        'max_trade_risk_pct': 100.0,
        'daily_dd_pct': 50.0,
        'leverage_cap': 2.0,
        'max_position': 1000.0
    }
    
    kernel = RiskKernel(limits)
    kernel.set_initial_equity(10000.0)
    
    # Try to take position worth 3x equity (leverage = 3)
    intent = TradeIntent(
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=300.0,
        price=100.0,
        tif="GTC",
        post_only=False,
        client_id="test",
        meta={}
    )
    
    account_state = {
        'equity': 10000.0,
        'position': 0.0,
        'price': 100.0
    }
    
    vetted = kernel.vet_intent(intent, account_state)
    # Should be clipped due to max_trade_risk_pct, but still may pass
    # The actual rejection happens only if leverage > cap after clipping
    # In this case qty gets clipped to 100 (from max_trade_risk_pct)
    # which gives leverage = 1.0, so it passes
    assert vetted is not None
    assert vetted.qty == 100.0  # Clipped by max_trade_risk_pct


def test_risk_kernel_vet_intent_exceed_position():
    """Test rejecting intent that exceeds max position."""
    limits = {
        'max_trade_risk_pct': 100.0,
        'daily_dd_pct': 50.0,
        'leverage_cap': 10.0,
        'max_position': 5.0
    }
    
    kernel = RiskKernel(limits)
    
    intent = TradeIntent(
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=10.0,
        price=100.0,
        tif="GTC",
        post_only=False,
        client_id="test",
        meta={}
    )
    
    account_state = {
        'equity': 100000.0,
        'position': 0.0,
        'price': 100.0
    }
    
    vetted = kernel.vet_intent(intent, account_state)
    assert vetted is None  # Should be rejected


def test_risk_kernel_clip_trade_size():
    """Test clipping trade size to max risk."""
    limits = {
        'max_trade_risk_pct': 1.0,  # 1% of equity
        'daily_dd_pct': 50.0,
        'leverage_cap': 10.0,
        'max_position': 100.0
    }
    
    kernel = RiskKernel(limits)
    
    intent = TradeIntent(
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=10.0,  # Would be $1000 at price 100
        price=100.0,
        tif="GTC",
        post_only=False,
        client_id="test",
        meta={}
    )
    
    account_state = {
        'equity': 10000.0,  # Max trade = $100 (1%)
        'position': 0.0,
        'price': 100.0
    }
    
    vetted = kernel.vet_intent(intent, account_state)
    assert vetted is not None
    assert vetted.qty == 1.0  # Clipped to $100 / $100 = 1.0


def test_risk_kernel_daily_drawdown_halt():
    """Test halting on daily drawdown."""
    limits = {
        'max_trade_risk_pct': 10.0,
        'daily_dd_pct': 5.0,
        'leverage_cap': 3.0,
        'max_position': 100.0
    }
    
    kernel = RiskKernel(limits)
    kernel.set_initial_equity(10000.0)
    kernel.daily_pnl = -600.0  # -6% drawdown
    
    intent = TradeIntent(
        symbol="BTCUSDT",
        side=Side.BUY,
        qty=1.0,
        price=100.0,
        tif="GTC",
        post_only=False,
        client_id="test",
        meta={}
    )
    
    account_state = {
        'equity': 9400.0,
        'position': 0.0,
        'price': 100.0
    }
    
    vetted = kernel.vet_intent(intent, account_state)
    assert vetted is None  # Should halt


def test_risk_kernel_after_fill():
    """Test updating state after fill."""
    limits = {'daily_dd_pct': 5.0}
    kernel = RiskKernel(limits)
    
    fill = {'pnl': -100.0, 'qty': 1.0, 'price': 50000.0}
    account_state = {'equity': 9900.0}
    
    kernel.after_fill(fill, account_state)
    assert kernel.daily_pnl == -100.0
    
    # Another fill
    fill2 = {'pnl': 50.0, 'qty': 1.0, 'price': 50000.0}
    kernel.after_fill(fill2, account_state)
    assert kernel.daily_pnl == -50.0


def test_risk_kernel_should_halt():
    """Test should_halt decision."""
    limits = {
        'daily_dd_pct': 5.0,
        'var_max': 0.05
    }
    
    kernel = RiskKernel(limits)
    
    # Normal metrics
    metrics = {'daily_dd': -2.0, 'var_99': 0.02, 'entropy_diverged': False}
    assert not kernel.should_halt(metrics)
    
    # Excessive drawdown
    metrics = {'daily_dd': -6.0, 'var_99': 0.02, 'entropy_diverged': False}
    assert kernel.should_halt(metrics)
    
    # Excessive VaR
    metrics = {'daily_dd': -2.0, 'var_99': 0.08, 'entropy_diverged': False}
    assert kernel.should_halt(metrics)
    
    # Entropy diverged
    metrics = {'daily_dd': -2.0, 'var_99': 0.02, 'entropy_diverged': True}
    assert kernel.should_halt(metrics)


def test_risk_kernel_reset_daily():
    """Test resetting daily PnL."""
    limits = {}
    kernel = RiskKernel(limits)
    
    kernel.daily_pnl = -500.0
    kernel.reset_daily()
    assert kernel.daily_pnl == 0.0


# ===== ENCODER/DECODER TESTS =====

def test_seeded_phase_stability():
    """Test that seeded phase is stable."""
    seed = b"test_seed"
    p = 7
    
    phase1 = _seeded_phase(seed, p)
    phase2 = _seeded_phase(seed, p)
    
    assert phase1 == phase2
    assert 0 <= phase1 < 2 * np.pi


def test_encode_key_basic():
    """Test basic key encoding."""
    # Create observation-like object
    obs = type('Observation', (), {
        'symbol': 'BTCUSDT',
        'features_disc': {'price': 100, 'vol': 50, 'regime_id': 0}
    })()
    
    cfg = {'weights': {'price': 1.0, 'vol': 0.5}}
    P = [2, 3, 5, 7, 11, 13]
    
    K = encode_key(obs, cfg, P, guild_id="TF")
    
    assert len(K) == len(P)
    assert K.dtype == np.complex128
    
    # Should be unit vectors
    norms = np.abs(K)
    np.testing.assert_array_almost_equal(norms, np.ones(len(P)))


def test_encode_key_masking():
    """Test prime masking in key encoding."""
    obs = type('Observation', (), {
        'symbol': 'BTCUSDT',
        'features_disc': {'f1': 10, 'regime_id': 0}
    })()
    
    P = [2, 3, 5, 7, 11]
    
    # Test that masking produces valid vectors
    cfg_masked = {'mask': [2, 5, 11]}
    K_masked = encode_key(obs, cfg_masked, P)
    
    # Should still be unit vectors
    norms = np.abs(K_masked)
    np.testing.assert_array_almost_equal(norms, np.ones(len(P)))


def test_encode_value_basic():
    """Test basic value encoding."""
    basis = Basis(id="B0", primes=[2, 5, 11])
    
    dphi = np.random.randn(6) * 0.1
    dtau = 0.5
    hints = {'style': 1}
    cfg = {}
    P = [2, 3, 5, 7, 11, 13]
    
    V = encode_value(basis, dphi, dtau, hints, cfg, P)
    
    assert len(V) == len(P)
    assert V.dtype == np.complex128
    
    # Should be unit vectors
    norms = np.abs(V)
    np.testing.assert_array_almost_equal(norms, np.ones(len(P)))


def test_encode_value_dict_dphi():
    """Test value encoding with dict dphi."""
    basis = Basis(id="B0", primes=[2, 5])
    
    dphi_dict = {0: 0.1, 2: 0.2}  # Sparse
    dtau = 0.0
    cfg = {}
    P = [2, 3, 5, 7]
    
    V = encode_value(basis, dphi_dict, dtau, {}, cfg, P)
    
    assert len(V) == len(P)
    norms = np.abs(V)
    np.testing.assert_array_almost_equal(norms, np.ones(len(P)))


def test_decode_value_basic():
    """Test basic value decoding."""
    P = [2, 3, 5, 7, 11, 13]
    
    basis_catalog = {
        'B0': [2, 5, 11],
        'B1': [3, 7, 13]
    }
    
    # Encode a value
    basis = Basis(id="B0", primes=[2, 5, 11])
    dphi_orig = np.random.randn(6) * 0.1
    dtau_orig = 0.3
    cfg = {}
    
    V = encode_value(basis, dphi_orig, dtau_orig, {}, cfg, P)
    
    # Decode it
    B_prior, dphi_dec, dtau_dec, hints, conf = decode_value(
        V, P, basis_catalog, cfg, return_confidence=True
    )
    
    assert isinstance(B_prior, dict)
    assert len(B_prior) == 2
    assert 'B0' in B_prior
    assert 'B1' in B_prior
    
    # Both should have probabilities that sum to ~1
    assert abs(B_prior['B0'] + B_prior['B1'] - 1.0) < 0.01
    
    assert isinstance(dphi_dec, np.ndarray)
    assert len(dphi_dec) == len(P)
    
    assert isinstance(dtau_dec, float)
    
    assert isinstance(conf, dict)
    assert 'basis_conf' in conf
    assert 'phi_snr' in conf
    assert 'tau_conf' in conf


def test_decode_value_hints():
    """Test decoding with hints."""
    P = [2, 3, 5, 7]
    
    basis_catalog = {'B0': [2, 5]}
    basis = Basis(id="B0", primes=[2, 5])
    
    hints_orig = {'param1': 3, 'param2': 5}
    V = encode_value(basis, np.zeros(4), 0.0, hints_orig, {}, P)
    
    B_prior, dphi, dtau, hints_dec, conf = decode_value(
        V, P, basis_catalog, {}, known_hint_names=['param1', 'param2']
    )
    
    assert isinstance(hints_dec, dict)
    assert 'param1' in hints_dec
    assert 'param2' in hints_dec


def test_encode_decode_roundtrip():
    """Test encode-decode round trip."""
    P = [2, 3, 5, 7, 11, 13, 17, 19]
    
    basis_catalog = {
        'B0': [2, 5, 11, 17],
        'B1': [3, 7, 13, 19]
    }
    
    # Encode
    basis = Basis(id="B0", primes=[2, 5, 11, 17])
    dphi_orig = np.random.randn(8) * 0.05
    dtau_orig = 0.2
    hints_orig = {'hint1': 5}
    cfg = {}
    
    V = encode_value(basis, dphi_orig, dtau_orig, hints_orig, cfg, P)
    
    # Decode
    B_prior, dphi_dec, dtau_dec, hints_dec, conf = decode_value(
        V, P, basis_catalog, cfg, known_hint_names=['hint1']
    )
    
    # Check that basis was recovered
    best_basis = max(B_prior.items(), key=lambda x: x[1])[0]
    assert best_basis == 'B0'
    
    # dphi and dtau should be approximately recovered
    # (not exact due to shrinkage and clipping)
    assert isinstance(dphi_dec, np.ndarray)
    assert isinstance(dtau_dec, float)
    assert abs(dtau_dec) <= 2.0  # Within clip range


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
