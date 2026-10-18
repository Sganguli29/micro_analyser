"""
tests/test_math_utils.py - Unit tests for Bayesian fusion, uncertainty metrics, and Monte Carlo EIG.
"""

import numpy as np
import pytest
from scipy.special import softmax

from src.math_utils import (
    compute_entropy,
    compute_max_posterior,
    compute_margin,
    fuse_log_opinion_pool,
    fit_log_opinion_pool_weights,
    compute_monte_carlo_eig
)

def test_entropy_properties():
    # Uniform over K=6 classes: H = ln(6)
    K = 6
    p_uniform = np.ones(K) / K
    h_uni = compute_entropy(p_uniform)
    assert np.isclose(h_uni, np.log(K), atol=1e-5), f"Expected ln(6)={np.log(6)}, got {h_uni}"

    # Deterministic: H = 0
    p_det = np.zeros(K)
    p_det[0] = 1.0
    h_det = compute_entropy(p_det)
    assert np.isclose(h_det, 0.0, atol=1e-4), f"Expected 0.0, got {h_det}"

def test_max_posterior_and_margin():
    p = np.array([0.1, 0.6, 0.25, 0.05])
    assert np.isclose(compute_max_posterior(p), 0.6)
    # top 1 is 0.6, top 2 is 0.25 -> margin = 0.35
    assert np.isclose(compute_margin(p), 0.35)

def test_log_opinion_pool_fusion():
    K = 6
    prior = np.ones(K) / K
    log_prior = np.log(prior)

    # 1. No blocks evaluated: must return prior
    fused_empty = fuse_log_opinion_pool(log_prior, {}, {})
    assert np.allclose(fused_empty, prior, atol=1e-6)

    # 2. Unevaluated block produces zero contribution
    expert_probs = {
        "block1": np.array([0.8, 0.04, 0.04, 0.04, 0.04, 0.04]),
        "block2": None
    }
    weights = {"block1": 1.0, "block2": 1.0}
    fused = fuse_log_opinion_pool(log_prior, expert_probs, weights)
    assert np.isclose(np.sum(fused), 1.0, atol=1e-6)
    assert np.argmax(fused) == 0

def test_monte_carlo_eig_non_negative():
    K = 6
    prior = np.ones(K) / K
    log_prior = np.log(prior)

    # Mock simulation pool: 50 per class -> M=300
    M = 50 * K
    pool_cls = np.repeat(np.arange(K), 50)
    pool_counts = np.bincount(pool_cls, minlength=K).astype(float)
    
    np.random.seed(42)
    pool_lr = {
        "b1": np.random.randn(M, K) * 0.5,
        "b2": np.random.randn(M, K) * 0.5
    }
    block_costs = {"b1": 2.5, "b2": 5.0}

    scores, eigs = compute_monte_carlo_eig(
        log_post=log_prior,
        candidate_blocks=["b1", "b2"],
        block_costs=block_costs,
        pool_lr=pool_lr,
        pool_cls=pool_cls,
        pool_counts=pool_counts
    )

    for b in ["b1", "b2"]:
        assert eigs[b] >= 0.0, f"Negative EIG in block {b}: {eigs[b]}"
        assert scores[b] >= 0.0, f"Negative Score in block {b}: {scores[b]}"
        assert np.isclose(scores[b], eigs[b] / block_costs[b])
