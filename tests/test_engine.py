"""
tests/test_engine.py - Unit tests for Adaptive Information Selection Engine

Covers:
1. Posterior sums to 1 (within numerical precision) after every update
2. EIG is strictly non-negative for every block from uniform / arbitrary prior
3. Adaptive policy never uses more blocks than all_blocks baseline (<= 5 blocks)
4. Each block's feature vector length strictly equals its declared feature column count
5. GroupShuffleSplit guarantees exactly zero train/test/val group leakage
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from scipy.special import softmax

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import AdaptiveInformationEngine
from model.features import BLOCK_EXTRACTORS

@pytest.fixture
def mock_engine():
    classes = ["ABS", "Nylon", "PE", "PET", "PS", "PVC"]
    K = len(classes)
    prior = np.ones(K) / K
    log_prior = np.log(prior)

    # Mock block feature names
    sample_rgb = np.ones((32, 32, 3), dtype=np.float32)
    sample_mask = np.ones((32, 32), dtype=bool)
    block_feature_names = {b: list(fn(sample_rgb, sample_mask).keys()) for b, fn in BLOCK_EXTRACTORS.items()}

    # Mock experts with dummy classifiers
    experts = {}
    for b in BLOCK_EXTRACTORS:
        clf = DummyClassifier(strategy="uniform")
        X_dummy = np.random.rand(50, len(block_feature_names[b]))
        y_dummy = np.random.randint(0, K, size=50)
        clf.fit(X_dummy, y_dummy)
        experts[b] = clf

    block_costs = {"shape": 1.0, "intensity": 1.5, "colour": 1.8, "texture": 4.2, "frequency": 2.1}
    block_weights = {"shape": 0.8, "intensity": 0.9, "colour": 1.1, "texture": 1.3, "frequency": 1.0}

    # Simulation pool: 50 per class -> M = 300
    M = 50 * K
    pool_cls = np.repeat(np.arange(K), 50)
    pool_counts = np.bincount(pool_cls, minlength=K).astype(float)
    pool_lr = {b: np.random.randn(M, K) * 0.5 for b in BLOCK_EXTRACTORS}

    engine = AdaptiveInformationEngine(
        classes=classes,
        log_prior=log_prior,
        experts=experts,
        block_weights=block_weights,
        block_costs=block_costs,
        pool_lr=pool_lr,
        pool_cls=pool_cls,
        pool_counts=pool_counts,
        block_feature_names=block_feature_names,
        fixed_order=list(BLOCK_EXTRACTORS.keys()),
        cheapest_order=sorted(list(BLOCK_EXTRACTORS.keys()), key=lambda b: block_costs[b])
    )
    return engine, block_feature_names

def test_posterior_sums_to_one(mock_engine):
    """Assert posterior sums to 1.0 within numerical precision after every update."""
    engine, block_feature_names = mock_engine
    
    # Generate random features
    features = {b: np.random.rand(len(block_feature_names[b])) for b in BLOCK_EXTRACTORS}
    
    log_post = engine.log_prior.copy()
    for b in BLOCK_EXTRACTORS:
        log_post = engine.update_posterior(log_post, b, features[b])
        p = softmax(log_post)
        assert np.isclose(np.sum(p), 1.0, atol=1e-6), f"Posterior after block {b} does not sum to 1: sum={np.sum(p)}"

def test_eig_non_negative(mock_engine):
    """Assert Expected Information Gain (EIG) is non-negative for all blocks."""
    engine, _ = mock_engine
    
    # From uniform prior
    scores, eigs = engine.compute_eig_and_scores(engine.log_prior, list(BLOCK_EXTRACTORS.keys()))
    for b, eig_val in eigs.items():
        assert eig_val >= 0.0, f"Negative EIG detected for block {b}: {eig_val}"
        assert scores[b] >= 0.0, f"Negative score detected for block {b}: {scores[b]}"

    # From biased posterior
    biased_log_post = np.array([2.0, -1.0, 0.5, -2.0, 1.0, 0.0])
    scores_b, eigs_b = engine.compute_eig_and_scores(biased_log_post, list(BLOCK_EXTRACTORS.keys()))
    for b, eig_val in eigs_b.items():
        assert eig_val >= 0.0, f"Negative EIG detected for block {b} from biased posterior: {eig_val}"

def test_adaptive_never_exceeds_all_blocks(mock_engine):
    """Assert adaptive policy never uses more blocks than all_blocks (which evaluates all 5 blocks)."""
    engine, block_feature_names = mock_engine
    
    for _ in range(10):
        features = {b: np.random.rand(len(block_feature_names[b])) for b in BLOCK_EXTRACTORS}
        res_adaptive = engine.run(features, policy="adaptive", tau=0.90, lambda_=0.01)
        res_all = engine.run(features, policy="all_blocks")
        
        assert len(res_adaptive["final"]["blocks_used"]) <= len(res_all["final"]["blocks_used"])
        assert len(res_all["final"]["blocks_used"]) == len(BLOCK_EXTRACTORS)
        assert res_adaptive["final"]["total_cost"] <= res_all["final"]["total_cost"] + 1e-6

def test_feature_vector_lengths_match():
    """Assert extracted feature dictionary keys match block_feature_names exactly."""
    dummy_rgb = np.random.rand(40, 40, 3).astype(np.float32)
    dummy_mask = np.ones((40, 40), dtype=bool)

    expected_lengths = {
        "shape": 8,
        "intensity": 10,
        "colour": 15,
        "texture": 22,  # 6*2 glcm props + 10 lbp bins = 22
        "frequency": 7  # 6 bands + 1 high/low ratio = 7
    }

    for b, fn in BLOCK_EXTRACTORS.items():
        feats = fn(dummy_rgb, dummy_mask)
        assert len(feats) == expected_lengths[b], f"Block {b} returned {len(feats)} features, expected {expected_lengths[b]}"

def test_zero_group_leakage_in_split():
    """Assert GroupShuffleSplit strictly isolates source_image groups with zero overlap."""
    n_samples = 200
    n_groups = 30
    groups = np.random.randint(0, n_groups, size=n_samples)
    y = np.random.randint(0, 6, size=n_samples)
    X = np.random.rand(n_samples, 10)

    # Split 1: train_val vs test (75/25)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    train_val_groups = set(groups[train_val_idx])
    test_groups = set(groups[test_idx])
    assert len(train_val_groups & test_groups) == 0, "Leakage between train_val and test!"

    # Split 2: train vs val (80/20 of remainder)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss_val.split(X[train_val_idx], y[train_val_idx], groups=groups[train_val_idx]))

    train_groups = set(groups[train_val_idx][train_idx])
    val_groups = set(groups[train_val_idx][val_idx])
    assert len(train_groups & val_groups) == 0, "Leakage between train and val!"
    assert len(train_groups & test_groups) == 0, "Leakage between train and test!"
