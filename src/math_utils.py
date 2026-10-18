r"""
src/math_utils.py - Explicit mathematical implementations of Bayesian opinion pools, uncertainty metrics, and Monte Carlo EIG.

Mathematical Foundations:
1. Log-Opinion Pool Fusion:
   Given a set of evaluated feature blocks B ⊆ {1, ..., B_total}, prior P_0(y),
   calibrated expert predictions P_b(y | x_b), and weights w_b > 0:
       log P(y | B) = log P_0(y) + \sum_{b \in B} w_b [ log P_b(y | x_b) - log P_0(y) ] - log Z
   where log Z = logsumexp( ... ) ensures \sum_y P(y | B) = 1.
   Unevaluated blocks contribute identically zero to the summation.
   Weights w_b are fit by minimizing validation Negative Log-Likelihood (NLL).

2. Posterior Uncertainty Metrics:
   - Shannon Entropy: H(p) = -\sum_{k=1}^K p_k ln(p_k + 1e-9)
   - Max Posterior: Confidence(p) = \max_k p_k
   - Margin: Margin(p) = p_{(1)} - p_{(2)}, the difference between top-1 and top-2 probabilities.

3. Expected Information Gain (EIG):
   For candidate block b given current belief P = softmax(log_post):
       EIG(b | P) = H(current) - \mathbb{E}_{x_b}[ H( P_{new}(x_b) ) ]
   Approximated via Monte Carlo expectation over a balanced training simulation pool M:
       w_i = P(c_i) / count(c_i), normalized such that \sum_{i=1}^M w_i = 1.
       newP_i = softmax( log_post + w_b * [ log P_b(y | x_{b, i}) - log P_0(y) ] )
       EIG(b | P) = \max( 0, H_0 - \sum_{i=1}^M w_i H(newP_i) )
   Acquisition Score:
       Score(b) = EIG(b | P) / Cost(b)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.metrics import log_loss

def compute_entropy(p: np.ndarray, eps: float = 1e-9) -> float:
    r"""
    Computes Shannon entropy H(p) in nats for a 1D probability distribution.
    
    Formula:
        H(p) = -\sum_{k=1}^K p_k \ln(p_k + \epsilon)
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))

def compute_max_posterior(p: np.ndarray) -> float:
    r"""
    Computes top-1 maximum posterior probability confidence:
        Conf(p) = \max_k p_k
    """
    return float(np.max(p))

def compute_margin(p: np.ndarray) -> float:
    """
    Computes probability margin between the highest and second highest classes:
        Margin(p) = p_{(1)} - p_{(2)}
    """
    p = np.asarray(p, dtype=np.float64)
    if len(p) < 2:
        return 1.0
    sorted_p = np.sort(p)[::-1]
    return float(sorted_p[0] - sorted_p[1])

def fuse_log_opinion_pool(
    log_prior: np.ndarray,
    expert_probs_dict: Dict[str, np.ndarray],
    weights_dict: Dict[str, float]
) -> np.ndarray:
    r"""
    Computes the normalized posterior under a weighted log-opinion pool.
    
    Formula:
        log P(y | B) = log P_0(y) + \sum_{b \in B} w_b [ log P_b(y | x_b) - log P_0(y) ]
        P(y | B) = softmax( log P(y | B) )
        
    Args:
        log_prior: 1D array of shape (K,) or 2D of shape (N, K).
        expert_probs_dict: Dict mapping evaluated block name to (N, K) or (K,) probability array.
        weights_dict: Dict mapping block name to float weight w_b.
        
    Returns:
        fused_probs: Array of same leading dimensions as log_prior, normalized along axis -1.
    """
    # Determine if output should be 1D (single sample) or 2D (batch)
    is_batch = (log_prior.ndim > 1) or any(
        p is not None and np.ndim(p) > 1 for p in expert_probs_dict.values()
    )
    
    if log_prior.ndim == 1:
        log_p = log_prior.copy()[None, :]
        log_prior_bcast = log_prior[None, :]
    else:
        log_p = log_prior.copy()
        log_prior_bcast = log_prior

    for b, p_b in expert_probs_dict.items():
        if p_b is None:
            continue
        w = weights_dict.get(b, 1.0)
        p_clipped = np.clip(p_b, 1e-6, 1.0)
        if p_clipped.ndim == 1:
            p_clipped = p_clipped[None, :]
        log_p = log_p + w * (np.log(p_clipped) - log_prior_bcast)

    log_p = log_p - logsumexp(log_p, axis=-1, keepdims=True)
    fused = softmax(log_p, axis=-1)
    return fused if is_batch else fused[0]

def fit_log_opinion_pool_weights(
    val_probs_by_block: Dict[str, np.ndarray],
    y_val: np.ndarray,
    log_prior: np.ndarray,
    block_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Fits positive weights w_b by minimizing multiclass cross-entropy (NLL) on validation split.
    
    Uses Nelder-Mead optimization on theta = log(w) to enforce strict positivity (w_b > 0).
    """
    if block_names is None:
        block_names = list(val_probs_by_block.keys())

    K = len(log_prior)
    val_lrs = {}
    for b in block_names:
        p_clipped = np.clip(val_probs_by_block[b], 1e-6, 1.0)
        p_clipped /= p_clipped.sum(axis=1, keepdims=True)
        val_lrs[b] = np.log(p_clipped) - log_prior[None, :]

    def objective(theta):
        weights = np.exp(theta)
        log_fused = log_prior.copy()[None, :]
        for i, b in enumerate(block_names):
            log_fused = log_fused + weights[i] * val_lrs[b]
        p_fused = softmax(log_fused, axis=1)
        return log_loss(y_val, p_fused, labels=list(range(K)))

    init_theta = np.zeros(len(block_names))
    res = minimize(objective, init_theta, method="Nelder-Mead")
    opt_weights = np.exp(res.x)

    return {b: round(float(opt_weights[i]), 4) for i, b in enumerate(block_names)}

def compute_monte_carlo_eig(
    log_post: np.ndarray,
    candidate_blocks: List[str],
    block_costs: Dict[str, float],
    pool_lr: Dict[str, np.ndarray],
    pool_cls: np.ndarray,
    pool_counts: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, float]]:
    r"""
    Computes Expected Information Gain (EIG) and acquisition score (EIG / Cost)
    via Monte Carlo expectation over a balanced training simulation pool.
    
    Formula:
        H_0 = H( P )
        w_i = P(c_i) / count(c_i), normalized to sum to 1.
        newP_i = softmax( log_post + pool_lr_b[i] )
        H_1 = \sum_{i=1}^M w_i H(newP_i)
        EIG = \max(0, H_0 - H_1)
        Score = EIG / Cost
        
    Args:
        log_post: 1D array of current log posterior over K classes.
        candidate_blocks: List of remaining unevaluated block names.
        block_costs: Dict mapping block name to execution cost in milliseconds.
        pool_lr: Dict mapping block name to precomputed (M, K) log-ratio matrix.
        pool_cls: Array of length M with class index for each simulation pool particle.
        pool_counts: Array of shape (K,) with sample count per class in simulation pool.
        
    Returns:
        (scores, eigs): Dicts mapping block name to score and EIG in nats.
    """
    P = softmax(log_post)
    H0 = compute_entropy(P)

    denom = np.maximum(pool_counts[pool_cls], 1.0)
    w = P[pool_cls] / denom
    w_sum = np.sum(w)
    if w_sum > 0:
        w /= w_sum
    else:
        w = np.ones_like(w) / len(w)

    scores = {}
    eigs = {}

    for b in candidate_blocks:
        lr = pool_lr[b]  # Shape (M, K)
        new_log_post = log_post[None, :] + lr
        newP = softmax(new_log_post, axis=1)  # Shape (M, K)
        H1_per_sample = -np.sum(newP * np.log(newP + 1e-9), axis=1)  # Shape (M,)
        exp_H1 = float(np.dot(w, H1_per_sample))
        eig = max(0.0, float(H0 - exp_H1))
        eigs[b] = eig
        cost = max(block_costs.get(b, 1.0), 1e-6)
        scores[b] = float(eig / cost)

    return scores, eigs
