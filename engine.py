"""
engine.py - Phase 4: Adaptive Information Selection Engine

Implements the adaptive Bayesian information acquisition policy and baseline policies:
- adaptive: Greedy EIG / cost based on evolving posterior re-weighted simulation pool
- fixed_best_first: Fixed order by descending validation accuracy
- fixed_cheapest: Fixed order by ascending cost
- random_order: Random sequence of blocks
- all_blocks: Evaluates all 5 blocks without early stopping (accuracy ceiling)

Self-contained with no web dependencies.
"""

import time
import numpy as np
from scipy.special import softmax, logsumexp
from typing import Dict, List, Any, Optional, Tuple

class AdaptiveInformationEngine:
    def __init__(
        self,
        classes: List[str],
        log_prior: np.ndarray,
        experts: Dict[str, Any],
        block_weights: Dict[str, float],
        block_costs: Dict[str, float],
        pool_lr: Dict[str, np.ndarray],
        pool_cls: np.ndarray,
        pool_counts: np.ndarray,
        block_feature_names: Dict[str, List[str]],
        fixed_order: Optional[List[str]] = None,
        cheapest_order: Optional[List[str]] = None,
    ):
        """
        Initializes the engine.
        
        Args:
            classes: List of class names (e.g. ['ABS', 'Nylon', 'PE', 'PET', 'PS', 'PVC'])
            log_prior: Log prior distribution array of shape (K,)
            experts: Dict of calibrated expert models per block
            block_weights: Dict of log-opinion pool weights w_b
            block_costs: Dict of normalized block execution costs
            pool_lr: Dict of (M, K) log-ratio matrices computed on training simulation pool
            pool_cls: Array of length M with class index for each simulation pool particle
            pool_counts: Array of shape (K,) with sample count per class in simulation pool
            block_feature_names: Dict mapping block name to list of feature column names
            fixed_order: Sequence of blocks ordered by validation accuracy
            cheapest_order: Sequence of blocks ordered by ascending cost
        """
        self.classes = list(classes)
        self.num_classes = len(classes)
        self.log_prior = np.array(log_prior, dtype=np.float64)
        self.prior_p = softmax(self.log_prior)
        self.experts = experts
        self.block_weights = block_weights
        self.block_costs = block_costs
        self.pool_lr = pool_lr
        self.pool_cls = np.array(pool_cls, dtype=np.int64)
        self.pool_counts = np.array(pool_counts, dtype=np.float64)
        self.block_feature_names = block_feature_names
        
        self.all_blocks = list(block_costs.keys())
        self.all_blocks_cost = float(sum(block_costs.values()))
        
        self.fixed_order = fixed_order if fixed_order else self.all_blocks
        self.cheapest_order = cheapest_order if cheapest_order else sorted(self.all_blocks, key=lambda b: self.block_costs[b])

    def compute_eig_and_scores(self, log_post: np.ndarray, remaining_blocks: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Computes EIG and EIG/cost score for all candidate remaining blocks.
        
        Formula:
        P = softmax(log_post)
        H0 = -sum(P * log(P + 1e-9))
        w = P[pool_cls] / max(pool_counts[pool_cls], 1.0); w /= sum(w)
        newP = softmax(log_post[None, :] + pool_lr[b], axis=1) # (M, K)
        H1 = -sum(newP * log(newP + 1e-9), axis=1)
        EIG = H0 - dot(w, H1)
        score = EIG / block_cost[b]
        """
        P = softmax(log_post)
        H0 = float(-np.sum(P * np.log(P + 1e-9)))

        # Weighting by current belief: what the engine expects to see next is re-weighted
        # by everything it already knows about this particle
        denom = np.maximum(self.pool_counts[self.pool_cls], 1.0)
        w = P[self.pool_cls] / denom
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum
        else:
            w = np.ones_like(w) / len(w)

        scores = {}
        eigs = {}

        for b in remaining_blocks:
            lr = self.pool_lr[b]  # Shape (M, K)
            # new log post is log_post + lr
            new_log_post = log_post[None, :] + lr
            newP = softmax(new_log_post, axis=1)  # Shape (M, K)
            H1 = -np.sum(newP * np.log(newP + 1e-9), axis=1)  # Shape (M,)
            EIG = max(0.0, float(H0 - np.dot(w, H1)))
            eigs[b] = EIG
            cost = max(self.block_costs.get(b, 1.0), 1e-6)
            scores[b] = float(EIG / cost)

        return scores, eigs

    def update_posterior(self, log_post: np.ndarray, block_name: str, x_b: np.ndarray) -> np.ndarray:
        """
        Updates log-posterior when block b is evaluated.
        
        Formula:
        p = clip(experts[b].predict_proba(x_b)[0], 1e-6, 1.0); p /= p.sum()
        log_post += block_weight[b] * (log(p) - log_prior)
        log_post -= logsumexp(log_post)
        """
        # Ensure x_b is 2D with exact column count
        if x_b.ndim == 1:
            x_b = x_b.reshape(1, -1)
        expected_len = len(self.block_feature_names[block_name])
        assert x_b.shape[1] == expected_len, f"Feature length mismatch for block {block_name}: got {x_b.shape[1]}, expected {expected_len}"

        expert = self.experts[block_name]
        p_raw = expert.predict_proba(x_b)[0]
        p = np.clip(p_raw, 1e-6, 1.0)
        p = p / np.sum(p)

        w_b = self.block_weights.get(block_name, 1.0)
        updated_log_post = log_post + w_b * (np.log(p) - self.log_prior)
        updated_log_post -= logsumexp(updated_log_post)
        return updated_log_post

    def run(
        self,
        features_dict: Dict[str, np.ndarray],
        policy: str = "adaptive",
        tau: float = 0.90,
        lambda_: float = 0.01,
        true_label: Optional[str] = None,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the decision policy on a particle given pre-extracted or available features.
        
        Args:
            features_dict: Dict mapping block name to 1D or 2D feature array
            policy: One of 'adaptive', 'fixed_best_first', 'fixed_cheapest', 'random_order', 'all_blocks'
            tau: Confidence threshold (e.g. 0.90)
            lambda_: Minimum value of information / score threshold (e.g. 0.01)
            true_label: Optional true label string for evaluation / display
            random_seed: Seed for random_order policy
            
        Returns:
            Structured dictionary matching the API trace specification.
        """
        t_start = time.perf_counter()
        
        log_post = self.log_prior.copy()
        remaining_blocks = list(self.all_blocks)
        
        # Policy sequence setup
        if policy == "fixed_best_first" or policy == "fixed":
            sequence = [b for b in self.fixed_order if b in remaining_blocks]
        elif policy == "fixed_cheapest":
            sequence = [b for b in self.cheapest_order if b in remaining_blocks]
        elif policy == "random_order":
            rng = np.random.RandomState(random_seed)
            sequence = list(remaining_blocks)
            rng.shuffle(sequence)
        else:
            sequence = list(remaining_blocks)

        steps = []
        blocks_used = []
        cumulative_cost = 0.0
        stopped_because = "blocks_exhausted"
        
        step_idx = 1
        
        while len(remaining_blocks) > 0:
            # 1. Compute scores for all remaining blocks (for transparency and decision)
            scores, eigs = self.compute_eig_and_scores(log_post, remaining_blocks)
            
            # 2. Select chosen block
            if policy == "adaptive":
                # Best score
                chosen_block = max(remaining_blocks, key=lambda b: scores[b])
                best_score = scores[chosen_block]
                
                # Check stopping condition before evaluating if lambda threshold applies
                if len(blocks_used) >= 1 and best_score < lambda_:
                    stopped_because = "low_value_of_information"
                    break
            else:
                # Baseline sequence
                seq_idx = len(blocks_used)
                if seq_idx < len(sequence):
                    chosen_block = sequence[seq_idx]
                else:
                    break

            # 3. Evaluate chosen block
            cost = self.block_costs.get(chosen_block, 1.0)
            cumulative_cost += cost
            blocks_used.append(chosen_block)
            remaining_blocks.remove(chosen_block)
            
            x_b = features_dict[chosen_block]
            log_post = self.update_posterior(log_post, chosen_block, x_b)
            
            # 4. Current posterior stats
            current_P = softmax(log_post)
            current_entropy = float(-np.sum(current_P * np.log(current_P + 1e-9)))
            confidence = float(np.max(current_P))
            
            steps.append({
                "step": step_idx,
                "chosen_block": chosen_block,
                "scores": {b: round(float(scores[b]), 4) for b in scores},
                "posterior": [round(float(p), 4) for p in current_P],
                "entropy": round(current_entropy, 4),
                "confidence": round(confidence, 4),
                "cost_paid": round(float(cost), 2),
                "cumulative_cost": round(float(cumulative_cost), 2)
            })
            
            step_idx += 1
            
            # 5. Check confidence stopping condition
            if policy != "all_blocks" and confidence >= tau:
                stopped_because = "confidence_threshold"
                break

        final_P = softmax(log_post)
        pred_idx = int(np.argmax(final_P))
        predicted_class = self.classes[pred_idx]
        final_confidence = float(final_P[pred_idx])
        
        blocks_skipped = [b for b in self.all_blocks if b not in blocks_used]
        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        
        return {
            "policy": policy,
            "prior": [round(float(p), 4) for p in self.prior_p],
            "steps": steps,
            "final": {
                "predicted_class": predicted_class,
                "confidence": round(final_confidence, 4),
                "blocks_used": blocks_used,
                "blocks_skipped": blocks_skipped,
                "total_cost": round(float(cumulative_cost), 2),
                "all_blocks_cost": round(float(self.all_blocks_cost), 2),
                "stopped_because": stopped_because,
                "true_label": true_label
            },
            "elapsed_ms": elapsed_ms
        }
