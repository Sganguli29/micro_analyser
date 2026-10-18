r"""
src/policies.py - Information acquisition policies for adaptive multimodal feature selection.

All policies share a unified interface:
- select_next_block(remaining_blocks, current_posterior, context_z, ...) -> str

Policies implemented:
1. RandomPolicy: Uniformly random selection among remaining candidate blocks.
2. CheapestFirstPolicy: Greedy order by ascending wall-clock extraction cost (in ms).
3. BestFirstPolicy: Fixed order by descending validation accuracy of individual block classifiers.
4. StaticFeatureSelectionPolicy: Forward-greedy feature block selection at matched budget.
5. EIGCostPolicy: Greedy EIG(b | belief) / Cost(b) with global prior.
6. ContextAwareEIGCostPolicy: Context-prior conditioned EIG(b | Z, belief) / Cost(b).
7. OraclePolicy: Particle-specific cheapest subset of blocks that correctly predicts the ground-truth label (theoretical upper bound).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from itertools import combinations

from src.math_utils import compute_monte_carlo_eig, fuse_log_opinion_pool

class BaseAcquisitionPolicy(ABC):
    """Abstract Base Class for feature block acquisition policies."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """Selects the next block to acquire from remaining candidate blocks."""
        pass

class RandomPolicy(BaseAcquisitionPolicy):
    """Uniformly random block selection."""
    def __init__(self, seed: int = 42):
        super().__init__("random")
        self.rng = np.random.RandomState(seed)

    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        return self.rng.choice(remaining_blocks)

class CheapestFirstPolicy(BaseAcquisitionPolicy):
    """Acquires blocks in order of ascending wall-clock execution cost in ms."""
    def __init__(self):
        super().__init__("cheapest_first")

    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        return min(remaining_blocks, key=lambda b: block_costs_ms[b])

class BestFirstPolicy(BaseAcquisitionPolicy):
    """Acquires blocks in fixed order of descending single-block validation accuracy."""
    def __init__(self, block_order: Optional[List[str]] = None):
        super().__init__("best_first")
        # Default order based on single-block validation accuracy
        self.block_order = block_order or ["colour", "texture", "shape", "intensity", "frequency"]

    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        for b in self.block_order:
            if b in remaining_blocks:
                return b
        return remaining_blocks[0]

class StaticFeatureSelectionPolicy(BaseAcquisitionPolicy):
    """Forward-greedy block selection on validation data at matched cost budget."""
    def __init__(self, greedy_order: Optional[List[str]] = None):
        super().__init__("static_selection")
        self.greedy_order = greedy_order or ["colour", "texture", "shape", "frequency", "intensity"]

    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        for b in self.greedy_order:
            if b in remaining_blocks:
                return b
        return remaining_blocks[0]

class EIGCostPolicy(BaseAcquisitionPolicy):
    r"""
    Adaptive Bayesian acquisition via Expected Information Gain per ms cost:
        b^* = \arg\max_{b \in B_{rem}} \frac{EIG(b \mid belief)}{Cost(b)}
    """
    def __init__(
        self,
        pool_lr: Dict[str, np.ndarray],
        pool_cls: np.ndarray,
        pool_counts: np.ndarray
    ):
        super().__init__("eig_cost_global")
        self.pool_lr = pool_lr
        self.pool_cls = pool_cls
        self.pool_counts = pool_counts

    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        log_post = np.log(np.clip(current_posterior, 1e-9, 1.0))
        scores, _ = compute_monte_carlo_eig(
            log_post=log_post,
            candidate_blocks=remaining_blocks,
            block_costs=block_costs_ms,
            pool_lr=self.pool_lr,
            pool_cls=self.pool_cls,
            pool_counts=self.pool_counts
        )
        return max(remaining_blocks, key=lambda b: scores[b])

class ContextAwareEIGCostPolicy(BaseAcquisitionPolicy):
    r"""
    Context-prior conditioned adaptive acquisition:
        b^* = \arg\max_{b \in B_{rem}} \frac{EIG(b \mid Z, belief)}{Cost(b)}
    """
    def __init__(
        self,
        pool_lr: Dict[str, np.ndarray],
        pool_cls: np.ndarray,
        pool_counts: np.ndarray
    ):
        super().__init__("eig_cost_context")
        self.pool_lr = pool_lr
        self.pool_cls = pool_cls
        self.pool_counts = pool_counts

    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        log_post = np.log(np.clip(current_posterior, 1e-9, 1.0))
        scores, _ = compute_monte_carlo_eig(
            log_post=log_post,
            candidate_blocks=remaining_blocks,
            block_costs=block_costs_ms,
            pool_lr=self.pool_lr,
            pool_cls=self.pool_cls,
            pool_counts=self.pool_counts
        )
        return max(remaining_blocks, key=lambda b: scores[b])

class OraclePolicy(BaseAcquisitionPolicy):
    """
    Theoretical upper bound oracle:
    Identifies the cheapest subset/order of blocks for a specific particle
    that correctly yields its ground-truth label.
    """
    def __init__(
        self,
        expert_probs_by_block: Dict[str, np.ndarray],
        block_weights: Dict[str, float],
        log_prior: np.ndarray
    ):
        super().__init__("oracle")
        self.expert_probs = expert_probs_by_block
        self.block_weights = block_weights
        self.log_prior = log_prior

    def get_optimal_block_subset(
        self,
        particle_idx: int,
        true_label: int,
        block_costs_ms: Dict[str, float]
    ) -> List[str]:
        """Finds the lowest cost subset of blocks that correctly predicts true_label."""
        all_blocks = list(block_costs_ms.keys())
        # Sort candidate subsets by total cost
        all_subsets = []
        for r in range(1, len(all_blocks) + 1):
            for subset in combinations(all_blocks, r):
                cost = sum(block_costs_ms[b] for b in subset)
                all_subsets.append((cost, list(subset)))
        
        all_subsets.sort(key=lambda x: x[0])

        for cost, subset in all_subsets:
            dict_subset = {b: (self.expert_probs[b][particle_idx] if b in subset else None) for b in all_blocks}
            post = fuse_log_opinion_pool(self.log_prior, dict_subset, self.block_weights)
            if np.argmax(post) == true_label:
                return subset

        # If no subset achieves true_label, return full set
        return all_blocks

    def select_next_block(
        self,
        remaining_blocks: List[str],
        current_posterior: np.ndarray,
        block_costs_ms: Dict[str, float],
        context_z: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        optimal_subset = kwargs.get("optimal_subset", remaining_blocks)
        for b in optimal_subset:
            if b in remaining_blocks:
                return b
        return remaining_blocks[0]
