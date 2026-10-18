r"""
src/pipeline.py - Unified adaptive multimodal feature acquisition and inference pipeline.

Unified loop:
1. Extract / retrieve context descriptors Z from segmentation mask.
2. Initialize belief posterior P_0(Y) from calibrated context prior P(Y|Z) or global prior.
3. While not STOP and remaining blocks exist:
   a. Check stopping criterion (fixed confidence threshold \tau or learned stopping classifier).
   b. Select next block b^* via acquisition policy (e.g. EIG(b|Z)/Cost(b)).
   c. Acquire block b^* and accumulate wall-clock cost in ms.
   d. Update posterior via weighted log-opinion pool fusion.
4. At STOP or block exhaustion, apply Open-Set OOD rejection rule:
   - If rejected: output UNKNOWN.
   - Else: output predicted polymer class.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import CLASSES, CLASS_TO_IDX
from src.math_utils import compute_entropy, compute_max_posterior, compute_margin, fuse_log_opinion_pool
from src.policies import BaseAcquisitionPolicy, ContextAwareEIGCostPolicy, EIGCostPolicy
from src.context_prior import Z_FEATURE_NAMES
from src.cost_model import load_block_costs_ms
from src.utils import PROJECT_ROOT

class AdaptivePipeline:
    """
    Unified end-to-end inference pipeline combining:
    - Context prior P(Y|Z)
    - Adaptive Bayesian EIG/Cost acquisition
    - Log-opinion pool fusion
    - Learned or fixed threshold stopping
    - Open-Set OOD rejection
    """
    def __init__(
        self,
        acquisition_policy: BaseAcquisitionPolicy,
        block_costs_ms: Dict[str, float],
        block_weights: Dict[str, float],
        global_log_prior: np.ndarray,
        context_prior_model: Optional[Any] = None,
        use_context_prior: bool = True,
        stopping_method: str = "fixed", # "fixed" or "learned"
        fixed_tau: float = 0.85,
        learned_stopping_model: Optional[Any] = None,
        learned_stopping_threshold: float = 0.40,
        use_ood: bool = False,
        ood_tau_conf: float = 0.50,
        ood_tau_ent: float = 1.20
    ):
        self.policy = acquisition_policy
        self.block_costs_ms = block_costs_ms
        self.block_weights = block_weights
        self.global_log_prior = global_log_prior
        self.context_prior_model = context_prior_model
        self.use_context_prior = use_context_prior
        self.stopping_method = stopping_method
        self.fixed_tau = fixed_tau
        self.learned_stopping_model = learned_stopping_model
        self.learned_stopping_threshold = learned_stopping_threshold
        self.use_ood = use_ood
        self.ood_tau_conf = ood_tau_conf
        self.ood_tau_ent = ood_tau_ent
        self.all_blocks = list(block_costs_ms.keys())

    def run_particle(
        self,
        particle_probs_by_block: Dict[str, np.ndarray],
        context_z: np.ndarray,
        particle_idx: int = 0,
        true_label: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes the adaptive acquisition pipeline on a single particle.
        
        Args:
            particle_probs_by_block: Dict mapping block name to 1D probability array of shape (K,).
            context_z: 1D array of 7 context descriptors Z.
            particle_idx: Sample index.
            true_label: Optional integer ground-truth class.
            
        Returns:
            Dictionary detailing final prediction, confidence, cost, trajectory, and OOD status.
        """
        K = len(CLASSES)
        
        # Step 1: Initialize prior belief
        if self.use_context_prior and self.context_prior_model is not None:
            pz = self.context_prior_model.predict_proba(context_z.reshape(1, -1))[0]
            current_log_prior = np.log(np.clip(pz, 1e-9, 1.0))
        else:
            current_log_prior = self.global_log_prior.copy()

        remaining_blocks = list(self.all_blocks)
        blocks_acquired = []
        cumulative_cost_ms = 0.0
        trajectory = []

        # Current posterior
        current_P = fuse_log_opinion_pool(current_log_prior, {}, self.block_weights)

        while len(remaining_blocks) > 0:
            max_p = compute_max_posterior(current_P)
            ent = compute_entropy(current_P)
            mrg = compute_margin(current_P)
            rem_cost = sum(self.block_costs_ms[b] for b in remaining_blocks)

            # Stopping Decision Check
            if len(blocks_acquired) >= 1:
                should_stop = False
                if self.stopping_method == "fixed":
                    if max_p >= self.fixed_tau:
                        should_stop = True
                elif self.stopping_method == "learned" and self.learned_stopping_model is not None:
                    # Assemble 19 state features
                    state_vec = np.array(
                        list(current_P) + [max_p, ent, mrg, float(len(blocks_acquired)), cumulative_cost_ms, float(len(remaining_blocks))] + list(context_z),
                        dtype=np.float32
                    ).reshape(1, -1)
                    prob_stop = self.learned_stopping_model.predict_proba(state_vec)[0, 1]
                    if prob_stop >= self.learned_stopping_threshold:
                        should_stop = True

                if should_stop:
                    break

            # Select Next Block
            chosen_block = self.policy.select_next_block(
                remaining_blocks=remaining_blocks,
                current_posterior=current_P,
                block_costs_ms=self.block_costs_ms,
                context_z={k: v for k, v in zip(Z_FEATURE_NAMES, context_z)},
                particle_idx=particle_idx,
                true_label=true_label
            )

            # Acquire Block
            blocks_acquired.append(chosen_block)
            remaining_blocks.remove(chosen_block)
            cumulative_cost_ms += self.block_costs_ms[chosen_block]

            # Update Posterior
            acquired_dict = {b: particle_probs_by_block[b] for b in blocks_acquired}
            current_P = fuse_log_opinion_pool(current_log_prior, acquired_dict, self.block_weights)

            trajectory.append({
                "step": len(blocks_acquired),
                "chosen_block": chosen_block,
                "cumulative_cost_ms": cumulative_cost_ms,
                "max_posterior": float(np.max(current_P)),
                "current_pred": int(np.argmax(current_P))
            })

        final_max_p = compute_max_posterior(current_P)
        final_ent = compute_entropy(current_P)
        pred_idx = int(np.argmax(current_P))

        # Open-Set OOD Check
        is_unknown = False
        if self.use_ood:
            if final_max_p < self.ood_tau_conf or final_ent > self.ood_tau_ent:
                is_unknown = True

        predicted_label = "UNKNOWN" if is_unknown else CLASSES[pred_idx]

        return {
            "predicted_class": predicted_label,
            "predicted_idx": -1 if is_unknown else pred_idx,
            "is_unknown": is_unknown,
            "confidence": final_max_p,
            "entropy": final_ent,
            "total_cost_ms": cumulative_cost_ms,
            "n_blocks": len(blocks_acquired),
            "blocks_acquired": blocks_acquired,
            "first_block": blocks_acquired[0] if len(blocks_acquired) > 0 else None,
            "trajectory": trajectory
        }

    def evaluate_batch(
        self,
        expert_probs_by_block: Dict[str, np.ndarray],
        context_z_matrix: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, Any]:
        """Runs the pipeline over a batch of particles and computes summary metrics."""
        N = len(y_true)
        preds = []
        costs = []
        n_blocks = []
        first_blocks = []
        unknown_flags = []

        for i in range(N):
            sample_probs = {b: expert_probs_by_block[b][i] for b in self.all_blocks}
            res = self.run_particle(
                particle_probs_by_block=sample_probs,
                context_z=context_z_matrix[i],
                particle_idx=i,
                true_label=int(y_true[i])
            )
            preds.append(res["predicted_idx"])
            costs.append(res["total_cost_ms"])
            n_blocks.append(res["n_blocks"])
            first_blocks.append(res["first_block"])
            unknown_flags.append(res["is_unknown"])

        preds = np.array(preds)
        costs = np.array(costs)
        n_blocks = np.array(n_blocks)
        unknown_flags = np.array(unknown_flags)

        # Accuracy and F1 on non-rejected samples
        valid_mask = (preds >= 0)
        acc = float(np.mean(preds[valid_mask] == y_true[valid_mask])) if np.sum(valid_mask) > 0 else 0.0
        macro_f1 = float(f1_score(y_true[valid_mask], preds[valid_mask], average="macro", zero_division=0)) if np.sum(valid_mask) > 0 else 0.0

        all_blocks_cost = float(sum(self.block_costs_ms.values()))
        mean_cost = float(np.mean(costs))
        cost_saving = float((1.0 - mean_cost / all_blocks_cost) * 100.0)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "mean_cost_ms": mean_cost,
            "mean_blocks": float(np.mean(n_blocks)),
            "cost_saving_pct": cost_saving,
            "unknown_rate": float(np.mean(unknown_flags)),
            "preds": preds,
            "costs": costs,
            "n_blocks": n_blocks,
            "first_blocks": first_blocks,
            "unknown_flags": unknown_flags
        }
