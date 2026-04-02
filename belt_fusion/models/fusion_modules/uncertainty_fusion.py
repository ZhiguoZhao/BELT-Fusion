"""
Fusion-Level Uncertainty Quantification Module

This module implements uncertainty quantification for multi-agent fusion:
- Regression uncertainty via Mahalanobis distance
- Classification uncertainty via Dempster-Shafer evidence theory
- Uncertainty-aware adaptive fusion strategy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class RegressionUncertaintyQuantifier(nn.Module):
    """
    Quantify regression uncertainty between multi-agent bounding box predictions.
    
    Uses Mahalanobis distance to measure discrepancy between Gaussian distributions
    from different agents:
        delta = sqrt((mu_i - mu_j)^T * Sigma^{-1} * (mu_i - mu_j))
    
    where Sigma = (Sigma_i + Sigma_j) / 2
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, mu_1: torch.Tensor, sigma_1: torch.Tensor,
                mu_2: torch.Tensor, sigma_2: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance between two Gaussian distributions.
        
        Args:
            mu_1: Mean of first distribution (N, 7) [x, y, z, l, w, h, theta]
            sigma_1: Covariance of first distribution (N, 7, 7)
            mu_2: Mean of second distribution (N, 7)
            sigma_2: Covariance of second distribution (N, 7, 7)
        
        Returns:
            mahalanobis_dist: Mahalanobis distance (N,)
        """
        # Difference in means
        diff = mu_1 - mu_2  # (N, 7)
        
        # Combined covariance
        sigma_combined = (sigma_1 + sigma_2) / 2.0  # (N, 7, 7)
        
        # Add small epsilon for numerical stability
        sigma_combined = sigma_combined + self_epsilon * torch.eye(
            7, device=sigma_combined.device
        ).unsqueeze(0)
        
        # Compute Mahalanobis distance
        # d^2 = (x-y)^T * Sigma^{-1} * (x-y)
        try:
            sigma_inv = torch.inverse(sigma_combined)  # (N, 7, 7)
            left = torch.bmm(diff.unsqueeze(1), sigma_inv)  # (N, 1, 7)
            mahalanobis_sq = torch.bmm(left, diff.unsqueeze(2)).squeeze()  # (N,)
            mahalanobis_dist = torch.sqrt(torch.clamp(mahalanobis_sq, min=0))
        except RuntimeError:
            # Fallback to Euclidean distance if inversion fails
            mahalanobis_dist = torch.norm(diff, dim=1)
        
        return mahalanobis_dist


class ClassificationUncertaintyQuantifier(nn.Module):
    """
    Quantify and fuse classification uncertainty using Dempster-Shafer theory.
    
    Each agent provides a mass assignment M = [{b_k}_{k=1}^K, u], where:
    - b_k: Belief mass for class k
    - u: Uncertainty mass
    
    D-S fusion rule combines masses from multiple agents:
        M = M_1 ⊕ M_2 ⊕ ... ⊕ M_V
    
    For two agents:
        b_k = (b_k^1 * b_k^2 + b_k^1 * u_2 + b_k^2 * u_1) / (1 - C)
        u = (u_1 * u_2) / (1 - C)
    
    where C = sum_{i≠j} b_i^1 * b_j^2 is the conflict factor.
    """
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of object categories
        """
        super().__init__()
        self.num_classes = num_classes
    
    def compute_mass_from_evidence(self, evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert evidence to belief masses and uncertainty.
        
        Given evidence e, compute:
            alpha = e + 1
            S = sum(alpha)
            b_k = e_k / S
            u = K / S
        
        Args:
            evidence: Evidence values (N, num_classes)
        
        Returns:
            belief_mass: Belief masses b (N, num_classes)
            uncertainty: Uncertainty u (N,)
        """
        alpha = evidence + 1.0  # (N, num_classes)
        S = alpha.sum(dim=1, keepdim=True)  # (N, 1)
        
        belief_mass = evidence / S  # (N, num_classes)
        uncertainty = self.num_classes / S.squeeze(1)  # (N,)
        
        return belief_mass, uncertainty
    
    def ds_fusion_two_agents(self, belief_1: torch.Tensor, unc_1: torch.Tensor,
                             belief_2: torch.Tensor, unc_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse mass assignments from two agents using Dempster's rule.
        
        Args:
            belief_1: Belief masses from agent 1 (N, num_classes)
            unc_1: Uncertainty from agent 1 (N,)
            belief_2: Belief masses from agent 2 (N, num_classes)
            unc_2: Uncertainty from agent 2 (N,)
        
        Returns:
            fused_belief: Fused belief masses (N, num_classes)
            fused_uncertainty: Fused uncertainty (N,)
        """
        # Conflict factor: C = sum_{i≠j} b_i^1 * b_j^2
        # This can be computed as: C = sum_i(b_i^1) * sum_j(b_j^2) - sum_i(b_i^1 * b_i^2)
        belief_sum_1 = belief_1.sum(dim=1, keepdim=True)  # (N, 1)
        belief_sum_2 = belief_2.sum(dim=1, keepdim=True)  # (N, 1)
        dot_product = (belief_1 * belief_2).sum(dim=1, keepdim=True)  # (N, 1)
        
        conflict = belief_sum_1 * belief_sum_2 - dot_product  # (N, 1)
        
        # Normalization factor
        norm_factor = 1.0 - conflict  # (N, 1)
        norm_factor = torch.clamp(norm_factor, min=1e-8)  # Avoid division by zero
        
        # Fused belief: b_k = (b_k^1 * b_k^2 + b_k^1 * u_2 + b_k^2 * u_1) / (1 - C)
        fused_belief = (
            belief_1 * belief_2 + 
            belief_1 * unc_2.unsqueeze(1) + 
            belief_2 * unc_1.unsqueeze(1)
        ) / norm_factor  # (N, num_classes)
        
        # Fused uncertainty: u = (u_1 * u_2) / (1 - C)
        fused_uncertainty = (unc_1 * unc_2) / norm_factor.squeeze(1)  # (N,)
        
        return fused_belief, fused_uncertainty
    
    def ds_fusion_multi_agent(self, beliefs: torch.Tensor, 
                              uncertainties: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Iteratively fuse mass assignments from multiple agents.
        
        Args:
            beliefs: Belief masses from V agents (V, N, num_classes)
            uncertainties: Uncertainties from V agents (V, N)
        
        Returns:
            fused_belief: Fused belief masses (N, num_classes)
            fused_uncertainty: Fused uncertainty (N,)
        """
        V, N, _ = beliefs.shape
        
        # Start with first agent
        fused_belief = beliefs[0]
        fused_uncertainty = uncertainties[0]
        
        # Iteratively fuse with remaining agents
        for v in range(1, V):
            fused_belief, fused_uncertainty = self.ds_fusion_two_agents(
                fused_belief, fused_uncertainty,
                beliefs[v], uncertainties[v]
            )
        
        return fused_belief, fused_uncertainty


class UncertaintyAwareAdaptiveFusion(nn.Module):
    """
    Uncertainty-aware adaptive fusion module for late fusion collaborative perception.
    
    This module:
    1. Matches objects across agents using Hungarian algorithm
    2. Selects optimal pairs based on minimum uncertainty
    3. Fuses bounding boxes weighted by belief masses
    4. Outputs fused boxes with quantified uncertainty
    
    This is a plug-and-play module that requires no retraining.
    """
    
    def __init__(self, num_classes: int, score_threshold: float = 0.3,
                 nms_iou_threshold: float = 0.1):
        """
        Args:
            num_classes: Number of object categories
            score_threshold: Confidence threshold for filtering detections
            nms_iou_threshold: IoU threshold for non-maximum suppression
        """
        super().__init__()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        
        self.reg_uncertainty = RegressionUncertaintyQuantifier()
        self.cls_uncertainty = ClassificationUncertaintyQuantifier(num_classes)
    
    def filter_detections(self, scores: torch.Tensor, boxes: torch.Tensor,
                         covariances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Filter low-confidence detections and apply NMS.
        
        Args:
            scores: Confidence scores (N, num_classes)
            boxes: Bounding boxes (N, 7) [x, y, z, l, w, h, yaw]
            covariances: Covariance matrices (N, 7, 7)
        
        Returns:
            filtered_scores: Filtered scores (M, num_classes)
            filtered_boxes: Filtered boxes (M, 7)
            filtered_covariances: Filtered covariances (M, 7, 7)
        """
        # Score thresholding
        max_scores = scores.max(dim=1)[0]
        mask = max_scores > self.score_threshold
        
        filtered_scores = scores[mask]
        filtered_boxes = boxes[mask]
        filtered_covariances = covariances[mask]
        
        # Apply NMS (simplified version)
        # In practice, use torchvision.ops.nms or custom 3D NMS
        keep_indices = self._apply_nms(filtered_boxes, self.nms_iou_threshold)
        
        return (
            filtered_scores[keep_indices],
            filtered_boxes[keep_indices],
            filtered_covariances[keep_indices]
        )
    
    def _apply_nms(self, boxes: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        """
        Apply non-maximum suppression.
        
        Args:
            boxes: Bounding boxes (N, 7)
            iou_threshold: IoU threshold
        
        Returns:
            keep_indices: Indices to keep
        """
        # Simplified NMS - in production, use proper 3D IoU computation
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        # Placeholder: keep all boxes
        # TODO: Implement proper 3D NMS
        return torch.arange(len(boxes), device=boxes.device)
    
    def match_objects(self, boxes_list: List[torch.Tensor], 
                     scores_list: List[torch.Tensor]) -> List[Tuple[int, int]]:
        """
        Match objects across agents using Hungarian algorithm.
        
        Args:
            boxes_list: List of boxes from V agents [(N_v, 7), ...]
            scores_list: List of scores from V agents [(N_v, num_classes), ...]
        
        Returns:
            matches: List of (agent_idx, object_idx) tuples
        """
        # For simplicity, match ego vehicle with each infrastructure agent
        # In production, implement multi-agent matching
        matches = []
        ego_boxes = boxes_list[0]
        
        for agent_idx in range(1, len(boxes_list)):
            other_boxes = boxes_list[agent_idx]
            
            # Compute cost matrix (Euclidean distance)
            if len(ego_boxes) == 0 or len(other_boxes) == 0:
                continue
            
            # Expand dimensions for broadcasting
            ego_expanded = ego_boxes.unsqueeze(1)  # (N_ego, 1, 7)
            other_expanded = other_boxes.unsqueeze(0)  # (1, N_other, 7)
            
            # Distance matrix
            dist_matrix = torch.norm(ego_expanded[:, :, :2] - other_expanded[:, :, :2], dim=2)
            
            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(dist_matrix.cpu().numpy())
            
            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] < 5.0:  # Distance threshold (5 meters)
                    matches.append((0, agent_idx, r.item(), c.item()))
        
        return matches
    
    def fuse_matched_pairs(self, boxes_1: torch.Tensor, cov_1: torch.Tensor,
                          evidence_1: torch.Tensor, boxes_2: torch.Tensor,
                          cov_2: torch.Tensor, evidence_2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fuse matched object pairs from two agents.
        
        Args:
            boxes_1: Boxes from agent 1 (N, 7)
            cov_1: Covariances from agent 1 (N, 7, 7)
            evidence_1: Evidence from agent 1 (N, num_classes)
            boxes_2: Boxes from agent 2 (N, 7)
            cov_2: Covariances from agent 2 (N, 7, 7)
            evidence_2: Evidence from agent 2 (N, num_classes)
        
        Returns:
            Dictionary containing:
                - fused_boxes: Fused bounding boxes (N, 7)
                - fused_centers: Fused box centers (N, 3)
                - reg_uncertainty: Regression uncertainty (N,)
                - cls_belief: Fused classification belief (N, num_classes)
                - cls_uncertainty: Fused classification uncertainty (N,)
        """
        # 1. Compute regression uncertainty (Mahalanobis distance)
        reg_unc = self.reg_uncertainty(boxes_1, cov_1, boxes_2, cov_2)
        
        # 2. Compute classification masses and fuse
        belief_1, unc_1 = self.cls_uncertainty.compute_mass_from_evidence(evidence_1)
        belief_2, unc_2 = self.cls_uncertainty.compute_mass_from_evidence(evidence_2)
        
        fused_belief, fused_unc = self.cls_uncertainty.ds_fusion_two_agents(
            belief_1, unc_1, belief_2, unc_2
        )
        
        # 3. Select optimal pair based on minimum uncertainty
        # (for matched pairs, we already have one-to-one correspondence)
        
        # 4. Weighted fusion of box centers using belief masses
        center_1 = boxes_1[:, :3]  # (N, 3)
        center_2 = boxes_2[:, :3]  # (N, 3)
        
        # Normalize belief masses for weighting
        total_belief_1 = belief_1.sum(dim=1, keepdim=True)  # (N, 1)
        total_belief_2 = belief_2.sum(dim=1, keepdim=True)  # (N, 1)
        
        # Softmax-like weighting
        weight_1 = torch.exp(total_belief_1) / (torch.exp(total_belief_1) + torch.exp(total_belief_2))
        weight_2 = 1.0 - weight_1
        
        fused_center = weight_1 * center_1 + weight_2 * center_2
        
        # Adjust boxes based on center offset
        fused_boxes = boxes_1.clone()
        fused_boxes[:, :3] = fused_center
        fused_boxes[:, 3:6] += (fused_center - center_1)  # Adjust dimensions
        
        return {
            'fused_boxes': fused_boxes,
            'fused_centers': fused_center,
            'reg_uncertainty': reg_unc,
            'cls_belief': fused_belief,
            'cls_uncertainty': fused_unc
        }
    
    def forward(self, detections: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Perform uncertainty-aware adaptive fusion across multiple agents.
        
        Args:
            detections: List of detection dictionaries from V agents.
                       Each dict contains:
                       - boxes: (N, 7) [x, y, z, l, w, h, yaw]
                       - scores: (N, num_classes)
                       - covariances: (N, 7, 7)
                       - evidence: (N, num_classes)
        
        Returns:
            fused_detections: List of fused detection dictionaries
        """
        if len(detections) == 0:
            return []
        
        if len(detections) == 1:
            return detections
        
        # Extract tensors from each agent
        boxes_list = [det['boxes'] for det in detections]
        scores_list = [det['scores'] for det in detections]
        cov_list = [det['covariances'] for det in detections]
        evidence_list = [det['evidence'] for det in detections]
        
        # Match objects across agents
        matches = self.match_objects(boxes_list, scores_list)
        
        # Initialize fused results
        fused_results = []
        
        # Process matched pairs
        processed = set()
        for match in matches:
            agent_1, agent_2, idx_1, idx_2 = match
            
            if (agent_1, idx_1) in processed or (agent_2, idx_2) in processed:
                continue
            
            # Extract matched detections
            det_1 = {
                'boxes': boxes_list[agent_1][idx_1:idx_1+1],
                'covariances': cov_list[agent_1][idx_1:idx_1+1],
                'evidence': evidence_list[agent_1][idx_1:idx_1+1]
            }
            det_2 = {
                'boxes': boxes_list[agent_2][idx_2:idx_2+1],
                'covariances': cov_list[agent_2][idx_2:idx_2+1],
                'evidence': evidence_list[agent_2][idx_2:idx_2+1]
            }
            
            # Fuse
            fused = self.fuse_matched_pairs(
                det_1['boxes'], det_1['covariances'], det_1['evidence'],
                det_2['boxes'], det_2['covariances'], det_2['evidence']
            )
            
            # Determine class from maximum belief
            pred_class = fused['cls_belief'].argmax(dim=1)
            
            fused_results.append({
                'boxes': fused['fused_boxes'],
                'centers': fused['fused_centers'],
                'reg_uncertainty': fused['reg_uncertainty'],
                'cls_belief': fused['cls_belief'],
                'cls_uncertainty': fused['cls_uncertainty'],
                'pred_class': pred_class
            })
            
            processed.add((agent_1, idx_1))
            processed.add((agent_2, idx_2))
        
        # Add unmatched detections from ego vehicle
        ego_boxes = boxes_list[0]
        for i in range(len(ego_boxes)):
            if (0, i) not in processed:
                fused_results.append({
                    'boxes': ego_boxes[i:i+1],
                    'centers': ego_boxes[i:i+1, :3],
                    'reg_uncertainty': torch.zeros(1, device=ego_boxes.device),
                    'cls_belief': evidence_list[0][i:i+1],
                    'cls_uncertainty': torch.ones(1, device=ego_boxes.device),
                    'pred_class': scores_list[0][i:i+1].argmax(dim=1)
                })
        
        return fused_results
