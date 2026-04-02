"""
Probabilistic Detection Head for 3D Object Detection

This module implements uncertainty quantification for object detection:
- Regression uncertainty using heteroscedastic Gaussian modeling
- Classification uncertainty using evidential deep learning (Dirichlet distribution)

Based on:
- Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning?", NeurIPS 2017
- Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ProbabilisticRegressionHead(nn.Module):
    """
    Probabilistic regression head that predicts mean and uncertainty (variance).
    
    For each regression target, predicts both the mean (u) and log-variance (s = log(sigma^2)).
    The loss function is:
        L_reg = 0.5 * exp(-s) * ||y_gt - u||^2 + 0.5 * s
    
    This corresponds to maximizing the likelihood under a Gaussian assumption with 
    learnable variance.
    """
    
    def __init__(self, in_channels: int, num_regs: int = 7):
        """
        Args:
            in_channels: Number of input channels from RoI features
            num_regs: Number of regression parameters (default: 7 for 3D bbox)
                     [dx, dy, dz, log(l), log(w), log(h), sin/cos(theta)]
        """
        super().__init__()
        self.num_regs = num_regs
        
        # Mean prediction branch
        self.mean_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_regs)
        )
        
        # Uncertainty (log-variance) prediction branch
        self.var_branch = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_regs)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features of shape (N, in_channels)
        
        Returns:
            mean: Predicted mean of shape (N, num_regs)
            log_var: Predicted log-variance of shape (N, num_regs)
        """
        mean = self.mean_branch(x)
        log_var = self.var_branch(x)
        return mean, log_var
    
    @staticmethod
    def regression_loss(mean: torch.Tensor, log_var: torch.Tensor, 
                       target: torch.Tensor) -> torch.Tensor:
        """
        Compute heteroscedastic regression loss.
        
        L = 0.5 * exp(-s) * ||y_gt - u||^2 + 0.5 * s
        where s = log(sigma^2)
        
        Args:
            mean: Predicted mean (N, num_regs)
            log_var: Predicted log-variance (N, num_regs)
            target: Ground truth values (N, num_regs)
        
        Returns:
            loss: Scalar loss value
        """
        diff = target - mean
        loss = 0.5 * torch.exp(-log_var) * (diff ** 2) + 0.5 * log_var
        return loss.mean()


class EvidentialClassificationHead(nn.Module):
    """
    Evidential classification head based on Subjective Logic and Dirichlet distributions.
    
    Instead of predicting point estimates (softmax probabilities), this head predicts
    evidence values for each class, which are then used to parameterize a Dirichlet 
    distribution over class probabilities.
    
    The Dirichlet parameters alpha = e + 1, where e >= 0 is the evidence.
    Uncertainty u = K / S, where S = sum(alpha_k) and K is the number of classes.
    Belief mass b_k = e_k / S.
    
    Based on: Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty"
    """
    
    def __init__(self, in_channels: int, num_classes: int):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of object categories
        """
        super().__init__()
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
            nn.ReLU(inplace=True)  # ReLU ensures non-negative evidence
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features of shape (N, in_channels)
        
        Returns:
            evidence: Evidence values e (N, num_classes), e >= 0
            alpha: Dirichlet parameters alpha = e + 1 (N, num_classes)
            uncertainty: Uncertainty measure u (N,)
        """
        evidence = self.classifier(x)
        alpha = evidence + 1.0  # Ensure alpha > 0
        
        # Compute uncertainty: u = K / S, where S = sum(alpha)
        S = alpha.sum(dim=1, keepdim=True)
        uncertainty = self.num_classes / S.squeeze(1)
        
        return evidence, alpha, uncertainty
    
    @staticmethod
    def mse_loss(alpha: torch.Tensor, target: torch.Tensor, 
                 epoch: int = 0) -> torch.Tensor:
        """
        Compute integrated squared error loss for evidential classification.
        
        L = sum_k [(y_k - alpha_k/S)^2 + alpha_k*(S - alpha_k)/(S^2*(S+1))]
        
        Args:
            alpha: Dirichlet parameters (N, num_classes)
            target: One-hot encoded ground truth (N, num_classes)
            epoch: Current training epoch (for annealing)
        
        Returns:
            loss: MSE loss without KL regularization
        """
        S = alpha.sum(dim=1, keepdim=True)
        
        # Bias term: (y - alpha/S)^2
        bias_term = ((target - alpha / S) ** 2).sum(dim=1)
        
        # Variance term: alpha*(S-alpha)/(S^2*(S+1))
        var_term = (alpha * (S - alpha) / (S ** 2 * (S + 1))).sum(dim=1)
        
        loss = bias_term + var_term
        return loss.mean()
    
    @staticmethod
    def kl_divergence_loss(alpha: torch.Tensor, target: torch.Tensor, 
                          epoch: int = 0, max_epoch: int = 10) -> torch.Tensor:
        """
        Compute KL divergence regularization to prevent overconfidence.
        
        KL[Dir(p|alpha_tilde) || Dir(p|1)] where alpha_tilde = y + (1-y)*alpha
        
        This pushes non-target evidence toward zero, encouraging uniform 
        distributions when uncertain.
        
        Args:
            alpha: Dirichlet parameters (N, num_classes)
            target: One-hot encoded ground truth (N, num_classes)
            epoch: Current training epoch
            max_epoch: Maximum epochs for annealing
        
        Returns:
            kl_loss: KL divergence regularization loss
        """
        # Annealing coefficient
        lambda_t = min(1.0, epoch / max_epoch)
        
        # Masked Dirichlet parameters (remove non-target evidence)
        alpha_tilde = target + (1 - target) * alpha
        
        # KL divergence computation
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        # Log beta function terms
        ln_B = torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True) - \
               torch.lgamma(S_tilde)
        ln_B_ones = -torch.lgamma(torch.tensor(float(alpha_tilde.shape[1]), 
                                               device=alpha.device))
        
        # Digamma terms
        psi_alpha = torch.digamma(alpha_tilde)
        psi_S = torch.digamma(S_tilde)
        
        digamma_term = ((alpha_tilde - 1) * (psi_alpha - psi_S)).sum(dim=1, keepdim=True)
        
        kl = ln_B - ln_B_ones + digamma_term
        kl_loss = lambda_t * kl.mean()
        
        return kl_loss
    
    def total_loss(self, alpha: torch.Tensor, target: torch.Tensor,
                  epoch: int = 0, max_epoch: int = 10) -> torch.Tensor:
        """
        Compute total evidential classification loss (MSE + KL regularization).
        
        Args:
            alpha: Dirichlet parameters (N, num_classes)
            target: One-hot encoded ground truth (N, num_classes)
            epoch: Current training epoch
            max_epoch: Maximum epochs for annealing
        
        Returns:
            total_loss: Combined MSE + KL loss
        """
        mse_loss = self.mse_loss(alpha, target, epoch)
        kl_loss = self.kl_divergence_loss(alpha, target, epoch, max_epoch)
        return mse_loss + kl_loss


class ProbabilisticDetectionHead(nn.Module):
    """
    Combined probabilistic detection head with both classification and regression 
    uncertainty quantification.
    
    This is a plug-and-play module that can replace standard detection heads in 
    existing 3D object detectors (PointPillars, SECOND, VoxelNet, etc.).
    """
    
    def __init__(self, in_channels: int, num_classes: int, num_regs: int = 7):
        """
        Args:
            in_channels: Input feature channels
            num_classes: Number of object categories
            num_regs: Number of regression parameters per box
        """
        super().__init__()
        self.regression_head = ProbabilisticRegressionHead(in_channels, num_regs)
        self.classification_head = EvidentialClassificationHead(in_channels, num_classes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features (N, in_channels)
        
        Returns:
            Dictionary containing:
                - reg_mean: Regression mean predictions (N, num_regs)
                - reg_log_var: Regression log-variance (N, num_regs)
                - evidence: Classification evidence (N, num_classes)
                - alpha: Dirichlet parameters (N, num_classes)
                - cls_uncertainty: Classification uncertainty (N,)
        """
        reg_mean, reg_log_var = self.regression_head(x)
        evidence, alpha, cls_uncertainty = self.classification_head(x)
        
        return {
            'reg_mean': reg_mean,
            'reg_log_var': reg_log_var,
            'evidence': evidence,
            'alpha': alpha,
            'cls_uncertainty': cls_uncertainty
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                    reg_target: torch.Tensor,
                    cls_target: torch.Tensor,
                    epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute combined classification and regression losses.
        
        Args:
            outputs: Forward pass outputs
            reg_target: Regression targets (N, num_regs)
            cls_target: Classification targets (one-hot, N, num_classes)
            epoch: Current training epoch
        
        Returns:
            Dictionary of losses:
                - loss_reg: Regression loss
                - loss_cls: Classification loss
                - loss_total: Total loss
        """
        loss_reg = self.regression_head.regression_loss(
            outputs['reg_mean'], 
            outputs['reg_log_var'], 
            reg_target
        )
        
        loss_cls = self.classification_head.total_loss(
            outputs['alpha'],
            cls_target,
            epoch
        )
        
        return {
            'loss_reg': loss_reg,
            'loss_cls': loss_cls,
            'loss_total': loss_reg + loss_cls
        }
