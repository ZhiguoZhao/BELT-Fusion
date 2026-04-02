"""
Uncertainty Heads Module

This package provides probabilistic detection heads for uncertainty quantification:
- ProbabilisticRegressionHead: Heteroscedastic regression uncertainty
- EvidentialClassificationHead: Dirichlet-based classification uncertainty
- ProbabilisticDetectionHead: Combined head for 3D object detection
"""

from .probabilistic_head import (
    ProbabilisticRegressionHead,
    EvidentialClassificationHead,
    ProbabilisticDetectionHead
)

__all__ = [
    'ProbabilisticRegressionHead',
    'EvidentialClassificationHead',
    'ProbabilisticDetectionHead'
]
