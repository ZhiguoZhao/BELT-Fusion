"""
Fusion Modules Package

This package provides fusion-level uncertainty quantification and adaptive fusion:
- RegressionUncertaintyQuantifier: Mahalanobis distance-based regression uncertainty
- ClassificationUncertaintyQuantifier: Dempster-Shafer theory for classification fusion
- UncertaintyAwareAdaptiveFusion: Complete uncertainty-aware fusion pipeline
"""

from .uncertainty_fusion import (
    RegressionUncertaintyQuantifier,
    ClassificationUncertaintyQuantifier,
    UncertaintyAwareAdaptiveFusion
)

__all__ = [
    'RegressionUncertaintyQuantifier',
    'ClassificationUncertaintyQuantifier',
    'UncertaintyAwareAdaptiveFusion'
]
