"""Dataset builder."""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional


def build_dataset(dataset_cfg: Dict[str, Any]) -> Dataset:
    """
    Build a dataset from configuration.
    
    Args:
        dataset_cfg: Dataset configuration dictionary
    
    Returns:
        dataset: PyTorch dataset instance
    """
    dataset_type = dataset_cfg.get('type')
    
    if dataset_type == 'DAIRV2XDataset':
        from .dair_v2x_dataset import DAIRV2XDataset
        return DAIRV2XDataset(**dataset_cfg)
    elif dataset_type == 'OPV2VDataset':
        from .opv2v_dataset import OPV2VDataset
        return OPV2VDataset(**dataset_cfg)
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')
