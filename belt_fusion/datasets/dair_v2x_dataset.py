"""
DAIR-V2X Dataset for V2X Collaborative Perception

DAIR-V2X is a real-world V2X dataset with vehicle and infrastructure LiDAR sensors.
Reference: Yu et al., "DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection", CVPR 2022
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class DAIRV2XDataset(Dataset):
    """
    DAIR-V2X dataset for collaborative perception.
    
    Supports both single-agent and multi-agent (V2I) scenarios.
    Each sample contains data from ego vehicle and optionally infrastructure sensors.
    """
    
    def __init__(self, data_root: str, ann_file: str, pipeline: List = None,
                 classes: List[str] = None, modality: Dict = None,
                 test_mode: bool = False):
        """
        Args:
            data_root: Root directory of DAIR-V2X dataset
            ann_file: Annotation file path (pickle format)
            pipeline: Data processing pipeline
            classes: Class names to use
            modality: Sensor modality configuration
            test_mode: Whether in test mode
        """
        self.data_root = data_root
        self.ann_file = ann_file
        self.pipeline = pipeline or []
        self.classes = classes or ['Car', 'Pedestrian', 'Cyclist']
        self.modality = modality or dict(use_lidar=True)
        self.test_mode = test_mode
        
        # Load annotations
        self.data_infos = self._load_annotations(ann_file)
        
        # Class name to ID mapping
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
        self.id_to_class = {i: name for i, name in enumerate(self.classes)}
    
    def _load_annotations(self, ann_file: str) -> List[Dict]:
        """Load annotations from pickle file."""
        print(f'Loading annotations from {ann_file}...')
        with open(ann_file, 'rb') as f:
            data_infos = pickle.load(f)
        print(f'Loaded {len(data_infos)} samples')
        return data_infos
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_infos)
    
    def get_data_info(self, index: int) -> Dict:
        """
        Get basic data info for a sample.
        
        Args:
            index: Sample index
        
        Returns:
            info: Dictionary containing sample metadata
        """
        info = self.data_infos[index]
        
        input_dict = {
            'sample_idx': index,
            'lidar_path': os.path.join(self.data_root, info['lidar_path']),
            'timestamp': info.get('timestamp', 0),
        }
        
        # Add infrastructure data if available
        if 'infrastructure_lidar_path' in info:
            input_dict['infrastructure_lidar_path'] = os.path.join(
                self.data_root, info['infrastructure_lidar_path']
            )
        
        if not self.test_mode:
            input_dict['ann_info'] = {
                'gt_bboxes_3d': info.get('gt_bboxes_3d'),
                'gt_labels_3d': info.get('gt_labels_3d'),
            }
        
        return input_dict
    
    def prepare_data(self, index: int) -> Dict:
        """
        Prepare data for a sample by applying pipeline transforms.
        
        Args:
            index: Sample index
        
        Returns:
            data_dict: Processed data dictionary
        """
        input_dict = self.get_data_info(index)
        
        # Apply pipeline transforms
        for transform in self.pipeline:
            input_dict = transform(input_dict)
        
        return input_dict
    
    def __getitem__(self, index: int) -> Dict:
        """Get a data sample."""
        return self.prepare_data(index)
    
    def evaluate(self, results: List[Dict], metric: str = 'bbox', 
                 logger=None) -> Dict[str, float]:
        """
        Evaluate detection results.
        
        Args:
            results: List of prediction dictionaries
            metric: Evaluation metric ('bbox' or 'bev')
            logger: Logger object
        
        Returns:
            eval_results: Dictionary of evaluation metrics
        """
        from .evaluation import evaluate_detection
        
        eval_results = evaluate_detection(
            results, 
            self.data_infos,
            metric=metric,
            classes=self.classes
        )
        
        return eval_results
    
    def get_collaborative_sample(self, vehicle_idx: int) -> Dict:
        """
        Get a collaborative perception sample with both vehicle and infrastructure data.
        
        Args:
            vehicle_idx: Index of the vehicle sample
        
        Returns:
            collab_data: Dictionary containing multi-agent data
        """
        vehicle_info = self.data_infos[vehicle_idx]
        
        collab_data = {
            'vehicle': {
                'lidar_path': os.path.join(self.data_root, vehicle_info['lidar_path']),
                'pose': vehicle_info.get('pose'),
            },
            'infrastructure': None,
        }
        
        # Load infrastructure data if available
        if 'infrastructure_lidar_path' in vehicle_info:
            infra_info = vehicle_info['infrastructure_info']
            collab_data['infrastructure'] = {
                'lidar_path': os.path.join(self.data_root, vehicle_info['infrastructure_lidar_path']),
                'pose': infra_info.get('pose'),
            }
        
        return collab_data
