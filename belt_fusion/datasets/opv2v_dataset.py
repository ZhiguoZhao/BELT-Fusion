"""
OPV2V Dataset for V2V Collaborative Perception

OPV2V is a large-scale simulated V2V dataset generated with CARLA and OpenCDA.
Reference: Xu et al., "OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication", ICRA 2022
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class OPV2VDataset(Dataset):
    """
    OPV2V dataset for V2V collaborative perception.
    
    Supports multi-vehicle scenarios with different connectivity levels.
    """
    
    def __init__(self, data_root: str, ann_file: str, pipeline: List = None,
                 classes: List[str] = None, modality: Dict = None,
                 test_mode: bool = False, num_connected_vehicles: int = 1):
        """
        Args:
            data_root: Root directory of OPV2V dataset
            ann_file: Annotation file path
            pipeline: Data processing pipeline
            classes: Class names to use
            modality: Sensor modality configuration
            test_mode: Whether in test mode
            num_connected_vehicles: Number of connected vehicles (1 = single vehicle)
        """
        self.data_root = data_root
        self.ann_file = ann_file
        self.pipeline = pipeline or []
        self.classes = classes or ['Car']
        self.modality = modality or dict(use_lidar=True)
        self.test_mode = test_mode
        self.num_connected_vehicles = num_connected_vehicles
        
        # Load annotations
        self.data_infos = self._load_annotations(ann_file)
        
        self.class_to_id = {name: i for i, name in enumerate(self.classes)}
    
    def _load_annotations(self, ann_file: str) -> List[Dict]:
        """Load annotations from pickle file."""
        print(f'Loading OPV2V annotations from {ann_file}...')
        with open(ann_file, 'rb') as f:
            data_infos = pickle.load(f)
        print(f'Loaded {len(data_infos)} samples')
        return data_infos
    
    def __len__(self) -> int:
        return len(self.data_infos)
    
    def get_data_info(self, index: int) -> Dict:
        """Get basic data info."""
        info = self.data_infos[index]
        
        input_dict = {
            'sample_idx': index,
            'ego_lidar_path': os.path.join(self.data_root, info['ego_lidar_path']),
            'timestamp': info.get('timestamp', 0),
        }
        
        # Add other connected vehicles
        if 'connected_vehicles' in info:
            input_dict['connected_vehicles'] = []
            for veh_info in info['connected_vehicles'][:self.num_connected_vehicles - 1]:
                input_dict['connected_vehicles'].append({
                    'lidar_path': os.path.join(self.data_root, veh_info['lidar_path']),
                    'pose': veh_info.get('pose'),
                })
        
        if not self.test_mode:
            input_dict['ann_info'] = {
                'gt_bboxes_3d': info.get('gt_bboxes_3d'),
                'gt_labels_3d': info.get('gt_labels_3d'),
            }
        
        return input_dict
    
    def __getitem__(self, index: int) -> Dict:
        input_dict = self.get_data_info(index)
        
        for transform in self.pipeline:
            input_dict = transform(input_dict)
        
        return input_dict
    
    def evaluate(self, results: List[Dict], metric: str = 'bbox',
                 logger=None) -> Dict[str, float]:
        """Evaluate detection results."""
        from .evaluation import evaluate_detection
        
        eval_results = evaluate_detection(
            results,
            self.data_infos,
            metric=metric,
            classes=self.classes
        )
        
        return eval_results
