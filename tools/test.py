"""Evaluation and inference script for BELT-Fusion."""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BELT-Fusion model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file path')
    parser.add_argument('--data-root', type=str, required=True, help='Dataset root')
    parser.add_argument('--ann-file', type=str, required=True, help='Annotation file')
    parser.add_argument('--out-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--eval-metric', type=str, default='bbox', choices=['bbox', 'bev'])
    parser.add_argument('--show-dir', type=str, help='Directory to visualize results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # Load config
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    
    # Build dataset
    from belt_fusion.datasets import build_dataset
    dataset_cfg = dict(
        type='DAIRV2XDataset' if 'dair' in args.ann_file.lower() else 'OPV2VDataset',
        data_root=args.data_root,
        ann_file=args.ann_file,
        test_mode=True,
    )
    dataset = build_dataset(dataset_cfg)
    
    # Build model
    from belt_fusion.models import ProbabilisticDetectionHead, UncertaintyAwareAdaptiveFusion
    
    model = ProbabilisticDetectionHead(
        in_channels=cfg.model.bbox_head.in_channels,
        num_classes=cfg.model.bbox_head.num_classes,
        num_regs=cfg.model.bbox_head.num_regs
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    
    # Build fusion module
    fusion_module = UncertaintyAwareAdaptiveFusion(
        num_classes=cfg.model.fusion_module.num_classes,
        score_threshold=cfg.model.fusion_module.score_threshold,
    )
    
    # Run inference
    print('Running inference...')
    results = []
    
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        
        with torch.no_grad():
            # Get predictions from each agent
            agent_outputs = []
            
            # Ego vehicle
            ego_features = data['features'].cuda()
            ego_output = model(ego_features)
            agent_outputs.append({
                'boxes': ego_output['reg_mean'],
                'scores': ego_output['alpha'] / ego_output['alpha'].sum(dim=1, keepdim=True),
                'covariances': compute_covariance_from_log_var(ego_output['reg_log_var']),
                'evidence': ego_output['evidence'],
            })
            
            # Infrastructure (if available)
            if 'infrastructure_features' in data:
                infra_features = data['infrastructure_features'].cuda()
                infra_output = model(infra_features)
                agent_outputs.append({
                    'boxes': infra_output['reg_mean'],
                    'scores': infra_output['alpha'] / infra_output['alpha'].sum(dim=1, keepdim=True),
                    'covariances': compute_covariance_from_log_var(infra_output['reg_log_var']),
                    'evidence': infra_output['evidence'],
                })
            
            # Fuse predictions
            fused_results = fusion_module(agent_outputs)
            results.extend(fused_results)
    
    # Evaluate
    print('Evaluating results...')
    eval_results = dataset.evaluate(results, metric=args.eval_metric)
    
    # Print results
    print('\n=== Evaluation Results ===')
    for metric, value in eval_results.items():
        print(f'{metric}: {value:.4f}')
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'results.pkl')
    
    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump({'results': results, 'metrics': eval_results}, f)
    
    print(f'Results saved to {results_path}')


def compute_covariance_from_log_var(log_var: torch.Tensor) -> torch.Tensor:
    """Convert log-variance to diagonal covariance matrix."""
    var = torch.exp(log_var)
    batch_size = var.shape[0]
    num_dims = var.shape[1]
    
    cov = torch.zeros(batch_size, num_dims, num_dims, device=var.device)
    for i in range(batch_size):
        cov[i] = torch.diag(var[i])
    
    return cov


if __name__ == '__main__':
    main()
