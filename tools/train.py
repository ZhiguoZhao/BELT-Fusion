"""Training script for BELT-Fusion."""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mmcv.runner import EpochBasedRunner, get_dist_info, init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# Import BELT-Fusion components
from belt_fusion.models import ProbabilisticDetectionHead, UncertaintyAwareAdaptiveFusion
from belt_fusion.datasets import build_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BELT-Fusion model')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='dair-v2x',
                       choices=['dair-v2x', 'opv2v'],
                       help='Dataset to use')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--ann-file', type=str, required=True,
                       help='Annotation file path')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='PointPillars',
                       choices=['PointPillars', 'SECOND', 'VoxelNet'],
                       help='Backbone architecture')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='Number of object classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Uncertainty arguments
    parser.add_argument('--anneal-epoch', type=int, default=10,
                       help='Epoch for KL annealing')
    
    # Output arguments
    parser.add_argument('--work-dir', type=str, required=True,
                       help='Working directory for checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Distributed training
    parser.add_argument('--launcher', type=str, default='none',
                       choices=['none', 'pytorch', 'slurm', 'mpi'],
                       help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    return args


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize distributed training
    if args.launcher != 'none':
        distributed = True
        init_dist(args.launcher, **dict(dist_backend='nccl'))
        rank, world_size = get_dist_info()
    else:
        distributed = False
        rank, world_size = 0, 1
    
    # Build dataset
    dataset_cfg = dict(
        type='DAIRV2XDataset' if args.dataset == 'dair-v2x' else 'OPV2VDataset',
        data_root=args.data_root,
        ann_file=args.ann_file,
        classes=['Car', 'Pedestrian', 'Cyclist'] if args.dataset == 'dair-v2x' else ['Car'],
    )
    
    dataset = build_dataset(dataset_cfg)
    
    # Build data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # Build model with probabilistic head
    # Note: In practice, you would load a pre-trained backbone
    model = ProbabilisticDetectionHead(
        in_channels=256,  # Depends on backbone output
        num_classes=args.num_classes,
        num_regs=7
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    if distributed:
        model = MMDistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False
        )
    else:
        model = MMDataParallel(model, device_ids=[0])
    
    # Build optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Training loop
    print(f'Start training for {args.epochs} epochs...')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            
            # Forward pass
            # Note: This is simplified - actual implementation depends on backbone integration
            features = data['features']  # From backbone
            outputs = model(features)
            
            # Compute loss
            reg_target = data['gt_bboxes_3d']
            cls_target = data['gt_labels_3d']
            
            losses = model.module.compute_loss(
                outputs, 
                reg_target, 
                cls_target, 
                epoch=epoch
            )
            
            loss_total = losses['loss_total']
            loss_total.backward()
            optimizer.step()
            
            total_loss += loss_total.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.work_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    
    print('Training finished!')


if __name__ == '__main__':
    main()
