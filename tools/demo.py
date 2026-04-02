"""
Demo script for BELT-Fusion inference and visualization.

This script demonstrates how to:
1. Load a trained BELT-Fusion model
2. Run inference on LiDAR data
3. Visualize uncertainty predictions
4. Compare with and without uncertainty quantification
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='BELT-Fusion Demo')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--input', type=str, help='Input LiDAR file or directory')
    parser.add_argument('--output', type=str, default='./demo_output', help='Output directory')
    parser.add_argument('--show-uncertainty', action='store_true', help='Visualize uncertainty')
    args = parser.parse_args()
    return args


def load_point_cloud(file_path):
    """Load point cloud from file."""
    if file_path.endswith('.bin'):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    elif file_path.endswith('.pcd'):
        # Use open3d or pypcd for PCD files
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f'Unsupported file format: {file_path}')
    return points


def visualize_detections(points, boxes, uncertainties=None, output_path=None):
    """
    Visualize 3D detections with optional uncertainty heatmap.
    
    Args:
        points: Point cloud (N, 4)
        boxes: Detected boxes (M, 7) [x, y, z, l, w, h, yaw]
        uncertainties: Uncertainty values (M,)
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Plot point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                c=points[:, 3], cmap='viridis', s=0.5, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('LiDAR Point Cloud')
    
    # Plot detections with uncertainty coloring
    ax2 = fig.add_subplot(122)
    
    if uncertainties is not None:
        # Color boxes by uncertainty
        norm = plt.Normalize(vmin=uncertainties.min(), vmax=uncertainties.max())
        cmap = plt.cm.Reds
        
        for i, (box, unc) in enumerate(zip(boxes, uncertainties)):
            x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
            
            # Create rectangle for box
            rect = plt.Rectangle((x - l/2, y - w/2), l, w, angle=yaw*180/np.pi,
                                facecolor=cmap(norm(unc)), edgecolor='black', linewidth=2)
            ax2.add_patch(rect)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Uncertainty')
    else:
        # Standard visualization
        for box in boxes:
            x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
            rect = plt.Rectangle((x - l/2, y - w/2), l, w, angle=yaw*180/np.pi,
                                fill=False, color='red', linewidth=2)
            ax2.add_patch(rect)
    
    ax2.plot(points[:, 0], points[:, 1], '.', markersize=1, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Detection Results' + (' with Uncertainty' if uncertainties is not None else ''))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {output_path}')
    else:
        plt.show()


def demo_inference(config_path, checkpoint_path, input_data):
    """
    Run BELT-Fusion inference on input data.
    
    This is a simplified demo - full implementation requires integrating
    with a backbone network (PointPillars, etc.)
    """
    print(f'Loading config from {config_path}')
    print(f'Loading checkpoint from {checkpoint_path}')
    print(f'Processing input: {input_data}')
    
    # Load model
    from belt_fusion.models import ProbabilisticDetectionHead
    
    # Placeholder - in practice, load full model with backbone
    model = ProbabilisticDetectionHead(
        in_channels=256,
        num_classes=3,
        num_regs=7
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    
    print('Model loaded successfully!')
    
    # Process input
    if Path(input_data).is_dir():
        lidar_files = list(Path(input_data).glob('*.bin'))
        print(f'Found {len(lidar_files)} LiDAR files')
        
        for lidar_file in lidar_files[:5]:  # Process first 5 files
            points = load_point_cloud(str(lidar_file))
            print(f'\nProcessing {lidar_file.name}: {len(points)} points')
            
            # Inference would go here
            # For demo, just show statistics
            print(f'  X range: [{points[:, 0].min():.1f}, {points[:, 0].max():.1f}] m')
            print(f'  Y range: [{points[:, 1].min():.1f}, {points[:, 1].max():.1f}] m')
            print(f'  Z range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}] m')
    
    print('\nDemo completed!')


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.config and args.checkpoint:
        demo_inference(args.config, args.checkpoint, args.input)
    else:
        print('Usage:')
        print('  python demo.py --config CONFIG --checkpoint CHECKPOINT --input DATA')
        print('\nExample:')
        print('  python demo.py --config configs/belt_fusion_pointpillars_dairv2x.py \\')
        print('                 --checkpoint work_dirs/belt_fusion/checkpoint.pth \\')
        print('                 --input data/DAIR-V2X/sample/')
    
    print('\nTo visualize results, run:')
    print('  python tools/demo_visualize.py --input results.pkl --output vis/')


if __name__ == '__main__':
    main()
