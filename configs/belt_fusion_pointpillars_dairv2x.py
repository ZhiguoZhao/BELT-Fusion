# BELT-Fusion Configuration for PointPillars on DAIR-V2X
# Bayesian Evidential Late Fusion for Trustworthy V2X Perception

model = dict(
    type='BELTFusion',
    backbone=dict(
        type='PointPillarsBackbone',
        num_features=[64, 128, 256],
        use_norm=True,
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256],
        out_channels=256,
        num_outs=3,
    ),
    bbox_head=dict(
        type='ProbabilisticDetectionHead',
        in_channels=256,
        num_classes=3,  # Car, Pedestrian, Cyclist
        num_regs=7,     # [dx, dy, dz, log(l), log(w), log(h), cos/sin(theta)]
        loss_cfg=dict(
            type='UncertaintyLoss',
            reg_weight=1.0,
            cls_weight=1.0,
        ),
    ),
    fusion_module=dict(
        type='UncertaintyAwareAdaptiveFusion',
        num_classes=3,
        score_threshold=0.3,
        nms_iou_threshold=0.1,
    ),
)

# Dataset settings
dataset_type = 'DAIRV2XDataset'
data_root = 'data/DAIR-V2X/'
class_names = ['Car', 'Pedestrian', 'Cyclist']

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
)

# Training settings
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=[-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=[-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[
             dict(type='GlobalRotScaleTrans', rot_range=[0, 0], scale_ratio_range=[1., 1.], translation_std=[0, 0, 0]),
             dict(type='RandomFlip3D'),
             dict(type='NormalizeMultiviewImage', **img_norm_cfg),
             dict(type='PadMultiViewImage', size_divisor=32),
             dict(type='PointsRangeFilter', point_cloud_range=[-75.2, -75.2, -2.0, 75.2, 75.2, 4.0]),
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['points']),
         ]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'v2x_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'v2x_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'v2x_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
    ),
)

# Optimization
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5,
)

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metric='bbox')

# Runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# Uncertainty-specific settings
uncertainty_cfg = dict(
    # Agent-level uncertainty
    agent_uncertainty=dict(
        regression=dict(
            type='HeteroscedasticGaussian',
            init_log_var=0.0,
        ),
        classification=dict(
            type='EvidentialDirichlet',
            anneal_epoch=10,
        ),
    ),
    # Fusion-level uncertainty
    fusion_uncertainty=dict(
        regression=dict(
            type='MahalanobisDistance',
        ),
        classification=dict(
            type='DempsterShaferFusion',
        ),
    ),
    # Adaptive fusion
    adaptive_fusion=dict(
        enabled=True,
        uncertainty_threshold=0.5,
        min_belief_threshold=0.3,
    ),
)
