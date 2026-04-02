# BELT-Fusion Configuration for PointPillars on OPV2V

_base_ = './belt_fusion_pointpillars_dairv2x.py'

dataset_type = 'OPV2VDataset'
data_root = 'data/OPV2V/'

# OPV2V-specific settings
class_names = ['Car']

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
)

# OPV2V has different point cloud range
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=[-70.4, -70.4, -2.0, 70.4, 70.4, 4.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.3925, 0.3925], scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=[-70.4, -70.4, -2.0, 70.4, 70.4, 4.0]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'opv2v_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'opv2v_infos_val.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'opv2v_infos_test.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
    ),
)

# OPV2V training schedule
runner = dict(type='EpochBasedRunner', max_epochs=80)
