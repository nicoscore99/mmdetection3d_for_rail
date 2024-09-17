# dataset settings
custom_imports = dict(imports=['mmdet3d.datasets.kitti_osdar23_merge_dataset', 
                               'mmdet3d.engine.hooks.wandb_logger_hook',
                               'mmdet3d.evaluation.metrics.general_3ddet_metric_mmlab'], allow_failed_imports=False)
dataset = dict(type='KittiOSDaR23Merge')
dataset_type = 'KittiOSDaR23Merge'
data_root = 'data/kitti_osdar23_merge/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'merged_infos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(Pedestrian=5, Cyclist=5, Car=5),
    classes=class_names,
    sample_groups=dict(Pedestrian=5, Cyclist=5, Car=5),
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='Pack3DDetInputs',
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='merged_infos_train.pkl',
            data_prefix=dict(pts='training/points'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='testing/points'),
        ann_file='merged_infos_val.pkl',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(pts='testing/points'),
        ann_file='merged_infos_val.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='General_3dDet_Metric_MMLab',
    ann_file=data_root + 'merged_infos_val.pkl',
    metric='det3d',
    classes=class_names,
    # output_dir='data/osdar23_3class/training/rtx4090_pvrcnn_run1_3class/',
    pcd_limit_range=[0, -39.68, -10, 69.12, 39.68, 10],
    save_graphics=True,
    backend_args=backend_args)
test_evaluator = val_evaluator

# val_evaluator = dict()
# test_evaluator = dict()

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
