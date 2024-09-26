_base_ = [
    # Whatever this is
    '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmdet3d.datasets.kitti_dataset',
                               'mmdet3d.engine.hooks.wandb_logger_hook',
                               'mmdet3d.evaluation.metrics.general_3ddet_metric_mmlab'], allow_failed_imports=False)                

kitti_dataset = dict(type='KittiDataset')

############# Additional Hooks #############

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=4, by_epoch=True))
custom_hooks = [
    dict(type='WandbLoggerHook', 
         save_dir='/home/cws-ml-lab/mmdetection3d_for_rail/experiments/mixed_dataset_training/rtx4090_pp_run2_mix_kitti_osdar23_3class',
         yaml_config_path='wandb_auth.yaml',
         log_artifact=True,
         init_kwargs={
             'entity': 'railsensing',
             'project': 'pointpillars',
             'name': 'rtx4090_pp_run2_mix_kitti_osdar23_3class',
             })
]

############# Generic Variables #############

class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
point_cloud_range_inference = [0, -39.68, -3, 69.12, 39.68, 10]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

points_loader = dict(
    type='LoadPointsFromFile',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=4,
    backend_args=None)

generic_test_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

generic_eval_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

############# Kitti Specific Config #############

kitti_data_root = 'data/kitti/'
kitti_dataset_type = 'KittiDataset'

kitti_db_sampler = dict(
    data_root=kitti_data_root,
    info_path=kitti_data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=10, Pedestrian=10, Cyclist=10)),
    classes=class_names,
    sample_groups=dict(Car=10, Pedestrian=10, Cyclist=10),
    points_loader=points_loader,
    backend_args=None)
    
kitti_train_pipeline = [
    points_loader,
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=kitti_db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

kitti_train_dataset = dict(
    type=kitti_dataset_type,
    data_root=kitti_data_root,
    ann_file= 'kitti_infos_train.pkl',
    data_prefix=dict(pts='training/velodyne_reduced'),
    pipeline=kitti_train_pipeline,
    modality=input_modality,
    test_mode=False,
    metainfo=metainfo,
    backend_args=None)

kitti_val_dataset = dict(
    type=kitti_dataset_type,
    data_root=kitti_data_root,
    data_prefix=dict(pts='training/velodyne_reduced'),
    ann_file='kitti_infos_val.pkl',
    pipeline=generic_eval_pipeline,
    modality=input_modality,
    test_mode=True,
    metainfo=metainfo,
    backend_args=None)

kitti_test_dataset = kitti_val_dataset

kitti_repeat_dataset = dict(
    type='RepeatDataset',
    times=1,
    dataset=kitti_train_dataset)
 
############# Dataloader Config #############
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[kitti_repeat_dataset],
    )
)

val_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[kitti_val_dataset],
    )
)

test_dataloader = val_dataloader    

# For the val and test evaluator, we do not need to specify the annotation files
val_evaluator = dict(
    type='General_3dDet_Metric_MMLab',
    metric='det3d',
    classes=class_names,
    pcd_limit_range=point_cloud_range_inference,
    output_dir='/home/cws-ml-lab/mmdetection3d_for_rail/experiments/mixed_dataset_training/rtx4090_pp_run2_mix_kitti_osdar23_3class/evaluation',
    save_graphics = False,
    save_evaluation_results = False,
    save_random_viz = False,
    random_viz_keys = None)

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

############# Model Config #############

voxel_size = [0.16, 0.16, 4]

kitti_object_sizes = [[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=4,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -0.6, 69.12, 39.68, -0.6],
                [0, -39.68, -1.78, 69.12, 39.68, -1.78],
            ],
            sizes=kitti_object_sizes,
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

############# Scheduler #############

# The schedule is usually used by models trained on KITTI dataset
# The learning rate set in the cyclic schedule is the initial learning rate
# rather than the max learning rate. Since the target_ratio is (10, 1e-4),
# the learning rate will change from 0.0018 to 0.018, than go to 0.0018*1e-4
lr = 0.0018
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the official AdamW optimizer implemented by PyTorch.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
# learning rate
epoch_num = 80

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num * 1,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=6)

############# Default Runtime Config #############

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=-1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

############# Work Directory #############
work_dir = '/home/cws-ml-lab/mmdetection3d_for_rail/experiments/mixed_dataset_training/rtx4090_pp_run2_mix_kitti_osdar23_3class'