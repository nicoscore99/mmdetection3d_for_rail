# _base_ = [
#     # Whatever this is
#     '../_base_/default_runtime.py'
# ]

custom_imports = dict(imports=['mmdet3d.datasets.osdar23_dataset',
                               'mmdet3d.datasets.kitti_dataset',
                                 'mmdet3d.datasets.robosense_m1_plus_dataset',
                               'mmdet3d.engine.hooks.wandb_logger_hook',
                               'mmdet3d.evaluation.metrics.general_3ddet_metric_mmlab'], allow_failed_imports=False)                

kitti_dataset = dict(type='KittiDataset')
osdar23_dataset = dict(type='OSDaR23Dataset')

############# Additional Hooks #############

# default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=4, by_epoch=True))
custom_hooks = [
    dict(type='WandbLoggerHook', 
         save_dir='/home/cws-ml-lab/mmdetection3d_for_rail/checkpoints',
         yaml_config_path='wandb_auth.yaml',
         log_artifact=True,
         init_kwargs={
             'entity': 'railsensing',
             'project': 'centerpoint',
             'name': 'rtx4090_cp_run8_mix_kitti_osdar23_4class_80m',
             })
]

############# Generic variables #############

# class_names = ['Pedestrian', 'Cyclist', 'Car']
class_names = ['Pedestrian', 'Cyclist', 'RoadVehicle', 'Train']
point_cloud_range = [0, -40, -3.0, 80, 40, 1.0]
# point_cloud_range = [0, -40, -3, 70.4, 40, 3.0]
point_cloud_range_inference = point_cloud_range
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

kitti_data_root = 'data/kitti_cls/'
kitti_dataset_type = 'KittiDataset'

kitti_db_sampler = dict(
    data_root=kitti_data_root,
    info_path=kitti_data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Pedestrian=10, Cyclist=10, RoadVehicle=10, Train=10)),
    classes=class_names,
    sample_groups=dict(Pedestrian=10, Cyclist=10, RoadVehicle=10, Train=10),
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
    # indices=0.1,
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

############# OSDAR23 Specific Config #############

osdar23_data_root = 'data/osdar23_cls/'
osdar23_dataset_type = 'OSDaR23Dataset'

osdar23_db_sampler = dict(
    data_root=osdar23_data_root,
    info_path=osdar23_data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(Pedestrian=20, Cyclist=20, RoadVehicle=20, Train=20)
    ),
    classes=class_names,
    sample_groups=dict(Pedestrian=10, Cyclist=10, RoadVehicle=10, Train=10),
    points_loader=points_loader,
    backend_args=None)

osdar23_train_pipeline = [
    points_loader,
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=osdar23_db_sampler, use_ground_plane=False),
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
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

osdar23_train_dataset = dict(
    type=osdar23_dataset_type,
    data_root=osdar23_data_root,
    ann_file='kitti_infos_train.pkl',
    data_prefix=dict(pts='points'),
    pipeline=osdar23_train_pipeline,
    modality=input_modality,
    test_mode=False,
    metainfo=metainfo,
    backend_args=None)

class_balanced_osdar23_train_dataset = dict(
    type='ClassBalancedDataset',
    dataset=osdar23_train_dataset,
    oversample_thr=0.1
)

repeat_osdar23_train_dataset = dict(
    type='RepeatDataset',
    times=2,
    dataset=class_balanced_osdar23_train_dataset)

osdar23_val_dataset = dict(
    type=osdar23_dataset_type,
    data_root=osdar23_data_root,
    ann_file='kitti_infos_val.pkl',
    data_prefix=dict(pts='points'),
    pipeline=generic_eval_pipeline,
    modality=input_modality,
    test_mode=True,
    metainfo=metainfo,
    backend_args=None)

############# Robosense Specific Config #############

robosense_dataroot = 'data/robosense_cls/'
robosense_dataset_type = 'ROBOSENSE_M1_PLUS'

robosense_db_sampler = dict(
    data_root=robosense_dataroot,
    info_path=robosense_dataroot + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(Pedestrian=20, Cyclist=20, RoadVehicle=20, Train=20)
    ),
    classes=class_names,
    sample_groups=dict(Pedestrian=10, Cyclist=10, RoadVehicle=10, Train=10),
    points_loader=points_loader,
    backend_args=None)
    
robosense_train_pipeline = [
    points_loader,
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=robosense_db_sampler),
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

robosense_train_dataset = dict(
    type=robosense_dataset_type,
    data_root=robosense_dataroot,
    ann_file= 'kitti_infos_train.pkl',
    data_prefix=dict(pts='points'),
    pipeline=robosense_train_pipeline,
    modality=input_modality,
    test_mode=False,
    metainfo=metainfo,
    backend_args=None)

robosense_val_dataset = dict(
    type=robosense_dataset_type,
    data_root=robosense_dataroot,
    data_prefix=dict(pts='points'),
    ann_file='kitti_infos_val.pkl',
    pipeline=generic_eval_pipeline,
    modality=input_modality,
    test_mode=True,
    metainfo=metainfo,
    backend_args=None)

robosense_test_dataset = robosense_val_dataset

robosense_repeat_dataset = dict(
    type='RepeatDataset',
    times=1,
    dataset=robosense_train_dataset)

############# Dataloader Config #############
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        shuffle=True,
        datasets=[kitti_repeat_dataset, repeat_osdar23_train_dataset]
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        shuffle=True,
        datasets=[kitti_val_dataset, osdar23_val_dataset]
    )
)

# For the val and test evaluator, we do not need to specify the annotation files
val_evaluator = dict(
    type='General_3dDet_Metric_MMLab',
    metric='det3d',
    classes=class_names,
    pcd_limit_range=point_cloud_range_inference,
    output_dir='/home/cws-ml-lab/mmdetection3d_for_rail/checkpoints/rtx4090_cp_run8_mix_kitti_osdar23_4class_80m',
    save_evaluation_results = True,
    save_random_viz = False,
    random_viz_keys = None)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# No visualizer
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

repeat_osdar23_train_dataset

############# Model config #############

voxel_size = [0.1, 0.1, 0.2]
post_center_range = point_cloud_range

model = dict(
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_voxels=(90000, 120000))),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 800, 800],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['RoadVehicle']),
            dict(num_class=1, class_names=['Train'])
        ],
        common_heads=dict(
            # reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=post_center_range,
            max_num=100,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[800, 800, 40], #41, 800, 752
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            post_center_limit_range=post_center_range,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096, # TODO: These are ways higher in Yaziwel's implementation
            post_max_size=512,  # TODO: These are ways higher in Yaziwel's implementation
            nms_thr=0.2)))

############# Scheduler #############

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 20. Please change the interval accordingly if you do not
# use a default schedule.
# optimizer
lr = 1e-4
# This schedule is mainly used by models on nuScenes dataset
# max_norm=10 is better for SECOND

epoch_num = 40

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=0.4 * epoch_num,
        eta_min=lr * 10,
        begin=0,
        end=0.4 * epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=0.6 * epoch_num,
        eta_min=lr * 1e-4,
        begin=0.4 * epoch_num,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=0.4 * epoch_num,
        eta_min=0.85 / 0.95,
        begin=0,
        end=0.4 * epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=0.6 * epoch_num,
        eta_min=1,
        begin=0.4 * epoch_num,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=32)

############# Default Runtime Config #############

default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=4),
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
work_dir = '/home/cws-ml-lab/mmdetection3d_for_rail/checkpoints/rtx4090_cp_run8_mix_kitti_osdar23_4class_80m'