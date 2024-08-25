# custom_imports = dict(
#     imports=['mmdet3d.datasets.ground_truth_classification_dataset'], allow_failed_imports=False)

####### Dataset Config #######
dataset = 'GroundTruthClassificationDataset'
classes = ['Pedestrian', 'Cyclist', 'Car']

points_loader = dict(
    type='LoadPointsFromFileForClassification',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=3,
    backend_args=None)

train_pipeline = [
    points_loader,
    dict(type='LoadAnnotationsCls'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='RandomJitterPoints', jitter_std=[0.01, 0.01, 0.01], clip_range=[-0.05, 0.05]),
    dict(type='PointShuffle'),
    dict(type='PackClsInputs')
]

val_pipeline = [
    points_loader,
    dict(type='LoadAnnotationsCls'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.0),
    dict(type='PackClsInputs')
]    

data_root = 'data/osdar23_3class/'

train_dataset = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_train.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    min_num_pts=256,
    pipeline=train_pipeline)

val_dataset = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_val.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    min_num_pts=256,
    pipeline=val_pipeline)

val_evaluator = dict(
    type='PointCloudClsMetric',
    class_names=classes
)
    

############# Dataloader Config #############

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    collate_fn=dict(type='pseudo_collate'),
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        datasets=[train_dataset]
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    collate_fn=dict(type='pseudo_collate'),
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        datasets=[val_dataset]
    )
)

####### Model Config #######

# model settings
model = dict(
    type='EncoderCls3D',
    data_preprocessor=dict(
        type='Cls3DDataPreprocessor',
        normalize='True',
        downsample='True',
        num_pts_downsample=256),    
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=3,  # [xyz, rgb], should be modified with dataset
        num_points=(256, 128, 1),
        radius=(0.1, 0.2, 0.6),
        num_samples=(32, 64, 64),
        sa_channels=((64, 64, 128), (128, 128, 256), (256, 512, 1024)),
        fp_channels=(),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    cls_head=dict(
        type='PointNet2ClsHead',
        num_classes=3,  # should be modified with dataset
        lin_layers=((1024, 512), (512, 256)),
        dropout_ratio=0.4),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

#### Schedule Config ####

_lr = 0.001
num_epochs = 50

# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=_lr, weight_decay=0.001),
    clip_grad=None)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=num_epochs*1.0,
        eta_min=1e-5,
        by_epoch=True,
        begin=0,
        end=num_epochs*1.0)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=num_epochs, val_interval=1)
val_cfg = dict()
# test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (2 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

######### Default Runtime Config #############

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
work_dir = 'tbd'
