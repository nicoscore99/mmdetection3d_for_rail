custom_imports = dict(
    imports=['mmdet3d.apis.inferencers.lidar_classification_inferencers',
             'mmdet3d.models.data_preprocessors.cls_data_preprocessor',
             'mmdet3d.models.segmentors.encoder_classifier'],
    allow_failed_imports=False)

from mmdet3d.models.data_preprocessors import Cls3DDataPreprocessor

# custom_imports = dict(
#     imports=['mmdet3d.apis.inferencers.lidar_classification_inferencers',
#              'mmdet3d.models.data_preprocessors.cls_data_preprocessor',
#              'mmdet3d.models.backbones.pointnetpp_sa_ssg_torch_impl',
#                 'mmdet3d.models.decode_heads.pointnet2_cls_ssg',
#                 'mmdet3d.models.segmentors.encoder_classifer'],
#     allow_failed_imports=False)


######## Additional Hooks ########

custom_hooks = [
    dict(type='WandbLoggerHook', 
         save_dir='/home/cws-ml-lab/mmdetection3d_for_rail/experiments/cluster_classification/rtx4090_pointnetpp_cls_all_data_256pts_yanx27_with_upsampling',
         log_artifact=True,
         init_kwargs={
             'entity': 'railsensing',
             'project': 'classification',
             'name': 'rtx4090_pointnetpp_cls_all_data_256pts_yanx27_with_upsampling',
        })
]

####### Dataset Config #######
dataset = 'GroundTruthClassificationDataset'
classes = ['Pedestrian', 'Cyclist', 'RoadVehicle', 'Train']

points_loader = dict(
    type='LoadPointsFromFileForClassification',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=3,
    backend_args=None)

train_pipeline = [
    points_loader,
    dict(type='LoadAnnotationsCls'),
    dict(type='Open3DBallPivoting',
         min_points=50,
         num_pts_sample=256),        
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
    dict(type='Open3DBallPivoting',
         min_points=50,
         num_pts_sample=256),     
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.0),
    dict(type='PackClsInputs')
]    

val_evaluator = dict(
    type='PointCloudClsMetric',
    class_names=classes
)

############# OSADAR23 Dataset config #############

data_root = 'data/osdar23_cls'

train_dataset_osdar = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_train.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    input_point_size=256,
    min_num_pts=256,   
    pipeline=train_pipeline)

train_dataset_classbalanced_osdar = dict(
    type='ClassBalancedDataset',
    dataset=train_dataset_osdar,
    oversample_thr=0.1,
)

val_dataset_osdar = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_val.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    input_point_size=256,
    min_num_pts=256,   
    pipeline=val_pipeline)

############# Robosense Dataset config #############

data_root = 'data/robosense_cls'

train_dataset_robo = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_train.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    input_point_size=256,
    min_num_pts=256,     
    pipeline=train_pipeline)

train_dataset_classbalanced_robo = dict(
    type='ClassBalancedDataset',
    dataset=train_dataset_robo,
    oversample_thr=0.1,
)

val_dataset_robo = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_val.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    input_point_size=256,
    min_num_pts=256,
    pipeline=val_pipeline)

############# KITTI dataset config #############

data_root = 'data/kitti_cls'

train_dataset_kitti = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_train.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    input_point_size=256,
    min_num_pts=256,
    pipeline=train_pipeline)

train_dataset_classbalanced_kitti = dict(
    type='ClassBalancedDataset',
    dataset=train_dataset_kitti,
    oversample_thr=0.1,
)

val_dataset_kitti = dict(
    type=dataset,
    data_root=data_root,
    ann_file='gt_infos_database_val.pkl',
    data_prefix=dict(pts=''),
    metainfo=dict(classes=classes),
    input_point_size=256,
    min_num_pts=256,
    pipeline=val_pipeline)

############# Dataloader Config #############

batch_size = 32

train_dataloader = dict(
    batch_size=32,
    num_workers=1,
    collate_fn=dict(type='pseudo_collate'),
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        shuffle=True,
        datasets=[train_dataset_classbalanced_robo, train_dataset_classbalanced_osdar, train_dataset_classbalanced_kitti]
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    collate_fn=dict(type='pseudo_collate'),
    persistent_workers=True,
    dataset=dict(
        type='ConcatDataset',
        shuffle=True,
        datasets=[val_dataset_robo, val_dataset_osdar, val_dataset_kitti]
    )
)

test_dataloader_generic = dict(
    batch_size=1,
    num_workers=1,
    collate_fn=dict(type='pseudo_collate'),
    persistent_workers=True,
    dataset=None
)

test_dataloader = test_dataloader_generic
test_evaluator = val_evaluator

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
        type='PointNetPPSASSG',
        in_channels=3,  # [xyz] should be modified with dataset
        num_points=(256, 128, 1),
        radius=(0.1, 0.2, 0.6),
        num_samples=(32, 64, 64),
        sa_channels=((64, 64, 128), (128, 128, 256), (256, 512, 1024)),
        fp_channels=(),
        # norm_cfg=dict(type='BN2d'),
        # sa_cfg=dict(
        #     type='PointSAModule',
        #     pool_mod='max',
        #     use_xyz=True,
        #     normalize_xyz=False)
        ),
    cls_head=dict(
        type='PointNet2ClsHead',
        num_classes=4,  # should be modified with dataset
        lin_layers=((1024, 512), (512, 256)),
        dropout_ratio=0.4),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

#### Schedule Config ####

lr = 0.0001
epoch_num = 60

# optimizer
# This schedule is mainly used on S3DIS dataset in segmentation task
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=lr, weight_decay=0.001),
    clip_grad=None)

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

# runtime settings30
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()

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
    checkpoint=dict(type='CheckpointHook', interval=4, by_epoch=True),
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
work_dir = '/home/cws-ml-lab/mmdetection3d_for_rail/experiments/cluster_classification/rtx4090_pointnetpp_cls_all_data_256pts_yanx27_with_upsampling'
