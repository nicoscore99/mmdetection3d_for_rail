# custom_imports = dict(
#     imports=['mmdet3d.datasets.robosense_m1_plus_dataset',
#             'mmdet3d.apis.inferencers.lidar_classification_inferencers',
#              'mmdet3d.models.data_preprocessors.cls_data_preprocessor',
#              'mmdet3d.models.backbones.pointnetpp_sa_ssg_torch_impl',
#                 'mmdet3d.models.decode_heads.pointnet2_cls_ssg',
#                 'mmdet3d.models.segmentors.encoder_classifer'],
#     allow_failed_imports=False)

custom_imports = dict(imports=['mmdet3d.datasets.robosense_m1_plus_dataset', 
                               'mmdet3d.engine.hooks.wandb_logger_hook',
                               'mmdet3d.evaluation.metrics.general_3ddet_metric_mmlab_classification_version'], allow_failed_imports=False)             

# from mmdet3d.utils import register_all_modules
# register_all_modules()


#### Generic loader information ####

class_names = ['Pedestrian', 'Cyclist', 'RoadVehicle', 'Train']
point_cloud_range = [0, -40, -3, 80, 40, 10]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)

points_loader = dict(
    type='LoadPointsFromFile',
    coord_type='LIDAR',
    load_dim=4,
    use_dim=4,
    backend_args=None)

generic_eval_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
    dict(type='ClusterPointCloud'),
    dict(type='Open3DBallPivotingSequence',
         min_points=50,
         num_pts_sample=256),
]

############# Robosense Specific Config #############

robosense_dataroot = 'data/robosense_cls'
robosense_dataset_type = 'ROBOSENSE_M1_PLUS'
robosense_m1_plus_dataset = dict(type='ROBOSENSE_M1_PLUS')

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

############# Dataloader Config #############

val_dataloader = dict(
    batch_size=1, # Batch size has to be 1 for in order to not create problems with the evaluation
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=robosense_val_dataset
)

test_dataloader = val_dataloader    

# For the val and test evaluator, we do not need to specify the annotation files
val_evaluator = dict(
    type='General_3dDet_Metric_MMLab_Classification_Version',
    metric='det3d',
    classes=class_names,
    pcd_limit_range=point_cloud_range,
    output_dir='/home/cws-ml-lab/mmdetection3d_for_rail/checkpoints/rtx4090_pp_run12_robosense_m1_plus_sequences/evaluation',
    save_graphics = False,
    save_evaluation_results = False,
    save_random_viz = False,
    random_viz_keys = 5)

test_evaluator = val_evaluator

############# Model Config #############

model = dict(
    type='EncoderCls3D',
    data_preprocessor=dict(
        type='Cls3DDataPreprocessorEvaluation',
        normalize_mean='True',
        normalize_size='False',
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
        ),
    cls_head=dict(
        type='PointNet2ClsHead',
        num_classes=4,  # should be modified with dataset
        lin_layers=((1024, 512), (512, 256)),
        dropout_ratio=0.4),
    # model training and testing settings
    num_pts_sample=256,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

### runtime settings ###
val_cfg = dict()
test_cfg = dict()

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
work_dir = 'test_cls'
