_base_ = [
    '../_base_/models/pointpillars_hv_secfpn_osdar23.py',
    '../_base_/datasets/osdar23-3d.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

dataset = dict(type='OSDaR23Dataset')
point_cloud_range =  [0, -39.68, -3, 101.12, 39.68, 1]
# point_cloud_range = [0, -43.2, -3, 99.20, 42.2, 1]
# dataset settings
data_root = 'data/osdar23/'
class_names = ['pedestrian', 'car', 'train', 'bike', 'unknown', 'dontcare']
metainfo = dict(classes=class_names)
backend_args = None

# PointPillars adopted a different sampling strategies among classes
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_min_points=dict(pedestrian=5, car=5, train=5, bike=5, unknown=5, dontcare=5)
        filter_by_min_points=dict(pedestrian=5, car=5, train=0, bike=0, unknown=0, dontcare=0)
        ),
    classes=class_names,
    sample_groups=dict(pedestrian=5, car=5, train=0, bike=0, unknown=0, dontcare=0),
    sample_groups=dict(pedestrian=5, car=5, train=5, bike=5, unknown=5, dontcare=5),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
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
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs', 
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]

train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline, metainfo=metainfo)))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, metainfo=metainfo))
# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
epoch_num = 100
optim_wrapper = dict(
    optimizer=dict(lr=lr), clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
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
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# custom_hooks = [
#     dict(type='WandbLoggerHook', 
#          save_dir='data/osdar23/training/pointpillars_test_delete_after_use/',
#          yaml_config_path='wandb_auth.yaml',
#          log_artifact=True,
#          init_kwargs={
#              'project': 'testestest',
#              'entity': 'railsensing'
#              })
# ]
custom_hooks = [
    dict(type='WandbLoggerHook', 
         save_dir='data/osdar23/training/rtx4k_pointpillars_run1/',
         yaml_config_path='wandb_auth.yaml',
         log_artifact=True,
         init_kwargs={
             'entity': 'railsensing',
             'project': 'mmdetection3d',
             'name': 'rtx4k_pointpillars_run1'
             })
]
