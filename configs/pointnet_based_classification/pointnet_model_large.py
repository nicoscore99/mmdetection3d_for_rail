model = dict(
    type='EncoderCls3D',
    data_preprocessor=dict(
        type='Cls3DDataPreprocessorEvaluation',
        normalize_mean='True',
        normalize_size='True',
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