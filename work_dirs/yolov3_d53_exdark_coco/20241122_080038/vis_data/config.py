train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=273, val_interval=7)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
    dict(
        type='MultiStepLR', by_epoch=True, milestones=[
            218,
            246,
        ], gamma=0.1),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))
auto_scale_lr = dict(enable=True, base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=7),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[
        0,
        0,
        0,
    ],
    std=[
        255.0,
        255.0,
        255.0,
    ],
    bgr_to_rgb=True,
    pad_size_divisor=32)
model = dict(
    type='LLYOLOV3',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[
            0,
            0,
            0,
        ],
        std=[
            255.0,
            255.0,
            255.0,
        ],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(
            3,
            4,
            5,
        ),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[
            1024,
            512,
            256,
        ],
        out_channels=[
            512,
            256,
            128,
        ]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=12,
        in_channels=[
            512,
            256,
            128,
        ],
        out_channels=[
            1024,
            512,
            256,
        ],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[
                [
                    (
                        116,
                        90,
                    ),
                    (
                        156,
                        198,
                    ),
                    (
                        373,
                        326,
                    ),
                ],
                [
                    (
                        30,
                        61,
                    ),
                    (
                        62,
                        45,
                    ),
                    (
                        59,
                        119,
                    ),
                ],
                [
                    (
                        10,
                        13,
                    ),
                    (
                        16,
                        30,
                    ),
                    (
                        33,
                        23,
                    ),
                ],
            ],
            strides=[
                32,
                16,
                8,
            ]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[
            32,
            16,
            8,
        ],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100),
    enhancer=dict(
        type='TorchAdapt',
        number_f=32,
        scale_factor=3.0,
        already_normalized=True))
dataset_type = 'CocoDataset'
data_root = ''
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=[
        0,
        0,
        0,
    ], to_rgb=True, ratio_range=(
        1,
        2,
    )),
    dict(
        type='MinIoURandomCrop',
        min_ious=(
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        ),
        min_crop_size=0.3),
    dict(
        type='RandomResize',
        scale=[
            (
                320,
                320,
            ),
            (
                608,
                608,
            ),
        ],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(
        608,
        608,
    ), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='train_coco_exdark_annotations.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Expand',
                mean=[
                    0,
                    0,
                    0,
                ],
                to_rgb=True,
                ratio_range=(
                    1,
                    2,
                )),
            dict(
                type='MinIoURandomCrop',
                min_ious=(
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                ),
                min_crop_size=0.3),
            dict(
                type='RandomResize',
                scale=[
                    (
                        320,
                        320,
                    ),
                    (
                        608,
                        608,
                    ),
                ],
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackDetInputs'),
        ],
        metainfo=dict(
            classes=(
                'person',
                'bicycle',
                'car',
                'motorcycle',
                'bus',
                'boat',
                'cat',
                'dog',
                'bottle',
                'cup',
                'chair',
                'dining table',
            ))))
val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='val_coco_exdark_annotations.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(
                608,
                608,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        metainfo=dict(
            classes=(
                'person',
                'bicycle',
                'car',
                'motorcycle',
                'bus',
                'boat',
                'cat',
                'dog',
                'bottle',
                'cup',
                'chair',
                'dining table',
            ))))
test_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='val_coco_exdark_annotations.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='Resize', scale=(
                608,
                608,
            ), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        metainfo=dict(
            classes=(
                'person',
                'bicycle',
                'car',
                'motorcycle',
                'bus',
                'boat',
                'cat',
                'dog',
                'bottle',
                'cup',
                'chair',
                'dining table',
            ))))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='val_coco_exdark_annotations.json',
    metric='bbox')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='val_coco_exdark_annotations.json',
    metric='bbox')
backend_args = None
classes = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'boat',
    'cat',
    'dog',
    'bottle',
    'cup',
    'chair',
    'dining table',
)
launcher = 'none'
work_dir = './work_dirs/yolov3_d53_exdark_coco'
