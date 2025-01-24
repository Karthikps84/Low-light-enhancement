default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=50),
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
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=5,
    dynamic_intervals=[
        (
            30,
            1,
        ),
    ])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=2e-05,
        begin=25,
        end=50,
        T_max=25,
        by_epoch=True,
        convert_to_iter_based=True),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
auto_scale_lr = dict(enable=True, base_batch_size=32)
dataset_type = 'CocoDataset'
data_root = ''
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='CachedMosaic',
        img_scale=(
            640,
            640,
        ),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(
            1280,
            1280,
        ),
        ratio_range=(
            0.5,
            2.0,
        ),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(
        640,
        640,
    )),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(
        640,
        640,
    ), pad_val=dict(img=(
        114,
        114,
        114,
    ))),
    dict(
        type='CachedMixUp',
        img_scale=(
            640,
            640,
        ),
        ratio_range=(
            1.0,
            1.0,
        ),
        max_cached_images=10,
        random_pop=False,
        pad_val=(
            114,
            114,
            114,
        ),
        prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(
        1,
        1,
    )),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(
        640,
        640,
    ), keep_ratio=True),
    dict(type='Pad', size=(
        640,
        640,
    ), pad_val=dict(img=(
        114,
        114,
        114,
    ))),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True,
        poly2mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'gt_maskscale_factor',
        )),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='LIS/annotations/lis_coco_JPG_train+1.json',
        data_prefix=dict(img='LIS/RGB_dark'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='CachedMosaic',
                img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                max_cached_images=20,
                random_pop=False),
            dict(
                type='RandomResize',
                scale=(
                    1280,
                    1280,
                ),
                ratio_range=(
                    0.5,
                    2.0,
                ),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(
                640,
                640,
            )),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad',
                size=(
                    640,
                    640,
                ),
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                ))),
            dict(
                type='CachedMixUp',
                img_scale=(
                    640,
                    640,
                ),
                ratio_range=(
                    1.0,
                    1.0,
                ),
                max_cached_images=10,
                random_pop=False,
                pad_val=(
                    114,
                    114,
                    114,
                ),
                prob=0.5),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(
                1,
                1,
            )),
            dict(type='PackDetInputs'),
        ],
        metainfo=dict(
            classes=(
                'bicycle',
                'car',
                'motorcycle',
                'bus',
                'bottle',
                'chair',
                'dining table',
                'tv',
            ))),
    pin_memory=True)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='LIS/annotations/lis_coco_JPG_test+1.json',
        data_prefix=dict(img='LIS/RGB_dark'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                640,
                640,
            ), keep_ratio=True),
            dict(
                type='Pad',
                size=(
                    640,
                    640,
                ),
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                ))),
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
                'bicycle',
                'car',
                'motorcycle',
                'bus',
                'bottle',
                'chair',
                'dining table',
                'tv',
            ))))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='',
        ann_file='LIS/annotations/lis_coco_JPG_test+1.json',
        data_prefix=dict(img='LIS/RGB_dark'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(
                640,
                640,
            ), keep_ratio=True),
            dict(
                type='Pad',
                size=(
                    640,
                    640,
                ),
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                ))),
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
                'bicycle',
                'car',
                'motorcycle',
                'bus',
                'bottle',
                'chair',
                'dining table',
                'tv',
            ))))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='LIS/annotations/lis_coco_JPG_test+1.json',
    metric=[
        'bbox',
        'segm',
    ],
    format_only=False,
    proposal_nums=(
        100,
        1,
        10,
    ))
test_evaluator = dict(
    type='CocoMetric',
    ann_file='LIS/annotations/lis_coco_JPG_test+1.json',
    metric=[
        'bbox',
        'segm',
    ],
    format_only=False,
    proposal_nums=(
        100,
        1,
        10,
    ))
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale=(
                    640,
                    640,
                ), keep_ratio=True),
                dict(type='Resize', scale=(
                    320,
                    320,
                ), keep_ratio=True),
                dict(type='Resize', scale=(
                    960,
                    960,
                ), keep_ratio=True),
            ],
            [
                dict(type='RandomFlip', prob=1.0),
                dict(type='RandomFlip', prob=0.0),
            ],
            [
                dict(
                    type='Pad',
                    size=(
                        960,
                        960,
                    ),
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    ))),
            ],
            [
                dict(
                    type='PackDetInputs',
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    )),
            ],
        ]),
]
model = dict(
    type='LLRTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
        )),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[
            96,
            192,
            384,
        ],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=8,
        in_channels=96,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=96,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[
                8,
                16,
                32,
            ]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-06, reduction='mean')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5),
    enhancer=dict(type='TorchAdapt', number_f=32, scale_factor=3.0))
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(
            640,
            640,
        ),
        ratio_range=(
            0.5,
            2.0,
        ),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(
            640,
            640,
        ),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(
        1,
        1,
    )),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(
        640,
        640,
    ), pad_val=dict(img=(
        114,
        114,
        114,
    ))),
    dict(type='PackDetInputs'),
]
max_epochs = 50
stage2_num_epochs = 20
base_lr = 0.0004
interval = 5
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=30,
        switch_pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(
                type='RandomResize',
                scale=(
                    640,
                    640,
                ),
                ratio_range=(
                    0.5,
                    2.0,
                ),
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(
                    640,
                    640,
                ),
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(
                1,
                1,
            )),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Pad',
                size=(
                    640,
                    640,
                ),
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                ))),
            dict(type='PackDetInputs'),
        ]),
]
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
classes = (
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'bottle',
    'chair',
    'dining table',
    'tv',
)
launcher = 'none'
work_dir = './work_dirs/lis-rtmdet-ins_tiny'
