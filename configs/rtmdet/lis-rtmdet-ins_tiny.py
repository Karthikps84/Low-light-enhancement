_base_ = './rtmdet-ins_s_8xb32-300e_coco.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    type = 'LLRTMDet',
    enhancer=dict(
        type='TorchAdapt',
        number_f=32,
        scale_factor=3.0),
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375, init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(num_classes=8, in_channels=96, feat_channels=96))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=10,
        random_pop=False,
        pad_val=(114, 114, 114),
        prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True,poly2mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'gt_mask'
                   'scale_factor'))
]
# dataset settings
backend_args = None
classes = ('bicycle', 'car', 'motorbike', 'bus','bottle', 'chair', 'diningtable', 'tvmonitor')

dataset_type = 'CocoDataset'
data_root= ''

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='LIS/annotations/lis_coco_JPG_train+1.json',
        data_prefix=dict(img='LIS/RGB_dark'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='LIS/annotations/lis_coco_JPG_test+1.json',
        data_prefix=dict(img='LIS/RGB_dark'),
        metainfo=dict(classes=classes)
        ))
test_dataloader = val_dataloader


val_evaluator = dict(
    ann_file=data_root +'LIS/annotations/lis_coco_JPG_test+1.json',
    metric=['bbox', 'segm'],
    proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

max_epochs = 50
stage2_num_epochs = 20
base_lr = 0.0004
interval = 5

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=50
    ))

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)])


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

auto_scale_lr = dict(enable=True, base_batch_size=32)
