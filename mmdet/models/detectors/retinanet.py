# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

from typing import List, Tuple, Union
from mmdet.structures import OptSampleList, SampleList
from torch import Tensor
import torch


@MODELS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

@MODELS.register_module()
class LLRetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 enhancer: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            enhancer=enhancer,
            init_cfg=init_cfg)

        # self.enhancement_module = UEAttention2()
        #print(self.enhancement_module)
        # model = torch.load('/netscratch/kallempudi/thesis/checkpoints/byol_feat_Resent_synimagenet_imagenet/epoch_50.pth')
        # for x in model['state_dict'].keys():
        #     if 'enhanced_module' not in x:
        #         model['state_dict'].pop(x, None)

        # self.enhancement_module.load_state_dict(model['state_dict'], strict = False)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if self.with_enhancer:
            # print(self.enhancer)
            batch_inputs, _ = self.enhancer(batch_inputs)

        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
