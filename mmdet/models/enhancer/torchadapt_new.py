import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import *

from mmdet.registry import MODELS
from mmengine.model import BaseModule
from typing import List, Optional, Tuple, Union

from ...utils.contextmanagers import logger


class Adaptor(nn.Module):
    def __init__(self):
        super(Adaptor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Added dropout for regularization, need to verify
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AdaptorWithMatrix(nn.Module):
    '''
        A different adaptor module which generates a projection of scalar matrix
        instead of a sinlge adaptiveness score to adapt enhancement.
        This matrix will be multiplied with the enhancement to adapt torch (enhancement)
        on each images based on its content.
    '''
    def __init__(self):
        super(AdaptorWithMatrix, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added Option batch normalization, need to verify
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # Added Option batch normalization, need to verify
            nn.ReLU()
        )
        self.scalar_matrix = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.scalar_matrix(x) # Output is a matrix with values between 0 and 1


        return x

class Torch(nn.Module):
    '''
        A very simple model with a single Conv for features (32 channels)
        Followed with another conv to reconstruct enhanced image
        Different from SCI in two ways:
            - TanH instead of Sigmoid to cater both high and low light images
            - Aggregation to cater negative values impact
    '''
    def __init__(self, layers, channels):
        super(Torch, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            block = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            self.blocks.append(block)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        features = self.in_conv(x)

        for conv in self.blocks:
            features = features + conv(features)

        features = self.out_conv(features)

        enhancement = x * (1 + features)
        enhancement = torch.clamp(enhancement, 0, 1)

        return enhancement

@MODELS.register_module()
class TorchAdapt(BaseModule):
    '''
        TorchAdapt module:
        Making a single Unified Module encompassing Torch and Adaptor Modules.
        Final module now encompassing a very light Torch with 2 Convs
        And a strong Adaptor module equipped with scaling factor based on ilumination



    '''
    def __init__(self, number_f=32, scale_factor=5.0, already_normalized=False, loss_color=None, loss_exposure=None,
                 init_cfg: Optional[Union[List[dict], dict]] = None):
        super(TorchAdapt, self).__init__(init_cfg=init_cfg)


        #For Torch using SCI
        self.torch = Torch(layers=1, channels=number_f).cuda()

        # condition to use singe value or a scalar matrix
        # Now only working with scalar matrix
        self.scale_factor = scale_factor

        self.adaptor = AdaptorWithMatrix().cuda()


        self.already_normalized = already_normalized # New flag to indicate normalization status

        if not already_normalized:
            # Mean and std tensors for unnormalization (precomputed for efficiency)
            self.register_buffer('mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

        if loss_color is not None:
            self.loss_color = MODELS.build(loss_color)
        if loss_exposure is not None:
            self.loss_exposure = MODELS.build(loss_exposure)

        # print(self.torch, self.adaptor)

    def forward(self, x):
        '''
         Forward method for TorchAdapt.

         Args:
             x (torch.Tensor): Input image tensor [batch_size, 3, H, W].

         Returns:
             final_image_normalized (torch.Tensor): Output image tensor in the same normalization as input.
             enhanced_image (torch.Tensor): Output from the Torch module.
         '''

        if not self.already_normalized:
            # print("Already normalized Falise, normalizing now")
            # Input images are normalized using mean and std (e.g., ImageNet)
            # Unnormalize and normalize images to [0, 1] range
            x_unnormalized = self.unnormalize(x)  # Convert to [0, 255]
            x_normalized = self.normalize(x_unnormalized)  # Convert to [0, 1]
        else:
            # Assume images are in [0, 1] range
            x_normalized = x

        # Step 3: Compute the adaptive scale_factor based on per-image luminance
        scale_factor = self.compute_scale_factor(x_normalized)
        # print(f"\n\n Scale factor mean value for a batch: {scale_factor.mean()}")
        # print(f"x_normalized stats in TorchAdapt - min: {x_normalized.min()}, max: {x_normalized.max()}, mean: {x_normalized.mean()}, std: {x_normalized.std()}")

        # Step 4: Get enhanced image from Torch module
        enhanced_image = self.torch(x_normalized)  # x_normalized may have values outside [0, 1]

        # Step 5: Get adaptiveness map from Adaptor module
        adaptiveness = self.adaptor(x_normalized)
        # print(f"Adaptiveness stats - min: {adaptiveness.min()}, max: {adaptiveness.max()}, mean: {adaptiveness.mean()}, std: {adaptiveness.std()}")

        # Step 6: Apply the adaptive scale_factor to adaptiveness
        adaptiveness = adaptiveness * scale_factor  # Scale adaptiveness adaptively

        # Step 7: Compute weighted enhancement and final image
        weighted_enhancement = adaptiveness * enhanced_image
        final_image = x_normalized + weighted_enhancement

        if not self.already_normalized:
            # Input images are normalized using mean and std (e.g., ImageNet)
            # Only clamp when not normalized
            final_image = torch.clamp(final_image, min=0.0, max=1.0)
            # Transform back to original normalized distribution
            final_image_unnormalized = final_image * 255.0
            final_image_normalized = (final_image_unnormalized - self.mean) / self.std
        else:
            # Output image in the same range as input
            final_image = torch.clamp(final_image, min=0.0, max=1.0)
            final_image_normalized = final_image

        return final_image_normalized, enhanced_image

    def unnormalize(self, x):
        '''
        Unnormalize the input image from normalized space to [0, 255] range.

        Args:
            x (torch.Tensor): Normalized input image tensor [batch_size, 3, H, W].

        Returns:
            torch.Tensor: Unnormalized image tensor in [0, 255] range.
        '''
        x_unnormalized = x * self.std + self.mean  # x_unnormalized in [0, 255]
        x_unnormalized = torch.clamp(x_unnormalized, min=0.0, max=255.0)  # Clamp to [0, 255]

        return x_unnormalized

    def normalize(self, x_unnormalized):
        '''
        Normalize the unnormalized image to [0, 1] range for luminance calculation.

        Args:
            x_unnormalized (torch.Tensor): Unnormalized image tensor [batch_size, 3, H, W].

        Returns:
            torch.Tensor: Normalized image tensor in [0, 1] range.
        '''
        x_normalized = x_unnormalized / 255.0  # x_normalized in [0, 1]
        return x_normalized

    def compute_scale_factor_old(self, x_normalized):
        '''
        Compute the adaptive scale_factor based on per-image luminance.
        Trying to handle images outside pixel range
        computing scale_factor with old method which is not very adaptive

        Args:
            x_normalized (torch.Tensor): Image tensor with pixel values possibly outside [0, 1].

        Returns:
            torch.Tensor: Adaptive scale_factor tensor [batch_size, 1, 1, 1].
        '''
        # Ensure x_normalized is of type float32
        x_normalized = x_normalized.float()

        # Weights for RGB channels based on human perception
        r_weight, g_weight, b_weight = 0.2126, 0.7152, 0.0722

        # Compute luminance for each image in the batch
        luminance = (r_weight * x_normalized[:, 0] +
                     g_weight * x_normalized[:, 1] +
                     b_weight * x_normalized[:, 2]).mean(dim=[1, 2])  # Shape: [batch_size]

        # Define expected min and max luminance values based on your observed data
        luminance_min = x_normalized.min()  # Adjust as needed
        luminance_max = x_normalized.max()  # Adjust as needed


        # Normalize luminance to [0, 1]
        luminance_normalized = (luminance - luminance_min) / (luminance_max - luminance_min)
        luminance_normalized = torch.clamp(luminance_normalized, 0.0, 1.0)

        # Calculate the adaptive scale_factor
        max_scale_factor = self.scale_factor  # e.g., 5.0
        # Invert luminance to get higher scale_factor for darker images
        scale_factor = (1.0 - luminance_normalized).unsqueeze(1) * max_scale_factor  # Shape: [batch_size, 1]
        # Clamp scale_factor to be within [0, max_scale_factor]
        scale_factor = torch.clamp(scale_factor, min=0.0, max=max_scale_factor)
        # Reshape scale_factor to match adaptiveness dimensions
        scale_factor = scale_factor.view(-1, 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]

        return scale_factor

    def compute_scale_factor(self, x_normalized):
        '''
        Compute the adaptive scale_factor based on per-image luminance.

        Args:
            x_normalized (torch.Tensor): Image tensor in [0, 1] range [batch_size, 3, H, W].

        Returns:
            torch.Tensor: Adaptive scale_factor tensor [batch_size, 1, 1, 1].
        '''
        # Weights for RGB channels based on human perception
        r_weight, g_weight, b_weight = 0.2126, 0.7152, 0.0722

        # Compute luminance for each image in the batch
        luminance = (r_weight * x_normalized[:, 0] +
                     g_weight * x_normalized[:, 1] +
                     b_weight * x_normalized[:, 2]).mean(dim=[1, 2])  # Shape: [batch_size]

        # Parameters for sigmoid function
        k = 10.0  # Controls the steepness of the curve
        x0 = 0.3  # Luminance value at the sigmoid's midpoint
        max_scale_factor = self.scale_factor  # e.g., 3.0

        # Compute scale factor using the sigmoid function
        scale_factor = max_scale_factor / (1 + torch.exp(k * (luminance - x0)))

        # Clamp scale_factor to be within [0, max_scale_factor]
        scale_factor = torch.clamp(scale_factor, min=0.0, max=max_scale_factor)
        # Reshape scale_factor to match adaptiveness dimensions
        scale_factor = scale_factor.view(-1, 1, 1, 1)  # Shape: [batch_size, 1, 1, 1]

        return scale_factor