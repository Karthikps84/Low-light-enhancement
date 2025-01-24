import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np
import math




class CrossAttnetion(nn.Module):
    def __init__(self, channels, query_image_size, key_image_size):
        super().__init__()

        query_stride = query_image_size // 79
        key_stride = key_image_size // 79
        key_kernel = key_stride
        if key_stride > 1:
            key_kernel = 3

        self.num_temporal_attention_blocks = 8
        if self.num_temporal_attention_blocks > 0:
            self.query_conv1 = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size= 7, stride=4)
            self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
            # self.key_conv = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size=key_kernel, stride=key_stride)
            self.key_conv = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size=3, stride=2)


    def forward(self, x, ref_x):
        """ Aggregate the X7 features`x` with the Qurter or Hexa features : ref_x
                `ref_x`.
                The aggregation mainly contains three steps:
                1. Pass through a tiny embed network.
                2. Use multi-head attention to computing the weight between `x` and
                `ref_x`.
                3. Use the normlized (i.e. softmax) weight to weightedly sum `x` and
                `ref_x`.

                Returns:
                    Tensor: The aggregated features with shape [roi_n, C, roi_h, roi_w].
        """
        #Saving x to add as a residual at the end
        orig_x = x
        x = self.query_conv1(x)
        x = self.query_conv2(x)
        ref_x = self.key_conv(ref_x)
        # print(f"Shapes in beging of CA X : {x.shape}  REF_X : {ref_x.shape}")
        batch_size, C, roi_h, roi_w = x.size()
        # (btach_size, img_n, C, H, W)
        x = x.view(batch_size, 1, C, roi_h, roi_w)
        ref_x = ref_x.view(batch_size, 1, C, roi_h, roi_w)

        x = torch.cat((x, ref_x), dim=1)
        batch_size, img_n, _, roi_h, roi_w = x.size()
        # print(f"Shapes after cat X : {x.shape}")

        num_attention_blocks = self.num_temporal_attention_blocks

        # 1. Pass through a tiny embed network
        # (img_n * roi_n, C, H, W)
        x_embed = x
        c_embed = x_embed.size(2)
        # 2. Perform multi-head attention
        # (img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        x_embed = x_embed.view(batch_size, img_n, num_attention_blocks, -1, roi_h,
                               roi_w)
        # (1, roi_n, num_attention_blocks, C / num_attention_blocks, H, W)
        target_x_embed = x_embed[:, [1]]
        # (batch_size, img_n, num_attention_blocks, 1, H, W)
        ada_weights = torch.sum(
            x_embed * target_x_embed, dim=3, keepdim=True) / (
                float(c_embed / num_attention_blocks)**0.5)
        # (batch_size, img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        ada_weights = ada_weights.expand(-1, -1, -1,
                                         int(c_embed / num_attention_blocks),
                                         -1, -1).contiguous()
        # (img_n, roi_n, C, H, W)
        ada_weights = ada_weights.view(batch_size, img_n, c_embed, roi_h, roi_w)
        ada_weights = ada_weights.softmax(dim=1)

        # print(f"Shapes before aggregation expand ada_weights : {ada_weights.shape} and X shape : {x.shape}")
        # 3. Aggregation
        x = (x * ada_weights).sum(dim=1)

        upsample = nn.UpsamplingBilinear2d((orig_x.size()[-2], orig_x.size()[-1]))
        out = upsample(x)
        out = orig_x + out
        return out

class UEAttention(nn.Module):
    # Incorporating Cross attention to merge Quarter scale and orginal features
    # Then employing Skip connection to merge enhanced features with hexa scale features
    # Incorporating Cross attention to merge Quarter scale and orginal features
    # Then employing Skip connection to merge enhanced features with hexa scale features
    def __init__(self, in_channels):
        super(UEAttention, self).__init__()

        int_out_channels = 32
        out_channels = 24

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)
        # our convolution layers to transform the concatenated feature maps into the required feature shapes
        self.ue_conv8 = nn.Conv2d(out_channels*2, int_out_channels, 3, 1, 1, bias=True)
        self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)

        #Convolutions for downsampling orignal image into multiple scales
        self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
        self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)


    def forward(self, x):

        quarter_scale_x = self.quarter_conv(x)
        hexa_scale_x = self.hexa_conv(quarter_scale_x)
        # print(f" quarter_scale_x: ·{quarter_scale_x.shape} hexa_scale SHAPE AFTER SECOND MAXPOOL : ·{hexa_scale_x.shape}")

        x1 = self.relu(self.e_conv1(x))
        quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x))
        hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))

        x2 = self.relu(self.e_conv2(x1))
        quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1))
        hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))

        x3 = self.relu(self.e_conv3(x2))
        quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2))
        hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))

        x4 = self.relu(self.e_conv4(x3))
        quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3))
        hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1)))
        hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))

        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1)))
        hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))

        x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1)))
        quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1)))
        hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))

        #APPLYING CROSS ATTTENTION BETWEEN X7 AND QUARTER SCALE FIRST then SC between Hexa and X7
        x7 = self.cross_attention_block(x7, quarter_scale_x7)

        #Now Upsampling hexa scale to make all them equal to x in H x W for SC
        x_upsample = nn.UpsamplingBilinear2d((x7.size()[-2], x7.size()[-1]))
        hexa_scale_x7 = x_upsample(hexa_scale_x7)
        # x8 = self.cross_attention_block(x7, hexa_scale_x7)
        x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))

        #Downsampling from 32 to 24, 32 features transformation helps
        x_r = torch.tanh(self.ue_conv9(x8))

        # Reconstrucing image, code identical to Zero-DCE
        # r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        # x = x + r1 * (torch.pow(x, 2) - x)
        # x = x + r2 * (torch.pow(x, 2) - x)
        # x = x + r3 * (torch.pow(x, 2) - x)
        # enhanced_image_after_fourth_curve = x + r4 * (torch.pow(x, 2) - x)
        # x = enhanced_image_after_fourth_curve + r5 * (torch.pow(enhanced_image_after_fourth_curve, 2) - enhanced_image_after_fourth_curve)
        # x = x + r6 * (torch.pow(x, 2) - x)
        # x = x + r7 * (torch.pow(x, 2) - x)
        # enhanced_image_final = x + r8 * (torch.pow(x, 2) - x)
        # concatenated_learned_curves = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)

        # Reconstrucing image, code identical to Zero-DCE
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + (r1**4)
        x = x + (r2**4)
        x = x + (r3**4)
        x = x + (r4**4)
        x = x + (r5**4)
        x = x + (r6**4)
        x = x + (r7**4)
        enhanced_image_final = x +   (r8**4)

        return enhanced_image_final

    def cross_attention_block(self, x, ref_x):
        #Just passing Standard Convolution Feature maps
        # Projections For Q,K, and V are convolutional layers

        cross_att = CrossAttnetion(channels=x.size()[1], query_image_size=x.size()[2], key_image_size=ref_x.size()[2]).to('cuda')
        out = cross_att(x, ref_x)
        # print(f"quarter_scale_x7 shape after ATTENTION: ·{x.shape}")
        return out


