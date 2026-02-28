import cv2
import math

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.nn.functional
from models.backbone import *
from libs.utils import *

import matplotlib.pyplot as plt
import torch
#条状池化模块(在 feat_squeeze 之后、feat_combine 之前加入 StripPooling)
def visualize_feature_map(feat, save_dir=None, prefix="feature_map", num_channels=6):
    """
    Visualizes the feature map by displaying the first few channels (num_channels).
    :param feat: Feature map tensor, should be in the shape [batch, channels, height, width]
    :param save_dir: Directory to save the image (optional)
    :param prefix: Prefix for the saved image filename (optional)
    :param num_channels: Number of channels to display (default 6)
    """
    batch_size, channels, height, width = feat.shape

    # Limit the number of channels to visualize
    num_channels = min(channels, num_channels)     

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for b in range(batch_size):
        
        
        if channels == 1:
            
            # If there is only one channel, no need to loop over channels
            feature = feat[b, 0, :, :].cpu().detach().numpy()  # Only the first channel
            plt.figure(figsize=(8, 8))
            plt.imshow(feature, cmap='jet')  # Using 'jet' colormap
            plt.axis('off')
            
            if save_dir:
                save_path = os.path.join(save_dir, f"{prefix}_batch{b}_channel0.png")
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        # Create a grid of feature maps
        else:
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
            for c in range(num_channels):
                feature = feat[b, c, :, :].cpu().detach().numpy()
                axes[c].imshow(feature, cmap='jet')
                axes[c].axis('off')  # Hide axes
                axes[c].set_title(f"Channel {c}")
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/{prefix}_batch{b}.png")
            else:
                plt.show()
            plt.close()
            
        
            
            
class conv_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class conv_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(conv_bn_relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv1d_bn_relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(conv1d_bn_relu, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']

        self.c_feat = 64
        
    
        
        self.strip_pooling = StripPooling(self.c_feat * 3)#####
        # model
        self.encoder = resnet(layers=self.cfg.backbone, pretrained=True)#####
        backbone = self.cfg.backbone

        self.feat_squeeze1 = torch.nn.Sequential(
            conv_bn_relu(128, 64, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(64, self.c_feat, 3, padding=1),
        )
        self.feat_squeeze2 = torch.nn.Sequential(
            conv_bn_relu(256, 64, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(1024, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(64, self.c_feat, 3, padding=1),
        )
        self.feat_squeeze3 = torch.nn.Sequential(
            conv_bn_relu(512, 64, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] else conv_bn_relu(2048, 128, kernel_size=3, stride=1, padding=1),
            conv_bn_relu(64, self.c_feat, 3, padding=1),
        )

        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(self.c_feat * 3, 64, 3, padding=1, dilation=1),
            torch.nn.Conv2d(64, self.c_feat, 1),
        )

        self.feat_refine = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, padding=1, dilation=1)
        )

        self.classifier = torch.nn.Sequential(
            conv_bn_relu(self.c_feat, self.c_feat, 3, stride=1, padding=1),
            torch.nn.Conv2d(self.c_feat, 2, 1)
        )

    def forward_for_encoding(self, img):
        # Feature extraction
        feat1, feat2, feat3 = self.encoder(img)
        #feat1.shape = torch.Size([32, 128, 48, 80])
        #feat2.shape = torch.Size([32, 256, 24, 40])
        #feat3.shape = torch.Size([32, 512, 12, 20])
        
        
    # # 可视化编码后的特征图
    #     visualize_feature_map(feat1, save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='feat1')#####
    #     visualize_feature_map(feat2, save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='feat2')
    #     visualize_feature_map(feat3, save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='feat3')#####
        
        
        
        self.feat = dict()
        self.feat[self.sf[0]] = feat1
        self.feat[self.sf[1]] = feat2
        self.feat[self.sf[2]] = feat3
        
        return self.feat
    
    def forward_for_squeeze(self):
        # Feature squeeze and concat
        x1 = self.feat_squeeze1(self.feat[self.sf[0]])
        x2 = self.feat_squeeze2(self.feat[self.sf[1]])
        x2 = torch.nn.functional.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.feat_squeeze3(self.feat[self.sf[2]])
        x3 = torch.nn.functional.interpolate(x3, scale_factor=4, mode='bilinear')
        
        #         # 可视化每个 squeeze 层的输出
        # visualize_feature_map(x1, save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='squeeze1')#####
        # visualize_feature_map(x2, save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='squeeze2')
        # visualize_feature_map(x3, save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='squeeze3')#####
        
        
        
        x_concat = torch.cat([x1, x2, x3], dim=1)
        
        x_concat = self.strip_pooling(x_concat)######
        
        x4 = self.feat_combine(x_concat)
        x4 = torch.nn.functional.interpolate(x4, scale_factor=2, mode='bilinear')
        self.img_feat = self.feat_refine(x4)
        
        return self.img_feat

    
    def forward_for_classification(self):
        out = self.classifier(self.img_feat)
        
        
        # visualize_feature_map(out, save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='img_feat')#####
        
        self.prob_map = F.softmax(out, dim=1)
        
        
        # visualize_feature_map(self.prob_map[:, 1:2], save_dir='/train1/gzp/VIL-100/RVLD-main/Modeling/VIL-100/ILD_seg/code/models/feature_maps', prefix='prob_map')#####
        return {'seg_map_logit': out,
                'seg_map': self.prob_map[:, 1:2]}


import torch
import torch.nn as nn
from torch.nn import functional as F


class StripPooling(nn.Module):
    def __init__(self, in_channels, pool_size=(20, 12), norm_layer=nn.BatchNorm2d):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)
