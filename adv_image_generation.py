from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
import pathlib
from fnmatch import fnmatch
import matplotlib.pyplot as plt
import os
import scipy.optimize
import numpy as np
from PIL import Image

PATCH_SIZE = (16, 16)
MIN_ROI_SIZE = PATCH_SIZE[0] * PATCH_SIZE[1] * 1
SOURCE_CLASS = 'person'
TARGET_CLASS = 'chair'


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            return [x2, x10, x18]


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neek = Neck(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        
        # head
        self.head = Yolov4Head(output_ch, n_classes, inference)


    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output


def custom_bbox(gt_coords, img, imgname):
    cbbox_coords = []
    for k in range(len(gt_coords)):
        if gt_coords[k][0] == imgname:
            print(gt_coords[k][0])
            box = [float(gt_coords[k][2]), float(gt_coords[k][3]), 50, 80]
            box = torch.tensor(box)
            bbox = box_center_to_corner(box)

            x1 = int(bbox[0].item())
            y1 = int(bbox[1].item())
            x2 = int(bbox[2].item())
            y2 = int(bbox[3].item())

            coords = [x1, y1, x2, y2]
            cbbox_coords.append(coords)
                
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
    return img, cbbox_coords

# def get_iou(a, b, epsilon=1e-5):
#     """ Given two boxes `a` and `b` defined as a list of four numbers:
#             [x1,y1,x2,y2]
#         where:
#             x1,y1 represent the upper left corner
#             x2,y2 represent the lower right corner
#         It returns the Intersect of Union score for these two boxes.

#     Args:
#         a:          (list of 4 numbers) [x1,y1,x2,y2]
#         b:          (list of 4 numbers) [x1,y1,x2,y2]
#         epsilon:    (float) Small value to prevent division by zero

#     Returns:
#         (float) The Intersect of Union score.
#     """

#     iou_list = []
#     # iou = 0.0
#     bj = b[0]
#     n_iou = []



#     for i in range(len(b)):
#         iou_l=[]
#         for j in range(len(a)):
#             # bj = b[j]
#     # COORDINATES OF THE INTERSECTION BOX
#             x1 = max(a[j][0], b[i][0])
#             y1 = max(a[j][1], b[i][1])
#             x2 = min(a[j][2], b[i][2])
#             y2 = min(a[j][3], b[i][3])



#     # AREA OF OVERLAP - Area where the boxes intersect
#             width = (x2 - x1)
#             height = (y2 - y1)
#             # print(width)
#             # print(height)
#             # handle case where there is NO overlap
#             if (width<0) or (height <0):
#                 iou = 0.0
#                 iou_l.append(iou)
#                 break
#             area_overlap = width * height

#         # COMBINED AREA
#             area_a = (a[j][2] - a[j][0]) * (a[j][3] - a[j][1])
#             area_b = (b[i][2] - b[i][0]) * (b[i][3] - b[i][1])
#             area_combined = area_a + area_b - area_overlap

#             # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
#             iou_l.append(area_overlap / (area_combined+epsilon))
            

#         max_iou = max(iou_l)
#         # print(max_iou)
#         iou_list.append([b[i], round(max_iou, 3)])
            
#     return iou_list
        

# def batch_iou(a, b, epsilon=1e-5):
#     """ Given two arrays `a` and `b` where each row contains a bounding
#         box defined as a list of four numbers:
#             [x1,y1,x2,y2]
#         where:
#             x1,y1 represent the upper left corner
#             x2,y2 represent the lower right corner
#         It returns the Intersect of Union scores for each corresponding
#         pair of boxes.

#     Args:
#         a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
#         b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
#         epsilon:    (float) Small value to prevent division by zero

#     Returns:
#         (numpy array) The Intersect of Union scores for each pair of bounding
#         boxes.

#     """

#     # print(a[:, 0])
#     # # print("b")
#     # # print(b)
#     # # COORDINATES OF THE INTERSECTION BOXES
#     x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
#     y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
#     x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
#     y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

#     # AREAS OF OVERLAP - Area where the boxes intersect
#     width = (x2 - x1)
#     height = (y2 - y1)

#     # handle case where there is NO overlap
#     width[width < 0] = 0
#     height[height < 0] = 0

#     area_overlap = width * height

#     # COMBINED AREAS
#     area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
#     area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
#     area_combined = area_a + area_b - area_overlap

#     # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
#     iou = area_overlap / (area_combined + epsilon)
#     return iou

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
#   print(iou)
  return iou



def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.0):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])
    

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate((iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate((iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)


    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 


def findClosest(time, camera_time_list):
    val = min(camera_time_list, key=lambda x: abs(x - time))
    return camera_time_list.index(val)

# def extract_frames(path,file_name, model, class_names, width, height, savename, gt, device):
#     detections=0
#     gt_actual=0
#     #===== process the index files of camera 1 ======#
#     with open('/home/dissana8/LAB/Visor/cam1/index.dmp') as f:
#         content = f.readlines()
#     cam_content = [x.strip() for x in content]
#     c1_frames = []
#     c1_times = []
#     for line in cam_content:
#         s = line.split(" ")
#         frame = s[0]
#         time = float(s[1]+'.'+s[2])
#         c1_frames.append(frame)
#         c1_times.append(time)

#     with open('/home/dissana8/LAB/Visor/cam2/index.dmp') as f:
#         content = f.readlines()

#     cam_content = [x.strip() for x in content]
#     c2_frames = []
#     c2_times = []
#     for line in cam_content:
#         s = line.split(" ")
#         frame = s[0]
#         time = float(s[1]+'.'+s[2])
#         c2_frames.append(frame)
#         c2_times.append(time)
    

#     # ===== process the index files of camera 3 ======#
#     with open('/home/dissana8/LAB/Visor/cam3/index.dmp') as f:
#         content = f.readlines()

#     cam_content = [x.strip() for x in content]
#     c3_frames = []
#     c3_times = []
#     for line in cam_content:
#         s = line.split(" ")
#         frame = s[0]
#         time = float(s[1] + '.' + s[2])
#         c3_frames.append(frame)
#         c3_times.append(time)
    

#     # ===== process the index files of camera 4 ======#
#     with open('/home/dissana8/LAB/Visor/cam4/index.dmp') as f:
#         content = f.readlines()

#     cam_content = [x.strip() for x in content]
#     c4_frames = []
#     c4_times = []
#     for line in cam_content:
#         s = line.split(" ")
#         frame = s[0]
#         time = float(s[1] + '.' + s[2])
#         c4_frames.append(frame)
#         c4_times.append(time)
      
#     #===== process the GT annotations  =======#
#     with open("/home/dissana8/LAB/"+file_name) as f:
#         content = f.readlines()
        

#     content = [x.strip() for x in content]
#     counter = -1
#     print('Extracting GT annotation ...')
#     for line in content:
#         # counter += 1
#         # if counter % 150 == 0:
#         #     print(counter)
#         s = line.split(" ")
        
#         time = float(s[0])
#         frame_idx = findClosest(time, c1_times) # we have to map the time to frame number
#         c1_frame_no = c1_frames[frame_idx]
        

#         frame_idx = findClosest(time, c2_times)  # we have to map the time to frame number
#         c2_frame_no = c2_frames[frame_idx]
        

#         frame_idx = findClosest(time, c3_times)  # we have to map the time to frame number
#         c3_frame_no = c3_frames[frame_idx]

        
#         frame_idx = findClosest(time, c4_times)  # we have to map the time to frame number
#         c4_frame_no = c4_frames[frame_idx]

#         cam = []

#         cam.append('/home/dissana8/LAB/Visor/cam1/'+c1_frame_no)
#         cam.append('/home/dissana8/LAB/Visor/cam2/'+c2_frame_no)
#         cam.append('/home/dissana8/LAB/Visor/cam3/'+c3_frame_no)
#         cam.append('/home/dissana8/LAB/Visor/cam4/'+c4_frame_no)

#         f, ax = plt.subplots(1, 4, figsize=(25, 4))

#         for i in range(4):
#             img = cv2.imread(cam[i])
#             sized = cv2.resize(img, (width, height))
#             sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

#             for j in range(2):  # This 'for' loop is for speed check
#                         # Because the first iteration is usually longer
#                 boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

#             imgfile = cam[i].split('/')[6:]
#             imgname = '/'.join(imgfile)
#             sname = savename + imgname

#             img, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)

#             image, cbbox = custom_bbox(gt[i], img, imgname)

#             # print(bbox)
            
#             if cbbox:
#                 print("sjbkjsbkjdvbskjdnkvjsbkjdvbsjbdjkbkbjvbkbkjsbkzj")
#                 cbbox = np.array(cbbox)
#                 print(cbbox)
#                 bbox = np.array(bbox)
#                 print(bbox)
#                 idx_gt_actual, idx_pred_actual, ious_actual, label = match_bboxes(cbbox, bbox)
#                 gt_actual+=len(cbbox)

#                 for h in range(len(idx_gt_actual)):
#                     t = idx_gt_actual[h]
#                     text_c = cbbox[t]
#                     # print(gt_actual)
#                     if round(ious_actual[h], 3)>=0.0:
#                         print(ious_actual[h])
#                         detections+=1
#     #                         # img = cv2.putText(img, str(round(ious_actual[h], 3)), (text_c[0], text_c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


#     #             # iou = get_iou(bbox, cbbox)
#     #             # print("iou")
#     #             # print(len(iou))

#     #             # for k in range(len(iou)):
#     #             #     img = cv2.putText(img, str(iou[k][1]), (iou[k][0][0], iou[k][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



#     #     #     ax[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     #     # savepath = "/home/dissana8/LAB/custom_bbox/"+c1_frame_no.split('/')[0]

#     #     # if not os.path.exists(savepath):
#     #     #     os.makedirs(savepath)

#     #     # plt.savefig(savepath+"/"+c1_frame_no.split('/')[-1])
#     #     # ax[0].cla()
#     #     # ax[1].cla()
#     #     # ax[2].cla()
#     #     # ax[3].cla()
#     print(detections)
#     print(gt_actual)
#     return detections/gt_actual*100, 0, 0, 0, 0


def adv_image_generation(path,file_name, model, class_names, width, height, savename, gt, device):
    cam1_det, cam2_det, cam3_det, cam4_det= 0, 0, 0, 0
    cam1_gt, cam2_gt, cam3_gt, cam4_gt = 0, 0, 0, 0
    # gt_actual=0
    #===== process the index files of camera 1 ======#
    with open('/home/dissana8/LAB/Visor/cam1/index.dmp') as f:
        content = f.readlines()
    cam_content = [x.strip() for x in content]
    c1_frames = []
    c1_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c1_frames.append(frame)
        c1_times.append(time)

    with open('/home/dissana8/LAB/Visor/cam2/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c2_frames = []
    c2_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c2_frames.append(frame)
        c2_times.append(time)
    

    # ===== process the index files of camera 3 ======#
    with open('/home/dissana8/LAB/Visor/cam3/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c3_frames = []
    c3_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c3_frames.append(frame)
        c3_times.append(time)
    

    # ===== process the index files of camera 4 ======#
    with open('/home/dissana8/LAB/Visor/cam4/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c4_frames = []
    c4_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c4_frames.append(frame)
        c4_times.append(time)
      
    #===== process the GT annotations  =======#
    with open("/home/dissana8/LAB/"+file_name) as f:
        content = f.readlines()
        

    content = [x.strip() for x in content]
    counter = -1
    print('Extracting GT annotation ...')
    c1_frame_no, c2_frame_no, c3_frame_no, c4_frame_no = [], [], [], []
    for line in content:
        counter += 1
        # if counter % 1000 == 0:
        # print(counter)
        s = line.split(" ")
        
        time = float(s[0])
        frame_idx = findClosest(time, c1_times) # we have to map the time to frame number
        c1_frame_no.append(c1_frames[frame_idx])
        

        frame_idx = findClosest(time, c2_times)  # we have to map the time to frame number
        c2_frame_no.append(c2_frames[frame_idx])
        

        frame_idx = findClosest(time, c3_times)  # we have to map the time to frame number
        c3_frame_no.append(c3_frames[frame_idx])

        
        frame_idx = findClosest(time, c4_times)  # we have to map the time to frame number
        c4_frame_no.append(c4_frames[frame_idx])


    patch = cv2.imread("/home/dissana8/Daedalus-physical/physical_examples/0.3 confidence__/adv_poster.png")
    resized_patch = cv2.resize(patch, (20, 20))

    # view 01 success rate
    print("View 01 success rate")
    for ele in enumerate(c1_frame_no):
        #real images
        im = "/home/dissana8/LAB/Visor/cam1/"+ele[1]

        #adversarial images
        # im = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam1/"+ele[1]
        img = cv2.imread(im)
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for j in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
            boxes = do_detect(model, sized, 0.7, 0.6, use_cuda)

        # print(boxes)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        print(sname)


        if not os.path.exists(directory):
            os.makedirs(directory)

        # sname = 'test_bbox.png'
        img_, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)
        replace = img.copy()
        for i in range(len(bbox)):
            x = int((bbox[i][0]+bbox[i][2])/2)
            # y = int((bbox[i][1]+bbox[i][3])/2)
            y = int((bbox[i][3]-bbox[i][1])/3)+bbox[i][1]
            # print(x)
            # print(y)

            if (y+10)>=480 or (x+10)>=640 or (x-10)<0 or (y-10)<0:
                continue
            else:
                replace[y-10: y + 10, x-10 : x + 10] = resized_patch
    
        cv2.imwrite(sname, replace)

        

#     # view 02 success rate
    print("View 02 success rate")
    for ele in enumerate(c2_frame_no):

        #real images
        im = "/home/dissana8/LAB/Visor/cam1/"+ele[1]


        # im = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam2/"+ele[1]
        img = cv2.imread(im)
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for j in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
            boxes = do_detect(model, sized, 0.7, 0.6, use_cuda)

        # print(boxes)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        print(sname)


        if not os.path.exists(directory):
            os.makedirs(directory)

        # sname = 'test_bbox.png'
        img_, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)
        print(bbox)
        replace = img.copy()
        for i in range(len(bbox)):
            x = int((bbox[i][0]+bbox[i][2])/2)
            # y = int((bbox[i][1]+bbox[i][3])/2)
            y = int((bbox[i][3]-bbox[i][1])/3)+bbox[i][1]
            # print(x)
            # print(y)

            if (y+10)>=480 or (x+10)>=640 or (x-10)<0 or (y-10)<0:
                continue
            else:
                replace[y-10: y + 10, x-10 : x + 10] = resized_patch
    
        cv2.imwrite(sname, replace)

        

#     # view 03 success rate
    print("View 03 success rate")
    for ele in enumerate(c3_frame_no):
        #real images
        im = "/home/dissana8/LAB/Visor/cam1/"+ele[1]

        # im = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam3/"+ele[1]
        img = cv2.imread(im)
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for j in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
            boxes = do_detect(model, sized, 0.7, 0.6, use_cuda)

        # print(boxes)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        print(sname)


        if not os.path.exists(directory):
            os.makedirs(directory)

        # sname = 'test_bbox.png'
        img_, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)
        print(bbox)
        replace = img.copy()
        for i in range(len(bbox)):
            x = int((bbox[i][0]+bbox[i][2])/2)
            # y = int((bbox[i][1]+bbox[i][3])/2)
            y = int((bbox[i][3]-bbox[i][1])/3)+bbox[i][1]
            # print(x)
            # print(y)

            if (y+10)>=480 or (x+10)>=640 or (x-10)<0 or (y-10)<0:
                continue
            else:
                replace[y-10: y + 10, x-10 : x + 10] = resized_patch
    
        cv2.imwrite(sname, replace)


#     # view 04 success rate
    print("View 04 success rate")
    for ele in enumerate(c4_frame_no):
        #real images
        im = "/home/dissana8/LAB/Visor/cam1/"+ele[1]

        # im = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam4/"+ele[1]
        img = cv2.imread(im)
        sized = cv2.resize(img, (width, height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        for j in range(2):  # This 'for' loop is for speed check
                    # Because the first iteration is usually longer
            boxes = do_detect(model, sized, 0.7, 0.6, use_cuda)

        # print(boxes)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        print(sname)


        if not os.path.exists(directory):
            os.makedirs(directory)

        # sname = 'test_bbox.png'
        img_, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)
        print(bbox)
        replace = img.copy()
        for i in range(len(bbox)):
            x = int((bbox[i][0]+bbox[i][2])/2)
            # y = int((bbox[i][1]+bbox[i][3])/2)
            y = int((bbox[i][3]-bbox[i][1])/3)+bbox[i][1]
            # print(x)
            # print(y)

            if (y+10)>=480 or (x+10)>=640 or (x-10)<0 or (y-10)<0:
                continue
            else:
                replace[y-10: y + 10, x-10 : x + 10] = resized_patch
    
        cv2.imwrite(sname, replace)

    


def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.8 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.2 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False,
                             edgecolor=color, linewidth=2)

def letterbox_image_padded(image, size=(416, 416)):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih = image_copy.size
    print(size)
    w = size
    h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_copy = image_copy.resize((416, 416), Image.BICUBIC)
    # print(np.asarray(image_copy))
    # new_image = Image.new('RGB', size, (0, 0, 0))
    # print(np.asarray(new_image))
    # new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
    # print("pasted image")
    # print(np.asarray(new_image))
    # new_image = np.asarray(new_image)[np.newaxis, :, :, :] / 255.
    new_image = np.asarray(image_copy)[np.newaxis, :, :, :] / 255.
    meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)

    return new_image, meta

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection area and dividing it by the sum
    # of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def bb_size(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def bb_inside(boxA, boxB):  # return True if boxA is inside boxB
    return boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and boxA[2] <= boxB[2] and boxA[3] <= boxB[3]


def bb_overlap(boxA, boxB):
    return bb_intersection_over_union(boxA, boxB) > 0


def extract_roi(detections, class_id, img_bbox, min_size, patch_size):
    roi_candidates = []
    for did, detection in enumerate(detections):
        if int(detection[0]) != class_id:
            continue
        obj_bbox = tuple(map(int, detection[-4:]))
        pat_bbox = (obj_bbox[0] + (obj_bbox[2] - obj_bbox[0]) // 2 - patch_size[1] // 2,
                    obj_bbox[1] + (obj_bbox[3] - obj_bbox[1]) // 2 - patch_size[0] // 2,
                    obj_bbox[0] + (obj_bbox[2] - obj_bbox[0]) // 2 + patch_size[1] // 2,
                    obj_bbox[1] + (obj_bbox[3] - obj_bbox[1]) // 2 + patch_size[0] // 2)
        if bb_size(obj_bbox) >= min_size and bb_inside(pat_bbox, img_bbox):
            roi_candidates.append((float(detection[1]), obj_bbox, pat_bbox, did))

    # Filter out rois with overlapped patches based on non-maximum suppression
    roi_candidates = sorted(roi_candidates, key=lambda x: -x[0])
    rois = []
    for roi_candidate in roi_candidates:
        if not np.any([bb_overlap(roi_candidate[2], roi[2]) for roi in rois]):
            rois.append(roi_candidate)
    return rois

def single_image_det(height, width):
    # patch = np.load('/home/dissana8/TOG/Adv_images/vanishing/2022-03-09_14:51:18_person/Epoch-19_Loss-8.84_ASR-0.80.npy')
    # patch_rand = np.reshape(patch.copy(), newshape=(patch.shape[0]*patch.shape[1]*patch.shape[2], patch.shape[3]))
    # np.random.shuffle(patch_rand)
    # patch_rand = np.reshape(patch_rand, newshape=patch.shape)


    patch = cv2.imread("/home/dissana8/Daedalus-physical/physical_examples/0.3 confidence__/adv_poster.png")
    resized_patch = cv2.resize(patch, (20, 20))
    # im = "/home/dissana8/pytorch-YOLOv4/images-6.jpg"
    im = "/home/dissana8/LAB/Visor/cam3/000005/005015.jpg"
    
    img = cv2.imread(im)
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for j in range(2):  # This 'for' loop is for speed check
                # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.7, 0.6, use_cuda)

    # print(boxes)

    imgfile = im.split('/')[6:]
    imgfile_ = im.split('/')[5:]

    imgname = '/'.join(imgfile)
    imgname_ = '/'.join(imgfile_)
    sname = savename + imgname_
    # imgname = '/'.join(sname)
    sname_ = sname.split('/')[:7]
    directory = '/'.join(sname_)
    print(sname)


    if not os.path.exists(directory):
        os.makedirs(directory)

    # sname = 'test_bbox.png'
    img_, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)
    print(bbox)
    replace = img.copy()
    for i in range(len(bbox)):
        x = int((bbox[i][0]+bbox[i][2])/2)
        # y = int((bbox[i][1]+bbox[i][3])/2)
        y = int((bbox[i][3]-bbox[i][1])/3)+bbox[i][1]
        # print(x)
        # print(y)

        if (y+10)>=480 or (x+10)>=640 or (x-10)<0 or (y-10)<0:
            continue
        else:
            replace[y-10: y + 10, x-10 : x + 10] = resized_patch
   
    cv2.imwrite(sname, replace)


    im = "/home/dissana8/pytorch-YOLOv4/output_adv/cam3/000005/005015.jpg"
    img = cv2.imread(im)
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for j in range(2):  # This 'for' loop is for speed check
                # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.7, 0.6, use_cuda)

    # print(boxes)
    # # imgfile = im.split('/')[6:]
    # # imgname = '/'.join(imgfile)
    # # print(imgname)
    # # sname = savename + imgname

    sname = 'test_bbox.png'
    img_, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)
    cv2.imwrite('boxed__1.png', img_)


if __name__ == "__main__":
    import sys
    import cv2

    namesfile = None
    if len(sys.argv) == 5:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        # imgfile = sys.argv[3]
        height = int(sys.argv[3])
        width = int(sys.argv[4])
    elif len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        # imgfile = sys.argv[3]
        height = sys.argv[3]
        width = int(sys.argv[4])
        namesfile = int(sys.argv[5])
    else:
        print('Usage: ')
        print('  python models.py num_classes weightfile imgfile namefile')

    model = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    model.cuda()

    use_cuda = True
    if use_cuda:
        model.cuda()

    #real images
    # path = "/home/dissana8/LAB/"

    #adversarial images
    path = "/home/dissana8/TOG/Adv_images/vanishing/LAB_16x16/"

    file_name = 'LAB-GROUNDTRUTH.ref'

    if namesfile == None:
            if n_classes == 20:
                namesfile = '/home/dissana8/pytorch-YOLOv4/data/voc.names'
            elif n_classes == 80:
                namesfile = '/home/dissana8/pytorch-YOLOv4/data/coco.names'
            else:
                print("please give namefile")

    class_names = load_class_names(namesfile)

    savename = '/home/dissana8/pytorch-YOLOv4/output_adv/'


    gt = []
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam1_coords__.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam2_coords__.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam3_coords__.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam4_coords__.npy', allow_pickle=True))


    height, width = 416, 416
    # single_image_det(height, width)
    adv_image_generation(path, file_name, model, class_names, width, height,  savename, gt, device)

    # fig, a = plt.subplots(4, 1)
    # extract_frames(path, file_name, model, class_names, width, height, savename, gt)

    # success_rate, cam1_success_rate, cam2_success_rate, cam3_success_rate, cam4_success_rate = extract_frames(path, file_name, model, class_names, width, height,  savename, gt, device)

    # f = open("success_rate_adv.txt", "a")
    # f.write("Success rate of Yolo-V4 : " +str(success_rate)+"\n")
    # f.write("Success rate of view 01" +": "+str(cam1_success_rate)+"\n")
    # f.write("Success rate of view 02" +": "+str(cam2_success_rate)+"\n")
    # f.write("Success rate of view 03" +": "+str(cam3_success_rate)+"\n")
    # f.write("Success rate of view 04" +": "+str(cam4_success_rate)+"\n")
    # f.write("\n")
    # f.write("\n")
    # f.close()
    # root = "/home/dissana8/LAB/Visor/cam1"
    # files=[]
    # pattern = "*.jpg"
    # success = 0

    # f = open("cam2_paths.txt", "a")
    # for path, subdirs, files in os.walk(root):
    #     for name in files:
    #         if fnmatch(name, pattern):
    #             f.write(os.path.join(path, name)+"\n")
    # f.close()

    # #f = open("cam1_paths.txt", "r")
    # f = open("sample_cam1.txt", "r")
    # fig, a = plt.subplots(4, 1)
    # files = f.readlines()
    # for i in range(len(files)):
    #     imgfile = files[i].strip('\n')
    #     img = cv2.imread(imgfile)

    # # Inference input size is 416*416 does not mean training size is the same
    # # Training size could be 608*608 or even other sizes
    # # Optional inference sizes:
    # #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    # #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    #     sized = cv2.resize(img, (width, height))
    #     sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    

    #     for j in range(2):  # This 'for' loop is for speed check
    #                     # Because the first iteration is usually longer
    #         boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

        

        # imgfile = imgfile.split('/')[6:]
        # imgname = '/'.join(imgfile)
        # savename = '/home/dissana8/pytorch-YOLOv4/output/'+imgname
        # print(savename)
        # img, det_count = plot_boxes_cv2(img, boxes[0], savename, class_names)
        # print("Number of people detected:", det_count)
            

        # for k in range(len(gt)): 
        #     if gt[k][0] == imgname:
        #         box = [float(gt[k][2]), float(gt[k][3]), 40, 80]
        #         box = torch.tensor(box)
        #         bbox = box_center_to_corner(box)
                    
        #         img = cv2.rectangle(img, (int(bbox[0].item()), int(bbox[1].item())), (int(bbox[2].item()), int(bbox[3].item())), (0,255,0), 1)
        
        # directory = '/home/dissana8/pytorch-YOLOv4/custom_bbox/'+imgname.split('/')[0]
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        
        # savename1 = '/home/dissana8/pytorch-YOLOv4/custom_bbox/'+imgname
        # cv2.imwrite(savename1, img)

