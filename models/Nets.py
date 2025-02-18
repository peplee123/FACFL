#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn

import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        # 如果输入和输出通道不同，则需要调整尺寸
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet9(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet9, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = BasicBlock(64, 128, stride=2)
        self.layer2 = BasicBlock(128, 256, stride=2)
        self.layer3 = BasicBlock(256, 512, stride=2)

        self.pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

# def conv_bn_relu_pool(in_channels, out_channels, pool=False):
#     layers = [
#         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
#         #nn.BatchNorm2d(out_channels),
#         nn.GroupNorm(32,out_channels),
#         nn.ReLU(inplace=True)
#     ]
#     if pool:
#         layers.append(nn.MaxPool2d(2))
#     return nn.Sequential(*layers)
#
# class ResNet9(nn.Module):
#     def __init__(self, in_channels, num_classes, dim=512):
#         super().__init__()
#         self.prep = conv_bn_relu_pool(in_channels, 64)
#         self.layer1_head = conv_bn_relu_pool(64, 128, pool=True)
#         self.layer1_residual = nn.Sequential(conv_bn_relu_pool(128, 128), conv_bn_relu_pool(128, 128))
#         self.layer2 = conv_bn_relu_pool(128, 256, pool=True)
#         self.layer3_head = conv_bn_relu_pool(256, 512, pool=True)
#         self.layer3_residual = nn.Sequential(conv_bn_relu_pool(512, 512), conv_bn_relu_pool(512, 512))
#         self.MaxPool2d = nn.Sequential(
#             nn.MaxPool2d(4))
#         self.linear = nn.Linear(dim, num_classes)
#         # self.classifier = nn.Sequential(
#         #     nn.MaxPool2d(4),
#         #     nn.Flatten(),
#         #     nn.Linear(512, num_classes))
#
#
#     def forward(self, x):
#         x = self.prep(x)
#         x = self.layer1_head(x)
#         x = self.layer1_residual(x) + x
#         x = self.layer2(x)
#         x = self.layer3_head(x)
#         x = self.layer3_residual(x) + x
#         x = self.MaxPool2d(x)
#         x = x.view(x.size(0), -1)
#         #print(x.shape)
#         x = self.linear(x)
#         return x
'''
prep.0.weight torch.Size([64, 3, 3, 3])
prep.0.bias torch.Size([64])
prep.1.weight torch.Size([64])
prep.1.bias torch.Size([64])
layer1_head.0.weight torch.Size([128, 64, 3, 3])
layer1_head.0.bias torch.Size([128])
layer1_head.1.weight torch.Size([128])
layer1_head.1.bias torch.Size([128])
layer1_residual.0.0.weight torch.Size([128, 128, 3, 3])
layer1_residual.0.0.bias torch.Size([128])
layer1_residual.0.1.weight torch.Size([128])
layer1_residual.0.1.bias torch.Size([128])
layer1_residual.1.0.weight torch.Size([128, 128, 3, 3])
layer1_residual.1.0.bias torch.Size([128])
layer1_residual.1.1.weight torch.Size([128])
layer1_residual.1.1.bias torch.Size([128])
layer2.0.weight torch.Size([256, 128, 3, 3])
layer2.0.bias torch.Size([256])
layer2.1.weight torch.Size([256])
layer2.1.bias torch.Size([256])
layer3_head.0.weight torch.Size([512, 256, 3, 3])
layer3_head.0.bias torch.Size([512])
layer3_head.1.weight torch.Size([512])
layer3_head.1.bias torch.Size([512])
layer3_residual.0.0.weight torch.Size([512, 512, 3, 3])
layer3_residual.0.0.bias torch.Size([512])
layer3_residual.0.1.weight torch.Size([512])
layer3_residual.0.1.bias torch.Size([512])
layer3_residual.1.0.weight torch.Size([512, 512, 3, 3])
layer3_residual.1.0.bias torch.Size([512])
layer3_residual.1.1.weight torch.Size([512])
layer3_residual.1.1.bias torch.Size([512])
linear.weight torch.Size([100, 512])
linear.bias torch.Size([100])
'''
# # 定义一个基础的卷积块
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#
# # 定义一个Residual Block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
# class ResNet9(nn.Module):
#     def __init__(self, num_classes=100):
#         super(ResNet9, self).__init__()
#         self.in_channels = 64
#         self.conv = conv3x3(3, 64)
#         self.bn = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(ResidualBlock, 64, 1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, 2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(256, num_classes)
#
#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(
#                 conv3x3(self.in_channels, out_channels, stride=stride),
#                 nn.BatchNorm2d(out_channels)
#             )
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out





class LeNet5Fmnist(nn.Module):
    def __init__(self, num_classes=10):

        super(LeNet5Fmnist, self).__init__()


        # 使用与CNNMnist相同的参数来调整LeNet的卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # 使用与CNNMnist相同的参数来调整LeNet的全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)
        return x


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CrossAttention(nn.Module):
#     def __init__(self, dim):
#         super(CrossAttention, self).__init__()
#         self.query_layer = nn.Linear(dim, dim)
#         self.key_layer = nn.Linear(dim, dim)
#         self.value_layer = nn.Linear(dim, dim)
#
#     def forward(self, queries, keys, values):
#         queries = self.query_layer(queries)
#         keys = self.key_layer(keys)
#         values = self.value_layer(values)
#
#         attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (keys.size(-1) ** 0.5)
#         attention_weights = F.softmax(attention_scores, dim=-1)
#
#         output = torch.matmul(attention_weights, values)
#         return output
#
#
# class LeNet5Cifar(nn.Module):
#     def __init__(self, num_classes=10):
#         super(LeNet5Cifar, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=5, padding=0, stride=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(1600, 512),
#             nn.ReLU(inplace=True)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True)
#         )
#
#         self.attention = CrossAttention(dim=256)  # 交叉注意力层
#         self.fc = nn.Linear(256, num_classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x1 = self.fc2(x)  # 记录第二个全连接层的输出
#
#         # 应用交叉注意力机制
#         attention_output = self.attention(x1, x1, x1)  # 自注意力，可以根据需求调整
#         x = attention_output + x1  # 残差连接
#         x = self.fc(x)
#         return x


class LeNet5Cifar(nn.Module):
    def __init__(self, num_classes=10):

        super(LeNet5Cifar, self).__init__()


        # 使用与CNNMnist相同的参数来调整LeNet的卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        # 使用与CNNMnist相同的参数来调整LeNet的全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)
        return x

class LeNet5fm(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5fm, self).__init__()

        # Layer 1: Conv2d + ReLU + MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True),  # in_channels=1, out_channels=6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # kernel_size=(2, 2)
        )

        # Layer 2: Conv2d + ReLU + MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),  # in_channels=6, out_channels=16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # kernel_size=(2, 2)
        )

        # Fully connected layers
        # After applying conv1 (28x28 -> 24x24), then conv2 (24x24 -> 20x20), and maxpool (20x20 -> 10x10)
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),  # in_features=16*4*4 (after conv and pool), out_features=120
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),  # in_features=120, out_features=84
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(84, num_classes)  # in_features=84, out_features=num_classes

    def forward(self, x):
        x = self.conv1(x)  # Layer 1
        x = self.conv2(x)  # Layer 2
        x = torch.flatten(x, 1)  # Flatten into a vector
        x = self.fc1(x)  # Layer 3
        x = self.fc2(x)  # Layer 4
        x = self.fc3(x)  # Layer 5
        return x



class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Layer 1: Conv2d + ReLU + MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0, bias=True),  # in_channels=3, out_channels=6
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # kernel_size=(2, 2)
        )

        # Layer 2: Conv2d + ReLU + MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True),  # in_channels=6, out_channels=16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # kernel_size=(2, 2)
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # in_features=400 (16*5*5 for 32x32 input), out_features=120
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),  # in_features=120, out_features=84
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(84, num_classes)  # in_features=84, out_features=num_classes

    def forward(self, x):
        x = self.conv1(x)  # Layer 1
        x = self.conv2(x)  # Layer 2
        x = torch.flatten(x, 1)  # Flatten into a vector
        x = self.fc1(x)  # Layer 3
        x = self.fc2(x)  # Layer 4
        x = self.fc3(x)  # Layer 5
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
class CNNMnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

class CNNCifar(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class CNNCifar100(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, 100)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class CNNTinyImage(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(10816, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, 200)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


class fastText(nn.Module):
    def __init__(self, hidden_dim, padding_idx=0, vocab_size=98635, num_classes=10):
        super(fastText, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # Hidden Layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # Output Layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        text = x

        embedded_sent = self.embedding(text)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc(h)
        out = z

        return out

class CNNFemnist(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
