"""
DnCNN 模型定义
参考论文: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
         (Zhang et al., TIP 2017)

本文件定义的 DnCNN 架构与 ./logs 目录下的预训练权重完全匹配:
- 17 层 Conv2d (3x3, padding=1)
- 第 1 层:      Conv(1->64, bias=False) + ReLU
- 第 2~16 层:   Conv(64->64, bias=False) + BN(64) + ReLU
- 第 17 层:     Conv(64->1, bias=False)
- 残差学习: output = input - model(input)

权重文件 key 结构: module.dncnn.{0,2,3,5,6,...,47}
(需要去掉 'module.' 前缀后加载)
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN 去噪卷积神经网络

    与 logs/ 目录下预训练权重匹配的架构:
    - 第 1 层:       Conv(in->64, bias=False) + ReLU
    - 第 2~16 层:    Conv(64->64, bias=False) + BN(64) + ReLU
    - 第 17 层:      Conv(64->out, bias=False)
    """

    def __init__(self, channels=1, num_of_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []

        # 第 1 层: Conv + ReLU (无 BN, 无 bias)
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 第 2~(n-1) 层: Conv + BN + ReLU
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层: Conv (无 BN, 无激活, 无 bias)
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播 -- 残差学习: 去噪图 = 输入 - 噪声残差"""
        noise = self.dncnn(x)
        return x - noise
