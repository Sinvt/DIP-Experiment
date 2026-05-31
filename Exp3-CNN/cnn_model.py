import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    轻量级卷积神经网络，专为 32x32 的 CIFAR-10 图像设计。
    结构：3 个卷积块 (Conv->BN->ReLU->MaxPool) + 2 个全连接层
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # 输入: 32x32x3 -> 输出: 16x16x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 输入: 16x16x32 -> 输出: 8x8x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 输入: 8x8x64 -> 输出: 4x4x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Dropout 防止过拟合
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征图
        x = self.classifier(x)
        return x
