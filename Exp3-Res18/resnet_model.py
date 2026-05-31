import torch.nn as nn
from torchvision import models

def get_resnet18_cifar10():
    """
    加载 ResNet-18，并针对 CIFAR-10 的 32x32 分辨率进行微调：
    1. 替换第一层的大卷积核（7x7 stride=2）为 3x3 stride=1
    2. 移除第一层的最大池化层（替换为 Identity）
    3. 修改最后的分类头适配 10 分类
    """
    model = models.resnet18(weights=None)
    
    # 1. 替换 Conv1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. 移除 MaxPool
    model.maxpool = nn.Identity()
    
    # 3. 替换全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    return model
