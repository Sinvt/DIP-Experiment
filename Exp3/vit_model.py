import torch.nn as nn
from torchvision import models

def get_vit_cifar10_model(pretrained=True):
    """
    加载 Vision Transformer (ViT-B/16) 模型，
    并将其最后的输出分类头修改为适配 CIFAR-10 的 10 个类别。
    
    参数:
        pretrained (bool): 是否加载官方在 ImageNet 上预训练的权重。
                           如果在训练阶段，应设为 True 进行迁移学习；
                           如果在测试阶段自行加载本地权重，则设为 False 即可。
    """
    if pretrained:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    else:
        model = models.vit_b_16(weights=None)
        
    # 获取原始分类头的输入特征维度
    in_features = model.heads.head.in_features
    # 替换为新的 10 分类全连接层
    model.heads.head = nn.Linear(in_features, 10)
    
    return model
