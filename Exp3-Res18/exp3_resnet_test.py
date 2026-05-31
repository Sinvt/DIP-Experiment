# %% [markdown]
# # Exp3: CIFAR-10 测试与加噪验证代码 (ResNet-18 + TTA)
# 
# 本脚本负责从 `resnet_model.py` 加载微调版的 ResNet-18 架构及权重，
# 对纯净测试集以及含有 20% 椒盐噪声的测试集进行推理。
# **评估时启用了 TTA (Test-Time Augmentation) 测试时增强** 以进一步压榨精度极限。

# %%
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入自定义模型
from resnet_model import get_resnet18_cifar10

# 中文显示配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
WEIGHT_PATH = os.path.join(RESULT_DIR, 'resnet18_cifar10.pth')

BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# %% [markdown]
# ## 1. 加噪函数与数据加载

# %%
class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.1, pepper_prob=0.1):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, img):
        img_arr = np.array(img)
        noise = np.random.rand(img_arr.shape[0], img_arr.shape[1])
        salt_mask = noise < self.salt_prob
        pepper_mask = (noise >= self.salt_prob) & (noise < self.salt_prob + self.pepper_prob)
        img_arr[salt_mask] = 255
        img_arr[pepper_mask] = 0
        return Image.fromarray(img_arr)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

test_clean_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_noisy_transforms = transforms.Compose([
    AddSaltPepperNoise(salt_prob=0.1, pepper_prob=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("加载测试集...")
test_clean_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_clean_transforms)
test_noisy_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=test_noisy_transforms)

clean_loader = DataLoader(test_clean_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
noisy_loader = DataLoader(test_noisy_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %% [markdown]
# ## 2. 加载训练好的 ResNet-18 权重

# %%
print("\n加载微调版 ResNet-18 架构并载入权重...")
model = get_resnet18_cifar10()

if os.path.exists(WEIGHT_PATH):
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    print("[OK] 成功载入权重: ", WEIGHT_PATH)
else:
    print("[Warning] 未找到权重文件，将使用随机权重！请先运行 exp3_resnet_train.py")

model = model.to(device)

# %% [markdown]
# ## 3. 测试与对比 (Clean vs Noisy + TTA 支持)

# %%
def eval_loader_tta(model, loader, device):
    """
    使用 TTA (测试时增强) 评估模型。
    对每张图片进行 原图前向传播 + 水平翻转前向传播，取 Logits 平均。
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 原图输出
            outputs_orig = model(inputs)
            
            # 水平翻转图输出
            inputs_flipped = torch.flip(inputs, dims=[3])
            outputs_flipped = model(inputs_flipped)
            
            # 取平均
            outputs_avg = (outputs_orig + outputs_flipped) / 2.0
            
            _, predicted = outputs_avg.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

print("\n开始评测 (启用 TTA)...")
clean_acc = eval_loader_tta(model, clean_loader, device)
noisy_acc = eval_loader_tta(model, noisy_loader, device)

print("-" * 40)
print(f"  纯净测试集准确率 (Clean Acc + TTA) : {clean_acc:.2f}%")
print(f"  噪声测试集准确率 (Noisy Acc + TTA) : {noisy_acc:.2f}%")
print("-" * 40)

# %% [markdown]
# ## 4. 预测结果可视化展示

# %%
def denormalize(tensor_img):
    img = tensor_img.clone().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)

clean_images, clean_labels = next(iter(clean_loader))
noisy_images, noisy_labels = next(iter(noisy_loader))

num_show = 4
clean_images = clean_images[:num_show].to(device)
noisy_images = noisy_images[:num_show].to(device)
labels = clean_labels[:num_show].cpu().numpy()

model.eval()
with torch.no_grad():
    clean_preds = model(clean_images).max(1)[1].cpu().numpy()
    noisy_preds = model(noisy_images).max(1)[1].cpu().numpy()

fig, axes = plt.subplots(2, num_show, figsize=(15, 7))
fig.suptitle('ResNet-18 预测: 纯净图像 vs 椒盐噪声图像', fontsize=16, fontweight='bold', y=1.02)

for i in range(num_show):
    # 纯净图
    ax = axes[0, i]
    ax.imshow(denormalize(clean_images[i]))
    ax.axis('off')
    color = 'green' if clean_preds[i] == labels[i] else 'red'
    ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[clean_preds[i]]}", color=color)
    
    # 噪声图
    ax = axes[1, i]
    ax.imshow(denormalize(noisy_images[i]))
    ax.axis('off')
    color = 'green' if noisy_preds[i] == labels[i] else 'red'
    ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[noisy_preds[i]]}", color=color)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'exp3_resnet_predictions_comparison.png'))
plt.close()
