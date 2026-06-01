# %% [markdown]
# # Exp3: CIFAR-10 训练代码 (Vision Transformer)
#
# 本脚本负责下载数据、进行数据预处理（放大至 224x224）、
# 从 `vit_model.py` 中导入搭建好的 ViT 模型进行迁移学习，并保存权重。

# %%
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 导入独立封装好的 ViT 模型
from vit_model import get_vit_cifar10_model

# 中文显示配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置参数
BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用计算设备: {device}")

# %% [markdown]
# ## 1. 数据预处理与加载
# 将 32x32 的 CIFAR-10 图像放大为 224x224，并进行标准化。

# %%
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_clean_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("正在加载 CIFAR-10 数据集...")
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transforms)
test_clean_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_clean_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_clean_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %% [markdown]
# ## 2. 搭建 ViT 模型 (迁移学习)

# %%
print("\n正在从 vit_model.py 初始化预训练的 Vision Transformer 模型...")
model = get_vit_cifar10_model(pretrained=True)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %% [markdown]
# ## 3. 开始训练
# 进行 Epoch 循环，并记录 Loss 和 Accuracy。

# %%
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

print("\n开始训练 (共 {} 个 Epochs)...".format(EPOCHS))
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    elapsed = time.time() - start_time
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Time: {elapsed:.0f}s | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

print("\n[OK] 训练完成！最高验证准确率: {:.2f}%".format(max(history['val_acc'])))

# %% [markdown]
# ## 4. 保存模型与训练曲线

# %%
# 保存权重
save_path = os.path.join(RESULT_DIR, 'vit_b_16_cifar10.pth')
torch.save(model.state_dict(), save_path)
print(f"模型权重已保存至: {save_path}")

# 绘制学习曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(range(1, EPOCHS+1), history['train_loss'], label='Train Loss', marker='o')
ax1.plot(range(1, EPOCHS+1), history['val_loss'], label='Val Loss', marker='s')
ax1.set_title('Training and Validation Loss')
ax1.legend()

ax2.plot(range(1, EPOCHS+1), history['train_acc'], label='Train Accuracy', marker='o')
ax2.plot(range(1, EPOCHS+1), history['val_acc'], label='Val Accuracy', marker='s')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'exp3_learning_curves.png'))
plt.show()
