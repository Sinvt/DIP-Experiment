# %% [markdown]
# # Exp3: CIFAR-10 训练代码 (自定义 ResNet-18 + 高级策略)
# 
# 本脚本集成了多项先进训练技巧：
# 1. 专门适配 32x32 分辨率的 ResNet-18
# 2. Mixup 数据增强
# 3. Label Smoothing (标签平滑)
# 4. 带 Momentum 和 Weight Decay 的 SGD 优化器
# 5. Cosine Annealing (余弦退火) 学习率调度

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

# 导入自定义模型
from resnet_model import get_resnet18_cifar10

# 中文显示配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置参数
BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

# 冲击 95% 的精度需要较长训练时间，建议设为 100 甚至 200。此处我们设定为 100
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 0.4
MIXUP_ALPHA = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用计算设备: {device}")

# %% [markdown]
# ## 1. 数据预处理与加载

# %%
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_clean_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("正在加载 CIFAR-10 数据集...")
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transforms)
test_clean_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_clean_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(test_clean_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# %% [markdown]
# ## 2. 模型搭建、损失函数与优化器

# %%
print("\n正在初始化微调版 ResNet-18 模型...")
model = get_resnet18_cifar10()
model = model.to(device)

# Label Smoothing: 0.1
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# SGD + Momentum + Weight Decay
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

# 余弦退火学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# %% [markdown]
# ## 3. Mixup 辅助函数

# %%
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# %% [markdown]
# ## 4. 开始训练

# %%
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 应用 Mixup
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

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

print("\n开始冲击 95% 精度大关 (共 {} 个 Epochs)...".format(EPOCHS))
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    elapsed = time.time() - start_time
    print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | LR: {current_lr:.5f} | Time: {elapsed:.0f}s | Train Mixup Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

print("\n[OK] 训练完成！最高验证准确率: {:.2f}%".format(max(history['val_acc'])))

# %% [markdown]
# ## 5. 保存模型与训练曲线

# %%
os.makedirs(RESULT_DIR, exist_ok=True)
save_path = os.path.join(RESULT_DIR, 'resnet18_cifar10.pth')
torch.save(model.state_dict(), save_path)
print(f"模型权重已保存至: {save_path}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(range(1, EPOCHS+1), history['train_loss'], label='Train Mixup Loss')
ax1.plot(range(1, EPOCHS+1), history['val_loss'], label='Val Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()

ax2.plot(range(1, EPOCHS+1), history['val_acc'], label='Val Accuracy')
ax2.set_title('Validation Accuracy (Target > 95%)')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'exp3_resnet_learning_curves.png'))
plt.close()
