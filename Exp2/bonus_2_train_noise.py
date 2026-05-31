# %% [markdown]
# # 选做 2: 改变训练集的加噪方式
#
# 探究不同训练加噪方式下，模型测试时面对不同加噪方式的表现。
# 具体操作：
# 1. 测试现有的 DnCNN-S-25 (纯高斯噪声训练) 在高斯噪声和椒盐噪声下的表现。
# 2. 使用训练集 (Data/train) 增加椒盐噪声，对模型进行微调 (Fine-tuning)。
# 3. 对比微调前后的模型，在高斯噪声和椒盐噪声上的 PSNR 变化和主观去噪效果。

# %%
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train', 'train')
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'bonus2')
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from dncnn_model import DnCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"训练设备: {device}")

# %%
# =============================================
# 1. 噪声生成函数 & 数据集定义
# =============================================

def add_gaussian_noise(image, sigma=25):
    """添加高斯噪声 (返回 0-255 float64)"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255)

def add_salt_pepper_noise(image, prob=0.05):
    """添加椒盐噪声 (返回 0-255 float64)"""
    noisy = np.copy(image)
    h, w = noisy.shape
    num_salt = np.ceil(prob * image.size * 0.5)
    num_pepper = np.ceil(prob * image.size * 0.5)

    # 盐噪声 (白色)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255

    # 椒噪声 (黑色)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy

class DenoisingDataset(Dataset):
    """去噪训练数据集: 随机裁剪 patch 并动态加噪"""
    def __init__(self, img_dir, patch_size=64, noise_type='sp', noise_param=0.05):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]
        self.patch_size = patch_size
        self.noise_type = noise_type
        self.noise_param = noise_param

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
        h, w = img.shape
        
        # 随机裁剪
        y = np.random.randint(0, h - self.patch_size + 1)
        x = np.random.randint(0, w - self.patch_size + 1)
        clean_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        
        # 加噪
        if self.noise_type == 'sp':
            noisy_patch = add_salt_pepper_noise(clean_patch, self.noise_param)
        elif self.noise_type == 'gaussian':
            noisy_patch = add_gaussian_noise(clean_patch, self.noise_param)
        else:
            noisy_patch = clean_patch

        # 归一化到 [0, 1]
        clean_patch = clean_patch / 255.0
        noisy_patch = noisy_patch / 255.0
        
        # 转换为 tensor
        clean_tensor = torch.from_numpy(clean_patch).float().unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy_patch).float().unsqueeze(0)
        
        return noisy_tensor, clean_tensor

# %%
# =============================================
# 2. 模型加载与评估工具
# =============================================

def load_dncnn(weight_path, device):
    """加载 DnCNN 模型权重"""
    sd = torch.load(weight_path, map_location=device, weights_only=False)
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    
    conv_indices = [int(k.split('.')[1]) for k in new_sd if k.endswith('.weight') and len(new_sd[k].shape) == 4]
    last_conv_idx = max(conv_indices)
    num_layers = (last_conv_idx - 2) // 3 + 2
    
    model = DnCNN(channels=1, num_of_layers=num_layers)
    model.load_state_dict(new_sd, strict=False)
    model = model.to(device)
    return model

def evaluate_model(model, img, device, name="Model"):
    """测试模型在高斯和椒盐上的表现"""
    model.eval()
    
    # 加噪
    np.random.seed(42)
    img_g = add_gaussian_noise(img, 25)
    img_sp = add_salt_pepper_noise(img, 0.05)
    
    # 推理函数
    def infer(noisy_img):
        with torch.no_grad():
            x = noisy_img / 255.0
            t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
            out = model(t)
            result = out.squeeze().cpu().numpy()
            return np.clip(result, 0, 1) * 255.0

    denoised_g = infer(img_g)
    denoised_sp = infer(img_sp)
    
    psnr_g = peak_signal_noise_ratio(img, denoised_g, data_range=255.0)
    psnr_sp = peak_signal_noise_ratio(img, denoised_sp, data_range=255.0)
    
    print(f"[{name}]")
    print(f"  高斯噪声 (sigma=25) 去噪 PSNR: {psnr_g:.2f} dB")
    print(f"  椒盐噪声 (d=0.05)   去噪 PSNR: {psnr_sp:.2f} dB")
    
    return {
        'noisy_g': img_g, 'denoised_g': denoised_g, 'psnr_g': psnr_g,
        'noisy_sp': img_sp, 'denoised_sp': denoised_sp, 'psnr_sp': psnr_sp
    }

# 加载测试图
lena = np.array(Image.open(os.path.join(BASE_DIR, 'images', 'lena.png')).convert('L'), dtype=np.float64)

# 加载预训练模型 (高斯噪声训练的)
print("\n--- 微调前: DnCNN-S-25 ---")
model_pretrained = load_dncnn(os.path.join(LOG_DIR, 'DnCNN-S-25', 'net.pth'), device)
res_pre = evaluate_model(model_pretrained, lena, device, "预训练模型 (DnCNN-S-25)")

# %%
# =============================================
# 3. 针对椒盐噪声微调模型
# =============================================
print("\n--- 开始微调模型 ---")
print("目标: 使用椒盐噪声 (prob=0.05) 微调模型, 让其具备去除椒盐噪声的能力")

# 复制预训练模型
model_finetuned = load_dncnn(os.path.join(LOG_DIR, 'DnCNN-S-25', 'net.pth'), device)
model_finetuned.train()

# 准备数据 (仅训练 5 个 epoch 以快速验证)
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4

dataset = DenoisingDataset(TRAIN_DIR, patch_size=50, noise_type='sp', noise_param=0.05)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_finetuned.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for noisy, clean in dataloader:
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model_finetuned(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.6f}")

print("\n微调完成！")

# %%
# =============================================
# 4. 微调后的评估
# =============================================
print("\n--- 微调后评估 ---")
res_ft = evaluate_model(model_finetuned, lena, device, "微调后模型 (SP-Tuned)")

# %%
# =============================================
# 5. 可视化对比
# =============================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('微调前后 DnCNN 面对不同加噪方式的去噪表现对比', fontsize=16, fontweight='bold')

# 第一行: 高斯噪声
axes[0, 0].imshow(res_pre['noisy_g'], cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('输入: 高斯噪声 (sigma=25)', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(res_pre['denoised_g'], cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title(f'微调前 (高斯模型) 去噪\nPSNR = {res_pre["psnr_g"]:.2f}dB', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(res_ft['denoised_g'], cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title(f'微调后 (增加椒盐训练) 去噪\nPSNR = {res_ft["psnr_g"]:.2f}dB', fontsize=12)
axes[0, 2].axis('off')

axes[0, 3].imshow(lena, cmap='gray', vmin=0, vmax=255)
axes[0, 3].set_title('原图 (Ground Truth)', fontsize=12)
axes[0, 3].axis('off')

# 第二行: 椒盐噪声
axes[1, 0].imshow(res_pre['noisy_sp'], cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title('输入: 椒盐噪声 (d=0.05)', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(res_pre['denoised_sp'], cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title(f'微调前 (高斯模型) 去噪\nPSNR = {res_pre["psnr_sp"]:.2f}dB', fontsize=12)
axes[1, 1].axis('off')

axes[1, 2].imshow(res_ft['denoised_sp'], cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title(f'微调后 (增加椒盐训练) 去噪\nPSNR = {res_ft["psnr_sp"]:.2f}dB', fontsize=12)
axes[1, 2].axis('off')

axes[1, 3].imshow(lena, cmap='gray', vmin=0, vmax=255)
axes[1, 3].set_title('原图 (Ground Truth)', fontsize=12)
axes[1, 3].axis('off')

plt.tight_layout()
save_path = os.path.join(RESULT_DIR, 'bonus2_comparison.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\n可视化结果已保存至: {save_path}")

# %% [markdown]
# ### 结论分析
#
# 1. **微调前 (DnCNN-S-25)**: 
#    因为它是纯高斯噪声训练出的模型，所以能完美去除高斯噪声 (PSNR 32+dB)，但在面临它没见过的**椒盐噪声**时，会留下密密麻麻的噪点，PSNR 很低，去噪效果极差。
#
# 2. **微调后 (引入椒盐噪声训练)**:
#    当我们用椒盐噪声对模型进行了几个 Epoch 的微调后，它**学会了**处理椒盐特征，椒盐噪声去噪效果大幅提升。
#    但同时可能会伴随灾难性遗忘：如果在微调时**只**用了椒盐噪声而没有混合高斯，模型在去高斯噪声上的表现会有所衰退。如果想要两者兼顾，训练集应当进行混合噪声训练 (Mixed Noise)。
