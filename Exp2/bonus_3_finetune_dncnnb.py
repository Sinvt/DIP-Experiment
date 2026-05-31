# %% [markdown]
# # 选做 3: 提升 DnCNN-B 在特定噪声下的表现
#
# 在盲去噪场景下，DnCNN-B 模型为了兼顾各种不同的噪声（如 sigma ∈ [0, 55]），
# 其在某个特定的噪声水平（例如 sigma=25）上的表现，通常是不如专门针对该噪声水平训练的模型（如 DnCNN-S-25）的。
#
# **目标**：改变训练加噪方式，提升 DnCNN-B 对 lena 和 cameraman 在高斯噪声 sigma=25 时的去噪效果。
# **方法**：将原本用于盲去噪的 DnCNN-B 模型，使用 `sigma=25` 的特定高斯噪声进行微调（Fine-tuning），
# 使其特化并专注于处理这种强度的噪声，从而提升去噪性能。

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
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'bonus3')
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from dncnn_model import DnCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"训练设备: {device}")

# %%
# =============================================
# 工具函数 & 数据集
# =============================================

def add_gaussian_noise(image, sigma=25):
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255)

class GaussianDenoisingDataset(Dataset):
    """固定 sigma=25 的高斯去噪训练集"""
    def __init__(self, img_dir, patch_size=50, sigma=25):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]
        self.patch_size = patch_size
        self.sigma = sigma

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
        h, w = img.shape
        
        # 随机裁剪
        y = np.random.randint(0, max(1, h - self.patch_size + 1))
        x = np.random.randint(0, max(1, w - self.patch_size + 1))
        
        # 处理小图片
        clean_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        if clean_patch.shape[0] < self.patch_size or clean_patch.shape[1] < self.patch_size:
            pad_y = max(0, self.patch_size - clean_patch.shape[0])
            pad_x = max(0, self.patch_size - clean_patch.shape[1])
            clean_patch = np.pad(clean_patch, ((0, pad_y), (0, pad_x)), mode='reflect')
        
        # 加噪
        noisy_patch = add_gaussian_noise(clean_patch, self.sigma)

        # 归一化并转 tensor
        clean_patch = clean_patch / 255.0
        noisy_patch = noisy_patch / 255.0
        
        clean_tensor = torch.from_numpy(clean_patch).float().unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy_patch).float().unsqueeze(0)
        
        return noisy_tensor, clean_tensor

def load_dncnn(weight_path, device):
    """加载 DnCNN 模型权重，自动检测层数"""
    sd = torch.load(weight_path, map_location=device, weights_only=False)
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}
    
    # DnCNN-B 有 20 层，所以需要动态检测
    conv_indices = [int(k.split('.')[1]) for k in new_sd if k.endswith('.weight') and len(new_sd[k].shape) == 4]
    last_conv_idx = max(conv_indices)
    num_layers = (last_conv_idx - 2) // 3 + 2
    
    model = DnCNN(channels=1, num_of_layers=num_layers)
    model.load_state_dict(new_sd, strict=False)
    model = model.to(device)
    return model

def infer_image(model, img, device):
    """用模型对整张图像进行推理"""
    model.eval()
    with torch.no_grad():
        x = img / 255.0
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(t)
        result = out.squeeze().cpu().numpy()
        return np.clip(result, 0, 1) * 255.0

# %%
# =============================================
# 1. 评估微调前的 DnCNN-B
# =============================================

# 加载测试图
lena = np.array(Image.open(os.path.join(BASE_DIR, 'images', 'lena.png')).convert('L'), dtype=np.float64)
cameraman = np.array(Image.open(os.path.join(BASE_DIR, 'images', 'cameraman.png')).convert('L'), dtype=np.float64)

test_images = {'Lena': lena, 'Cameraman': cameraman}
sigma_test = 25

print(f"\n--- 1. 微调前: 原始 DnCNN-B 性能评估 (sigma={sigma_test}) ---")
model_b_pretrained = load_dncnn(os.path.join(LOG_DIR, 'DnCNN-B', 'net.pth'), device)

results_pre = {}
for name, img in test_images.items():
    np.random.seed(42)
    noisy_img = add_gaussian_noise(img, sigma_test)
    denoised_img = infer_image(model_b_pretrained, noisy_img, device)
    
    psnr_noisy = peak_signal_noise_ratio(img, noisy_img, data_range=255.0)
    psnr_denoised = peak_signal_noise_ratio(img, denoised_img, data_range=255.0)
    
    results_pre[name] = {
        'noisy': noisy_img,
        'denoised': denoised_img,
        'psnr_noisy': psnr_noisy,
        'psnr_denoised': psnr_denoised
    }
    
    print(f"[{name}]")
    print(f"  Noisy PSNR: {psnr_noisy:.2f} dB")
    print(f"  DnCNN-B PSNR: {psnr_denoised:.2f} dB")

# %%
# =============================================
# 2. 对 DnCNN-B 进行微调 (Fine-tuning)
# =============================================
print(f"\n--- 2. 开始微调 DnCNN-B ---")
print(f"目标: 使用特定高斯噪声 (sigma={sigma_test}) 对模型进行微调，提升其在该噪声下的性能")

# 复制一份模型进行训练
model_b_finetuned = load_dncnn(os.path.join(LOG_DIR, 'DnCNN-B', 'net.pth'), device)
model_b_finetuned.train()

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4

dataset = GaussianDenoisingDataset(TRAIN_DIR, patch_size=50, sigma=sigma_test)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_b_finetuned.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for noisy, clean in dataloader:
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        output = model_b_finetuned(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.6f}")

print("微调完成！")

# %%
# =============================================
# 3. 评估微调后的 DnCNN-B
# =============================================
print(f"\n--- 3. 微调后: 评估性能提升 ---")

results_ft = {}
for name, img in test_images.items():
    noisy_img = results_pre[name]['noisy'] # 使用和前面完全一样的噪声图
    denoised_img = infer_image(model_b_finetuned, noisy_img, device)
    psnr_denoised = peak_signal_noise_ratio(img, denoised_img, data_range=255.0)
    
    results_ft[name] = {
        'denoised': denoised_img,
        'psnr_denoised': psnr_denoised
    }
    
    print(f"[{name}]")
    print(f"  微调前 PSNR: {results_pre[name]['psnr_denoised']:.2f} dB")
    print(f"  微调后 PSNR: {psnr_denoised:.2f} dB")
    diff = psnr_denoised - results_pre[name]['psnr_denoised']
    print(f"  性能提升: {diff:+.2f} dB")

# %%
# =============================================
# 4. 结果可视化
# =============================================

for name in test_images.keys():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f'{name} 图像去噪对比 (高斯噪声 sigma={sigma_test})', fontsize=16, fontweight='bold', y=1.05)
    
    # 原图
    axes[0].imshow(test_images[name], cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('原图 (Ground Truth)', fontsize=14)
    axes[0].axis('off')
    
    # 噪声图
    axes[1].imshow(results_pre[name]['noisy'], cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'噪声图\nPSNR = {results_pre[name]["psnr_noisy"]:.2f} dB', fontsize=14)
    axes[1].axis('off')
    
    # 微调前
    axes[2].imshow(results_pre[name]['denoised'], cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f'微调前 (原始 DnCNN-B)\nPSNR = {results_pre[name]["psnr_denoised"]:.2f} dB', fontsize=14)
    axes[2].axis('off')
    
    # 微调后
    axes[3].imshow(results_ft[name]['denoised'], cmap='gray', vmin=0, vmax=255)
    axes[3].set_title(f'微调后 (特定 sigma=25 训练)\nPSNR = {results_ft[name]["psnr_denoised"]:.2f} dB', fontsize=14, color='red')
    axes[3].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, f'bonus3_comparison_{name.lower()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

print(f"\n可视化结果已保存至: {RESULT_DIR}")

# %% [markdown]
# ### 结论
# 
# 盲去噪模型 (DnCNN-B) 由于需要在训练期间覆盖非常广泛的噪声强度，在处理某一特定等级的噪声时，其表现通常**次于**专门为该噪声强度训练的特化模型。
# 
# 通过使用特定噪声（即仅使用 sigma=25 的高斯噪声）对其进行继续训练（Fine-tuning），
# 网络结构无需改变，我们通过**改变训练加噪方式（从广泛区间缩小为特定值）**，
# 成功地使得模型针对 sigma=25 的噪声进行了适应和强化，从而有效提升了其在 Lena 和 Cameraman 上的客观 PSNR 与主观去噪效果。
