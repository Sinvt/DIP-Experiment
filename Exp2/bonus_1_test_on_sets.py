# %% [markdown]
# # 选做 1: DnCNN 在 Set12 和 Set68 测试集上的表现评估
#
# 使用 4 个预训练模型 (DnCNN-S-15, DnCNN-S-25, DnCNN-S-50, DnCNN-B)
# 分别在 Set12 (12张) 和 Set68 (68张) 上测试，
# 高斯噪声 sigma = 15, 25, 50 三档。
#
# 输出:
# - 每个模型在每个数据集/噪声等级下的平均 PSNR
# - 详细的逐图 PSNR 表
# - 可视化对比图

# %%
import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'Data')
RESULT_DIR = os.path.join(BASE_DIR, 'results', 'bonus1')
os.makedirs(RESULT_DIR, exist_ok=True)

# 导入模型
sys.path.insert(0, BASE_DIR)
from dncnn_model import DnCNN

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# %%
# =============================================
# 工具函数
# =============================================

def load_dncnn(weight_path, device):
    """加载 DnCNN 模型权重 (自动检测层数)"""
    sd = torch.load(weight_path, map_location=device, weights_only=False)
    new_sd = {k.replace('module.', ''): v for k, v in sd.items()}

    # 自动检测层数: 找最后一个 Conv 的索引
    # DnCNN-S: dncnn.47.weight (17层), DnCNN-B: dncnn.56.weight (20层)
    conv_indices = [int(k.split('.')[1]) for k in new_sd if k.endswith('.weight') and len(new_sd[k].shape) == 4]
    last_conv_idx = max(conv_indices)
    # 每层结构: Conv+BN+ReLU 占 3 个索引, 首层 Conv+ReLU 占 2, 末层 Conv 占 1
    # 层数 = (last_conv_idx - 2) / 3 + 2
    num_layers = (last_conv_idx - 2) // 3 + 2
    print(f"     检测到 {num_layers} 层结构 (最后Conv索引={last_conv_idx})")

    model = DnCNN(channels=1, num_of_layers=num_layers)
    model.load_state_dict(new_sd, strict=False)
    model = model.to(device)
    model.eval()
    return model


def dncnn_denoise(model, noisy_img, device):
    """DnCNN 去噪 (输入输出均为 [0,255] float64)"""
    with torch.no_grad():
        x = noisy_img / 255.0
        t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(t)
        result = out.squeeze().cpu().numpy()
        result = np.clip(result, 0, 1) * 255.0
    return result


def add_gaussian_noise(image, sigma):
    """手写高斯加噪 (纯 NumPy)"""
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy


def load_dataset(dataset_name):
    """加载数据集，返回 {name: image_array} 字典"""
    # Set12 解压后有嵌套目录: Data/Set12/Set12/
    dataset_path = os.path.join(DATA_DIR, dataset_name)
    # 检查是否有嵌套子目录
    nested = os.path.join(dataset_path, dataset_name)
    if os.path.isdir(nested):
        dataset_path = nested

    images = {}
    for fname in sorted(os.listdir(dataset_path)):
        if fname.lower().endswith(('.png', '.jpg', '.bmp', '.tif')):
            fpath = os.path.join(dataset_path, fname)
            img = np.array(Image.open(fpath).convert('L'), dtype=np.float64)
            name = os.path.splitext(fname)[0]
            images[name] = img
    return images


def pad_cjk(text, width, align='left'):
    """中英文混排对齐"""
    display_width = sum(2 if ord(c) > 127 else 1 for c in text)
    pad = max(0, width - display_width)
    if align == 'left':
        return text + ' ' * pad
    else:
        return ' ' * pad + text


# %%
# =============================================
# 加载所有模型
# =============================================
model_configs = {
    'DnCNN-S-15': os.path.join(LOG_DIR, 'DnCNN-S-15', 'net.pth'),
    'DnCNN-S-25': os.path.join(LOG_DIR, 'DnCNN-S-25', 'net.pth'),
    'DnCNN-S-50': os.path.join(LOG_DIR, 'DnCNN-S-50', 'net.pth'),
    'DnCNN-B':    os.path.join(LOG_DIR, 'DnCNN-B', 'net.pth'),
}

models = {}
for name, path in model_configs.items():
    if os.path.exists(path):
        models[name] = load_dncnn(path, device)
        print(f"[OK] {name} 加载成功")
    else:
        print(f"[SKIP] {name} 权重不存在: {path}")

# %%
# =============================================
# 加载数据集
# =============================================
datasets = {}
for ds_name in ['Set12', 'Set68']:
    ds = load_dataset(ds_name)
    datasets[ds_name] = ds
    print(f"[OK] {ds_name}: {len(ds)} 张图像")

# %%
# =============================================
# 测试: 所有模型 x 所有数据集 x 所有噪声等级
# =============================================
sigma_levels = [15, 25, 50]
np.random.seed(42)

# 结果存储: results[dataset][sigma][model] = list of psnr values
# noisy_psnr[dataset][sigma] = list of noisy psnr values
all_results = {}
noisy_psnr_results = {}

for ds_name, ds_images in datasets.items():
    all_results[ds_name] = {}
    noisy_psnr_results[ds_name] = {}

    for sigma in sigma_levels:
        all_results[ds_name][sigma] = {m: [] for m in models}
        noisy_psnr_results[ds_name][sigma] = []

        for img_name, clean in ds_images.items():
            noisy = add_gaussian_noise(clean, sigma)
            noisy_p = peak_signal_noise_ratio(clean, noisy, data_range=255.0)
            noisy_psnr_results[ds_name][sigma].append(noisy_p)

            for model_name, model in models.items():
                denoised = dncnn_denoise(model, noisy, device)
                p = peak_signal_noise_ratio(clean, denoised, data_range=255.0)
                all_results[ds_name][sigma][model_name].append(p)

        print(f"  {ds_name} | sigma={sigma}: 测试完成")

print("\n[OK] 所有测试完成!")

# %%
# =============================================
# 输出结果: 平均 PSNR 汇总表
# =============================================
print("\n" + "=" * 90)
print("  DnCNN 在 Set12 / Set68 上的平均 PSNR (dB)")
print("=" * 90)

for ds_name in ['Set12', 'Set68']:
    print(f"\n  {ds_name}")
    print(f"  {'-'*80}")

    # 表头
    header = f"  {pad_cjk('Sigma', 10)}{pad_cjk('Noisy', 10, 'right')}"
    for m in models:
        header += f"{m:>14}"
    print(header)
    print(f"  {'-'*80}")

    for sigma in sigma_levels:
        avg_noisy = np.mean(noisy_psnr_results[ds_name][sigma])
        row = f"  {pad_cjk(str(sigma), 10)}{avg_noisy:>10.2f}"
        for model_name in models:
            avg_p = np.mean(all_results[ds_name][sigma][model_name])
            row += f"{avg_p:>14.2f}"
        print(row)

print(f"\n{'='*90}")

# %%
# =============================================
# 可视化 1: 柱状图 -- 各模型在不同噪声等级下的平均 PSNR
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax_idx, ds_name in enumerate(['Set12', 'Set68']):
    ax = axes[ax_idx]
    x = np.arange(len(sigma_levels))
    width = 0.15
    n_models = len(models) + 1  # +1 for noisy baseline

    # 噪声基线
    noisy_avgs = [np.mean(noisy_psnr_results[ds_name][s]) for s in sigma_levels]
    bars = ax.bar(x - width * n_models / 2, noisy_avgs, width, label='Noisy', color='#e74c3c', alpha=0.8)
    for bar, val in zip(bars, noisy_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}',
                ha='center', va='bottom', fontsize=7)

    # 各模型
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12']
    for m_idx, (model_name, _) in enumerate(models.items()):
        avgs = [np.mean(all_results[ds_name][s][model_name]) for s in sigma_levels]
        offset = x - width * n_models / 2 + width * (m_idx + 1)
        bars = ax.bar(offset, avgs, width, label=model_name, color=colors[m_idx], alpha=0.85)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Sigma', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'{ds_name} -- 各模型平均 PSNR', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'sigma={s}' for s in sigma_levels])
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'average_psnr_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("已保存: average_psnr_comparison.png")

# %%
# =============================================
# 可视化 2: 热力图 -- 模型 x Sigma 的 PSNR 矩阵
# =============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, ds_name in enumerate(['Set12', 'Set68']):
    ax = axes[ax_idx]
    model_names = list(models.keys())
    matrix = np.zeros((len(model_names), len(sigma_levels)))

    for i, model_name in enumerate(model_names):
        for j, sigma in enumerate(sigma_levels):
            matrix[i, j] = np.mean(all_results[ds_name][sigma][model_name])

    im = ax.imshow(matrix, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(sigma_levels)))
    ax.set_xticklabels([f'sigma={s}' for s in sigma_levels])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_title(f'{ds_name} -- PSNR 热力图', fontsize=13, fontweight='bold')

    # 在每个格子里标注数值
    for i in range(len(model_names)):
        for j in range(len(sigma_levels)):
            val = matrix[i, j]
            color = 'white' if val < (matrix.max() + matrix.min()) / 2 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='PSNR (dB)')

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'psnr_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()
print("已保存: psnr_heatmap.png")

# %%
# =============================================
# 可视化 3: Set12 上选 3 张图的去噪效果主观对比 (sigma=25)
# =============================================
set12_images = datasets['Set12']
img_names = list(set12_images.keys())
# 选第 1, 5, 9 张（索引 0, 4, 8）展示
show_indices = [0, 4, 8] if len(img_names) >= 9 else list(range(min(3, len(img_names))))
show_names = [img_names[i] for i in show_indices]

sigma = 25
np.random.seed(42)

n_rows = len(show_names)
n_cols = 2 + len(models)  # 原图 + 噪声图 + 各模型
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
if n_rows == 1:
    axes = axes[np.newaxis, :]

fig.suptitle(f'Set12 去噪主观对比 (sigma={sigma})', fontsize=16, fontweight='bold', y=1.02)

for row, img_name in enumerate(show_names):
    clean = set12_images[img_name]
    noisy = add_gaussian_noise(clean, sigma)
    noisy_p = peak_signal_noise_ratio(clean, noisy, data_range=255.0)

    # 原图
    axes[row, 0].imshow(clean, cmap='gray', vmin=0, vmax=255)
    axes[row, 0].set_title(f'原图 ({img_name})', fontsize=10)
    axes[row, 0].axis('off')

    # 噪声图
    axes[row, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    axes[row, 1].set_title(f'Noisy\nPSNR={noisy_p:.2f}dB', fontsize=10)
    axes[row, 1].axis('off')

    # 各模型去噪结果
    for col, (model_name, model) in enumerate(models.items()):
        denoised = dncnn_denoise(model, noisy, device)
        p = peak_signal_noise_ratio(clean, denoised, data_range=255.0)
        axes[row, col + 2].imshow(denoised, cmap='gray', vmin=0, vmax=255)
        axes[row, col + 2].set_title(f'{model_name}\nPSNR={p:.2f}dB', fontsize=10)
        axes[row, col + 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'set12_subjective_sigma25.png'), dpi=150, bbox_inches='tight')
plt.show()
print("已保存: set12_subjective_sigma25.png")

# %%
# =============================================
# 详细逐图 PSNR 表 (Set12, sigma=25)
# =============================================
sigma = 25
print(f"\n{'='*80}")
print(f"  Set12 逐图 PSNR (dB) -- sigma={sigma}")
print(f"{'='*80}")

header = f"  {pad_cjk('Image', 12)}{pad_cjk('Noisy', 10, 'right')}"
for m in models:
    header += f"{m:>14}"
print(header)
print(f"  {'-'*72}")

np.random.seed(42)
for img_name, clean in set12_images.items():
    noisy = add_gaussian_noise(clean, sigma)
    noisy_p = peak_signal_noise_ratio(clean, noisy, data_range=255.0)
    row = f"  {pad_cjk(img_name, 12)}{noisy_p:>10.2f}"
    for model_name, model in models.items():
        denoised = dncnn_denoise(model, noisy, device)
        p = peak_signal_noise_ratio(clean, denoised, data_range=255.0)
        row += f"{p:>14.2f}"
    print(row)

# 平均值
print(f"  {'-'*72}")
avg_row = f"  {pad_cjk('Average', 12)}{np.mean(noisy_psnr_results['Set12'][sigma]):>10.2f}"
for model_name in models:
    avg_row += f"{np.mean(all_results['Set12'][sigma][model_name]):>14.2f}"
print(avg_row)

# %%
# =============================================
# 分析总结
# =============================================
print("\n" + "=" * 70)
print("  [*] 选做 1 分析总结")
print("=" * 70)
print("""
  关键发现:
  1. DnCNN-S-{sigma} 在对应 sigma 噪声下表现最佳 (matched case)
  2. DnCNN-B (盲去噪模型) 在所有噪声等级下都有不错的泛化能力
  3. 当噪声等级与训练不匹配时 (mismatched case), PSNR 会显著下降
     - 例: DnCNN-S-15 在 sigma=50 上效果差 (欠去噪)
     - 例: DnCNN-S-50 在 sigma=15 上可能过度去噪

  结论:
  - 特定噪声模型在匹配条件下性能最优
  - 盲去噪模型 DnCNN-B 提供了较好的鲁棒性与泛化能力的平衡
""")

print(f"\n结果保存在: {RESULT_DIR}")
for f in sorted(os.listdir(RESULT_DIR)):
    fsize = os.path.getsize(os.path.join(RESULT_DIR, f))
    print(f"  - {f} ({fsize/1024:.0f} KB)")
