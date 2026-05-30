# %% [markdown]
# # 实验二：图像增强复原实验
#
# ## 实验目的
# 1. 掌握图像噪声的产生与特性（高斯噪声、椒盐噪声）
# 2. 掌握空间域滤波去噪方法（均值滤波、中值滤波）----**纯手写实现**
# 3. 了解基于深度学习的图像去噪方法（DnCNN）
# 4. 通过 PSNR 定量评估不同去噪方法的效果
#
# ## 重要说明
# [!] 本实验所有加噪操作和传统滤波操作均使用 **纯手写 NumPy** 实现，
# **不使用** cv2.blur / cv2.medianBlur / cv2.filter2D / skimage.util.random_noise
# 等任何封装好的加噪或去噪函数。

# %% [markdown]
# ---
# ## Part 0: 导入库与基本设置

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120

# 随机种子 (可复现)
np.random.seed(42)
torch.manual_seed(42)

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
IMAGE_DIR = os.path.join(BASE_DIR, 'images')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"运行设备: {device}")

# %% [markdown]
# ---
# ## Part 1: 手写加噪函数 & 噪声图像生成
#
# ### 高斯噪声
# 均值 $\mu=0$，标准差 $\sigma=25$（方差 $\sigma^2=625$，像素值 [0, 255] 空间）
#
# ### 椒盐噪声
# 噪声密度 $d=0.05$（5% 的像素被随机置为 0 或 255）

# %%
# =============================================
# 手写加噪函数 (纯 NumPy, 无任何库封装)
# =============================================

def add_gaussian_noise(image, mean=0, var=5):
    """
    手写高斯噪声添加

    原理: 对每个像素叠加一个服从 N(mean, var) 分布的随机值
    实现: 利用 np.random.normal 生成与图像同尺寸的噪声矩阵，
          直接与原图做矩阵加法，最后 clip 到 [0, 255]

    Parameters
    ----------
    image : np.ndarray, float64, [0, 255]
        输入灰度图像
    mean : float
        噪声均值
    var : float
        噪声方差 (sigma^2 = var, sigma = sqrt(var))

    Returns
    -------
    np.ndarray, float64, [0, 255]
        加噪后的图像
    """
    sigma = np.sqrt(var)
    h, w = image.shape
    # 生成高斯噪声矩阵 N(mean, sigma)
    noise = np.random.normal(loc=mean, scale=sigma, size=(h, w))
    # 叠加噪声
    noisy = image + noise
    # 截断到有效范围
    noisy = np.clip(noisy, 0, 255)
    return noisy


def add_salt_pepper_noise(image, amount=0.05):
    """
    手写椒盐噪声添加

    原理: 随机选取 amount 比例的像素，一半置为 255(盐/白)，一半置为 0(椒/黑)
    实现: 用 np.random.randint 生成随机坐标，直接对矩阵元素赋值

    Parameters
    ----------
    image : np.ndarray, float64, [0, 255]
        输入灰度图像
    amount : float
        噪声密度 (0~1)

    Returns
    -------
    np.ndarray, float64, [0, 255]
        加噪后的图像
    """
    noisy = image.copy()
    h, w = image.shape
    total_pixels = h * w
    num_noise = int(total_pixels * amount)

    # 盐噪声 (白点 = 255)
    num_salt = num_noise // 2
    salt_rows = np.random.randint(0, h, size=num_salt)
    salt_cols = np.random.randint(0, w, size=num_salt)
    noisy[salt_rows, salt_cols] = 255.0

    # 椒噪声 (黑点 = 0)
    num_pepper = num_noise // 2
    pepper_rows = np.random.randint(0, h, size=num_pepper)
    pepper_cols = np.random.randint(0, w, size=num_pepper)
    noisy[pepper_rows, pepper_cols] = 0.0

    return noisy


# =============================================
# PSNR 计算 (使用 skimage 库函数)
# =============================================

def calculate_psnr(original, processed, max_val=255.0):
    """使用 skimage 计算 PSNR (峰值信噪比)"""
    return peak_signal_noise_ratio(original, processed, data_range=max_val)


def pad_cjk(text, width, align='left'):
    """
    中英文混排对齐辅助函数
    中文字符显示宽度为 2, ASCII 为 1, 需手动补齐空格
    """
    display_width = sum(2 if ord(c) > 127 else 1 for c in text)
    pad = max(0, width - display_width)
    if align == 'left':
        return text + ' ' * pad
    else:  # right
        return ' ' * pad + text


print("[OK] 手写加噪函数定义完成, PSNR 使用 skimage 库函数")

# %%
# =============================================
# 读取图像 & 生成噪声图像
# =============================================

image_names = ['lena', 'cameraman']
images = {}
for name in image_names:
    img_path = os.path.join(IMAGE_DIR, f'{name}.png')
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float64)
    images[name] = img
    print(f"已加载 {name}.png, 尺寸: {img.shape}, 像素范围: [{img.min():.0f}, {img.max():.0f}]")

# 噪声配置
GAUSSIAN_MEAN = 0
GAUSSIAN_VAR = 625     # 方差=625, 即标准差 sigma=25
SP_AMOUNT = 0.05       # 椒盐密度=0.05

noisy_images = {}
for name in image_names:
    noisy_images[name] = {
        'gaussian': add_gaussian_noise(images[name], mean=GAUSSIAN_MEAN, var=GAUSSIAN_VAR),
        'salt_pepper': add_salt_pepper_noise(images[name], amount=SP_AMOUNT),
    }

# 噪声类型定义 (后续 Part 2/3/4 都会使用)
noise_types = ['gaussian', 'salt_pepper']
noise_labels = {'gaussian': '高斯噪声(u=0,sigma=25)', 'salt_pepper': '椒盐噪声(d=0.05)'}

for name in image_names:
    p_gauss = calculate_psnr(images[name], noisy_images[name]['gaussian'])
    p_sp = calculate_psnr(images[name], noisy_images[name]['salt_pepper'])
    print(f"  {name} | 高斯噪声(u=0,sigma=25) PSNR={p_gauss:.2f}dB | 椒盐噪声(d=0.05) PSNR={p_sp:.2f}dB")

# %% [markdown]
# ### Lena 原图与噪声图像对比

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Lena -- 原图与噪声图像对比', fontsize=16, fontweight='bold')

# 原图
axes[0].imshow(images['lena'], cmap='gray', vmin=0, vmax=255)
axes[0].set_title('原图', fontsize=13)
axes[0].axis('off')

# 高斯噪声
p_g = calculate_psnr(images['lena'], noisy_images['lena']['gaussian'])
axes[1].imshow(noisy_images['lena']['gaussian'], cmap='gray', vmin=0, vmax=255)
axes[1].set_title(f'高斯噪声 (u=0, sigma=25)\nPSNR = {p_g:.2f} dB', fontsize=12)
axes[1].axis('off')

# 椒盐噪声
p_sp = calculate_psnr(images['lena'], noisy_images['lena']['salt_pepper'])
axes[2].imshow(noisy_images['lena']['salt_pepper'], cmap='gray', vmin=0, vmax=255)
axes[2].set_title(f'椒盐噪声 (d=0.05)\nPSNR = {p_sp:.2f} dB', fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'lena_noise_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("已保存: lena_noise_comparison.png")

# %% [markdown]
# ---
# ## Part 2: 手写空间域滤波去噪
#
# ### [!] 核心: 纯手写滑动窗口实现
# - **均值滤波**: 对窗口内所有像素求算术平均
# - **中值滤波**: 对窗口内所有像素取中值
# - 核大小: 3x3, 5x5, 7x7
# - 边界处理: 边缘复制填充 (replicate padding)

# %%
# =============================================
# 手写均值滤波 (纯 NumPy 滑动窗口卷积)
# =============================================

def hand_mean_filter(image, ksize):
    """
    手写均值滤波 -- 2D 滑动窗口卷积实现

    原理: 对图像中每个像素, 取其 ksizexksize 邻域窗口内所有像素的算术平均值
          作为该像素的新值。等价于用全 1/(ksize^2) 的卷积核做 2D 卷积。

    实现: 对图像做边缘复制填充(replicate padding), 然后通过对卷积核
          每个偏移位置做整体矩阵平移累加, 实现滑动窗口求和, 最后除以
          窗口面积得到均值。

    Parameters
    ----------
    image : np.ndarray, float64
        输入图像
    ksize : int
        滤波器核大小 (必须为奇数)

    Returns
    -------
    np.ndarray, float64
        均值滤波后的图像
    """
    pad = ksize // 2
    h, w = image.shape
    # 边缘复制填充
    padded = np.pad(image, pad, mode='edge')

    # 滑动窗口求和: 遍历核的每个偏移位置, 累加对应位置的像素值
    result = np.zeros((h, w), dtype=np.float64)
    for di in range(ksize):
        for dj in range(ksize):
            result += padded[di:di + h, dj:dj + w]

    # 除以窗口面积得到均值
    result /= (ksize * ksize)
    return result


# =============================================
# 手写中值滤波 (纯 NumPy 滑动窗口)
# =============================================

def hand_median_filter(image, ksize):
    """
    手写中值滤波 -- 2D 滑动窗口实现

    原理: 对图像中每个像素, 取其 ksizexksize 邻域窗口内所有像素值,
          排序后取中间值作为该像素的新值。中值滤波是非线性滤波器,
          对椒盐噪声有很好的抑制效果。

    实现: 对图像做边缘复制填充, 将每个像素的 ksizexksize 邻域收集到
          一个三维数组中 (H, W, ksize^2), 然后沿第三维取中值。

    Parameters
    ----------
    image : np.ndarray, float64
        输入图像
    ksize : int
        滤波器核大小 (必须为奇数)

    Returns
    -------
    np.ndarray, float64
        中值滤波后的图像
    """
    pad = ksize // 2
    h, w = image.shape
    # 边缘复制填充
    padded = np.pad(image, pad, mode='edge')

    # 收集每个像素的邻域窗口值到三维数组
    window_size = ksize * ksize
    neighborhoods = np.zeros((h, w, window_size), dtype=np.float64)
    idx = 0
    for di in range(ksize):
        for dj in range(ksize):
            neighborhoods[:, :, idx] = padded[di:di + h, dj:dj + w]
            idx += 1

    # 沿邻域维度取中值
    result = np.median(neighborhoods, axis=2)
    return result


print("[OK] 手写均值滤波和中值滤波函数定义完成")
print("  实现方式: 纯 NumPy 滑动窗口, 无 cv2/skimage 调用")

# %%
# =============================================
# 对所有图像应用滤波去噪
# =============================================

kernel_sizes = [3, 5, 7]

# 存储滤波结果和 PSNR
filter_results = {}   # filter_results[img][noise][method][ksize] = filtered_image
psnr_table = {}       # psnr_table[img][noise][method][ksize] = psnr_value

for img_name in image_names:
    filter_results[img_name] = {}
    psnr_table[img_name] = {}

    for noise_type in noise_types:
        filter_results[img_name][noise_type] = {'mean': {}, 'median': {}}
        psnr_table[img_name][noise_type] = {'mean': {}, 'median': {}}
        noisy = noisy_images[img_name][noise_type]

        for k in kernel_sizes:
            # 手写均值滤波
            mean_filtered = hand_mean_filter(noisy, k)
            filter_results[img_name][noise_type]['mean'][k] = mean_filtered
            psnr_table[img_name][noise_type]['mean'][k] = calculate_psnr(images[img_name], mean_filtered)

            # 手写中值滤波
            median_filtered = hand_median_filter(noisy, k)
            filter_results[img_name][noise_type]['median'][k] = median_filtered
            psnr_table[img_name][noise_type]['median'][k] = calculate_psnr(images[img_name], median_filtered)

        print(f"  {img_name} - {noise_labels[noise_type]}: 滤波完成")

print("\n[OK] 所有传统滤波处理完成")

# %% [markdown]
# ### 传统滤波 -- 主观去噪结果（图像可视化）
#
# 对每张图像的每种噪声，展示不同滤波方法和核大小的去噪效果

# %%
for img_name in image_names:
    for noise_type in noise_types:
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'{img_name.capitalize()} -- {noise_labels[noise_type]} -- 传统滤波去噪主观效果对比',
                     fontsize=16, fontweight='bold')

        noisy = noisy_images[img_name][noise_type]
        noisy_p = calculate_psnr(images[img_name], noisy)

        # 第 0 行: 原图 + 噪声图
        axes[0, 0].imshow(images[img_name], cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('原图', fontsize=12)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title(f'噪声图\nPSNR={noisy_p:.2f}dB', fontsize=12)
        axes[0, 1].axis('off')

        axes[0, 2].axis('off')
        axes[0, 3].axis('off')

        # 第 1 行: 均值滤波 3x3, 5x5, 7x7
        axes[1, 0].imshow(noisy, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title(f'噪声图', fontsize=11)
        axes[1, 0].set_ylabel('均值滤波', fontsize=13, fontweight='bold')
        axes[1, 0].axis('off')
        for ci, k in enumerate(kernel_sizes):
            p = psnr_table[img_name][noise_type]['mean'][k]
            axes[1, ci + 1].imshow(filter_results[img_name][noise_type]['mean'][k],
                                    cmap='gray', vmin=0, vmax=255)
            axes[1, ci + 1].set_title(f'均值 {k}x{k}\nPSNR={p:.2f}dB', fontsize=11)
            axes[1, ci + 1].axis('off')

        # 第 2 行: 中值滤波 3x3, 5x5, 7x7
        axes[2, 0].imshow(noisy, cmap='gray', vmin=0, vmax=255)
        axes[2, 0].set_title(f'噪声图', fontsize=11)
        axes[2, 0].set_ylabel('中值滤波', fontsize=13, fontweight='bold')
        axes[2, 0].axis('off')
        for ci, k in enumerate(kernel_sizes):
            p = psnr_table[img_name][noise_type]['median'][k]
            axes[2, ci + 1].imshow(filter_results[img_name][noise_type]['median'][k],
                                    cmap='gray', vmin=0, vmax=255)
            axes[2, ci + 1].set_title(f'中值 {k}x{k}\nPSNR={p:.2f}dB', fontsize=11)
            axes[2, ci + 1].axis('off')

        plt.tight_layout()
        save_name = f'{img_name}_{noise_type}_filter_subjective.png'
        plt.savefig(os.path.join(RESULT_DIR, save_name), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"已保存: {save_name}")

# %% [markdown]
# ### 传统滤波 -- 客观结果 (PSNR 数值表)

# %%
for img_name in image_names:
    print(f"\n{'='*85}")
    print(f"  {img_name.upper()} -- 传统滤波 PSNR 结果 (dB)")
    print(f"{'='*85}")
    header = (f"  {pad_cjk('噪声类型', 24)}"
              f"{pad_cjk('噪声图', 8, 'right')}  "
              f"{pad_cjk('均值3x3', 8, 'right')} "
              f"{pad_cjk('均值5x5', 8, 'right')} "
              f"{pad_cjk('均值7x7', 8, 'right')}  "
              f"{pad_cjk('中值3x3', 8, 'right')} "
              f"{pad_cjk('中值5x5', 8, 'right')} "
              f"{pad_cjk('中值7x7', 8, 'right')}")
    print(header)
    print(f"  {'-'*83}")

    for noise_type in noise_types:
        noisy_p = calculate_psnr(images[img_name], noisy_images[img_name][noise_type])
        row = f"  {pad_cjk(noise_labels[noise_type], 24)}{noisy_p:>8.2f}  "
        for method in ['mean', 'median']:
            for k in kernel_sizes:
                row += f"{psnr_table[img_name][noise_type][method][k]:>9.2f}"
            row += " "
        print(row)

# %% [markdown]
# ---
# ## Part 3: DnCNN 深度学习去噪
#
# 使用预训练的 DnCNN 模型 (Zhang et al., 2017) 进行图像去噪。
#
# - 架构: 17 层 CNN (Conv+BN+ReLU), 残差学习策略
# - 预训练权重来自 ./logs 目录 (DnCNN-S-25 用于 sigma=25 高斯噪声)
# - 输入: 归一化到 [0,1] 的噪声图像
# - 输出: 去噪后的图像

# %%
import sys
sys.path.insert(0, BASE_DIR)
from dncnn_model import DnCNN


def load_dncnn_model(model_path, device):
    """加载预训练 DnCNN 模型，自动处理不同权重格式的 key 映射"""
    model = DnCNN(channels=1, num_of_layers=17)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # 自动 key 映射: 处理 'module.' 前缀 (DataParallel) 和不同属性名
    model_keys = list(model.state_dict().keys())
    loaded_keys = list(state_dict.keys())

    if model_keys == loaded_keys:
        # key 完全匹配, 直接加载
        model.load_state_dict(state_dict)
    else:
        # 尝试替换常见前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # 去掉 DataParallel 的 'module.' 前缀
            if new_key.startswith('module.'):
                new_key = new_key[7:]
            # 将 'dncnn.' 映射到 'model.' 或反之
            if new_key.startswith('dncnn.') and model_keys[0].startswith('model.'):
                new_key = 'model.' + new_key[6:]
            elif new_key.startswith('model.') and model_keys[0].startswith('dncnn.'):
                new_key = 'dncnn.' + new_key[6:]
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def dncnn_denoise(model, noisy_image_255, device):
    """
    使用 DnCNN 去噪

    Parameters
    ----------
    model : DnCNN
    noisy_image_255 : np.ndarray, float64, [0, 255]
    device : torch.device

    Returns
    -------
    np.ndarray, float64, [0, 255]
    """
    with torch.no_grad():
        # 归一化到 [0,1]
        noisy_01 = noisy_image_255 / 255.0
        tensor_in = torch.from_numpy(noisy_01).float().unsqueeze(0).unsqueeze(0).to(device)
        tensor_out = model(tensor_in)
        denoised_01 = tensor_out.squeeze().cpu().numpy()
        denoised_01 = np.clip(denoised_01, 0, 1)
        # 还原到 [0, 255]
        denoised_255 = denoised_01 * 255.0
    return denoised_255


# =============================================
# 加载 DnCNN 预训练权重 (来自 ./logs 目录)
# =============================================
LOG_DIR = os.path.join(BASE_DIR, 'logs')
model_path = os.path.join(LOG_DIR, 'DnCNN-S-25', 'net.pth')  # sigma=25 对应模型

dncnn_available = False
dncnn_results = {}
dncnn_psnr = {}

if os.path.exists(model_path):
    try:
        dncnn_model = load_dncnn_model(model_path, device)
        print(f"[OK] DnCNN-S-25 模型加载成功 (设备: {device})")
        print(f"     权重路径: {model_path}")
        dncnn_available = True

        # 对所有图像做 DnCNN 去噪
        for img_name in image_names:
            dncnn_results[img_name] = {}
            dncnn_psnr[img_name] = {}
            for noise_type in noise_types:
                noisy = noisy_images[img_name][noise_type]
                denoised = dncnn_denoise(dncnn_model, noisy, device)
                dncnn_results[img_name][noise_type] = denoised
                p = calculate_psnr(images[img_name], denoised)
                dncnn_psnr[img_name][noise_type] = p
                print(f"  {img_name} - {noise_labels[noise_type]}: DnCNN PSNR = {p:.2f} dB")
    except Exception as e:
        print(f"[FAIL] DnCNN 加载失败: {e}")
        import traceback; traceback.print_exc()
        print("  跳过 DnCNN 去噪部分")
else:
    print(f"[FAIL] DnCNN 权重文件不存在: {model_path}")
    print("  跳过 DnCNN 去噪部分")

# %% [markdown]
# ### DnCNN 去噪 -- 主观效果可视化

# %%
if dncnn_available:
    for img_name in image_names:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{img_name.capitalize()} -- DnCNN 去噪效果', fontsize=16, fontweight='bold')

        for row_idx, noise_type in enumerate(noise_types):
            noisy = noisy_images[img_name][noise_type]
            noisy_p = calculate_psnr(images[img_name], noisy)
            denoised = dncnn_results[img_name][noise_type]
            denoised_p = dncnn_psnr[img_name][noise_type]

            axes[row_idx, 0].imshow(images[img_name], cmap='gray', vmin=0, vmax=255)
            axes[row_idx, 0].set_title('原图', fontsize=12)
            axes[row_idx, 0].set_ylabel(noise_labels[noise_type], fontsize=12, fontweight='bold')
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, 1].set_title(f'噪声图\nPSNR={noisy_p:.2f}dB', fontsize=11)
            axes[row_idx, 1].axis('off')

            axes[row_idx, 2].imshow(denoised, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, 2].set_title(f'DnCNN 去噪\nPSNR={denoised_p:.2f}dB', fontsize=11)
            axes[row_idx, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, f'{img_name}_dncnn_subjective.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"已保存: {img_name}_dncnn_subjective.png")
else:
    print("DnCNN 不可用, 跳过可视化")

# %% [markdown]
# ---
# ## Part 4: 综合对比 -- 主观去噪结果 + 客观 PSNR 结果
#
# 将原图、噪声图、最优均值滤波、最优中值滤波、DnCNN 去噪结果并排展示

# %%
# =============================================
# 4.1 综合主观对比图
# =============================================
for img_name in image_names:
    n_cols = 5 if dncnn_available else 4
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    fig.suptitle(f'{img_name.capitalize()} -- 去噪方法综合主观对比',
                 fontsize=18, fontweight='bold')

    for row_idx, noise_type in enumerate(noise_types):
        noisy = noisy_images[img_name][noise_type]
        noisy_p = calculate_psnr(images[img_name], noisy)

        # 原图
        axes[row_idx, 0].imshow(images[img_name], cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 0].set_title('原图', fontsize=12)
        axes[row_idx, 0].axis('off')

        # 噪声图
        axes[row_idx, 1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 1].set_title(f'{noise_labels[noise_type]}\nPSNR={noisy_p:.2f}dB', fontsize=10)
        axes[row_idx, 1].axis('off')

        # 最优均值滤波
        best_k_mean = max(kernel_sizes,
                          key=lambda k: psnr_table[img_name][noise_type]['mean'][k])
        best_mean_p = psnr_table[img_name][noise_type]['mean'][best_k_mean]
        axes[row_idx, 2].imshow(filter_results[img_name][noise_type]['mean'][best_k_mean],
                                 cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 2].set_title(f'均值滤波 {best_k_mean}x{best_k_mean}\nPSNR={best_mean_p:.2f}dB',
                                    fontsize=10)
        axes[row_idx, 2].axis('off')

        # 最优中值滤波
        best_k_med = max(kernel_sizes,
                         key=lambda k: psnr_table[img_name][noise_type]['median'][k])
        best_med_p = psnr_table[img_name][noise_type]['median'][best_k_med]
        axes[row_idx, 3].imshow(filter_results[img_name][noise_type]['median'][best_k_med],
                                 cmap='gray', vmin=0, vmax=255)
        axes[row_idx, 3].set_title(f'中值滤波 {best_k_med}x{best_k_med}\nPSNR={best_med_p:.2f}dB',
                                    fontsize=10)
        axes[row_idx, 3].axis('off')

        # DnCNN
        if dncnn_available:
            axes[row_idx, 4].imshow(dncnn_results[img_name][noise_type],
                                     cmap='gray', vmin=0, vmax=255)
            axes[row_idx, 4].set_title(f'DnCNN\nPSNR={dncnn_psnr[img_name][noise_type]:.2f}dB',
                                        fontsize=10)
            axes[row_idx, 4].axis('off')

    axes[0, 0].set_ylabel('高斯噪声', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('椒盐噪声', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'{img_name}_comprehensive_subjective.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {img_name}_comprehensive_subjective.png")

# %%
# =============================================
# 4.2 综合客观结果 -- PSNR 汇总表
# =============================================
print("\n" + "=" * 95)
print("  [*] 综合 PSNR 对比表 (dB) -- 客观去噪结果")
print("=" * 95)

for img_name in image_names:
    print(f"\n  {'-'*88}")
    print(f"  {img_name.upper()}")
    print(f"  {'-'*88}")

    header = (f"  {pad_cjk('噪声类型', 24)}"
              f"{pad_cjk('噪声图', 8, 'right')}  "
              f"{pad_cjk('均值3x3', 8, 'right')} "
              f"{pad_cjk('均值5x5', 8, 'right')} "
              f"{pad_cjk('均值7x7', 8, 'right')}  "
              f"{pad_cjk('中值3x3', 8, 'right')} "
              f"{pad_cjk('中值5x5', 8, 'right')} "
              f"{pad_cjk('中值7x7', 8, 'right')}")
    if dncnn_available:
        header += f"  {'DnCNN':>8}"
    print(header)
    print(f"  {'-'*86}")

    for noise_type in noise_types:
        noisy_p = calculate_psnr(images[img_name], noisy_images[img_name][noise_type])
        row = f"  {pad_cjk(noise_labels[noise_type], 24)}{noisy_p:>8.2f}  "
        for method in ['mean', 'median']:
            for k in kernel_sizes:
                row += f"{psnr_table[img_name][noise_type][method][k]:>9.2f}"
            row += " "
        if dncnn_available:
            row += f"  {dncnn_psnr[img_name][noise_type]:>8.2f}"
        print(row)

print(f"\n{'='*95}")

# %%
# =============================================
# 4.3 PSNR 柱状图
# =============================================
for img_name in image_names:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'{img_name.capitalize()} -- 各去噪方法 PSNR 客观对比',
                 fontsize=16, fontweight='bold')

    for ax_idx, noise_type in enumerate(noise_types):
        methods_list = []
        psnr_vals = []

        # 噪声图
        noisy_p = calculate_psnr(images[img_name], noisy_images[img_name][noise_type])
        methods_list.append('噪声图')
        psnr_vals.append(noisy_p)

        # 均值滤波
        for k in kernel_sizes:
            methods_list.append(f'均值{k}x{k}')
            psnr_vals.append(psnr_table[img_name][noise_type]['mean'][k])

        # 中值滤波
        for k in kernel_sizes:
            methods_list.append(f'中值{k}x{k}')
            psnr_vals.append(psnr_table[img_name][noise_type]['median'][k])

        # DnCNN
        if dncnn_available:
            methods_list.append('DnCNN')
            psnr_vals.append(dncnn_psnr[img_name][noise_type])

        # 颜色: 噪声图红色, 均值绿色系, 中值蓝色系, DnCNN 金色
        colors = ['#e74c3c'] + ['#2ecc71', '#27ae60', '#1e8449'] + \
                 ['#3498db', '#2980b9', '#1f618d']
        if dncnn_available:
            colors.append('#f39c12')

        bars = axes[ax_idx].bar(range(len(methods_list)), psnr_vals, color=colors,
                                 edgecolor='white', linewidth=0.5)
        axes[ax_idx].set_xticks(range(len(methods_list)))
        axes[ax_idx].set_xticklabels(methods_list, rotation=45, ha='right')
        axes[ax_idx].set_title(noise_labels[noise_type], fontsize=14)
        axes[ax_idx].set_ylabel('PSNR (dB)', fontsize=12)
        axes[ax_idx].grid(axis='y', alpha=0.3)

        # 柱上标注数值
        for bar_obj, val in zip(bars, psnr_vals):
            axes[ax_idx].text(bar_obj.get_x() + bar_obj.get_width() / 2,
                              bar_obj.get_height() + 0.15,
                              f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f'{img_name}_psnr_bar_chart.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {img_name}_psnr_bar_chart.png")

# %% [markdown]
# ---
# ## Part 5: 选做部分 (附录B探究)
#
# 此部分预留，等待附录B内容后补充。

# %%
# =============================================
# 选做部分: 附录B 探究 (预留)
# =============================================
# TODO: 等待附录B内容后在此处补充
#
# 预留代码区域:
# 1. 探究内容 1 -- pass
# 2. 探究内容 2 -- pass
#
print("选做部分暂未实现，等待附录B内容。")

# %% [markdown]
# ---
# ## 实验小结
#
# 1. **高斯噪声去噪**:
#    - 均值滤波能平滑噪声但会导致图像整体模糊，细节丢失
#    - 中值滤波在保边方面优于均值滤波
#    - DnCNN 作为深度学习方法，在去噪同时能更好地保留纹理和细节
#
# 2. **椒盐噪声去噪**:
#    - 中值滤波是处理椒盐噪声的最佳传统方法（因为中值不受极端值影响）
#    - 均值滤波对椒盐噪声效果较差（极端值会拉偏均值）
#    - DnCNN 主要针对高斯噪声训练，对椒盐噪声的处理能力有限
#
# 3. **滤波器核大小的影响**:
#    - 核越大，去噪能力越强，但图像越模糊（分辨率损失）
#    - 需要在去噪效果与细节保留之间权衡

# %%
print("\n" + "=" * 60)
print("  [*] 实验二完成!")
print("=" * 60)
print(f"\n结果保存在: {RESULT_DIR}")
for f in sorted(os.listdir(RESULT_DIR)):
    fpath = os.path.join(RESULT_DIR, f)
    fsize = os.path.getsize(fpath)
    print(f"  - {f} ({fsize/1024:.0f} KB)")
