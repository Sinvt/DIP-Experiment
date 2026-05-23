"""下载/保存 Lena 和 Cameraman 测试图像到 images/ 目录"""
import os
import numpy as np

os.makedirs('images', exist_ok=True)

# === Cameraman 512x512 灰度 (来自 skimage 内置) ===
from skimage import data, io
from skimage.color import rgb2gray

cam = data.camera()  # 512x512 uint8 灰度
io.imsave('images/cameraman.png', cam)
print(f"[OK] Cameraman saved: {cam.shape}, dtype={cam.dtype}")

# === Lena 512x512 灰度 (从 Wikipedia 下载) ===
import urllib.request

lena_url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
tmp_path = 'images/lena_color_tmp.png'

print("[..] Downloading Lena image...")
urllib.request.urlretrieve(lena_url, tmp_path)

lena_color = io.imread(tmp_path)
lena_gray = (rgb2gray(lena_color) * 255).astype(np.uint8)
io.imsave('images/lena.png', lena_gray)
os.remove(tmp_path)
print(f"[OK] Lena saved: {lena_gray.shape}, dtype={lena_gray.dtype}")

print("\nAll images ready!")
