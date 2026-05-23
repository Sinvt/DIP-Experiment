"""下载 Lena 灰度图像 - 使用多个备选源"""
import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import urllib.request

os.makedirs('images', exist_ok=True)

# 多个备选下载源
urls = [
    "https://raw.githubusercontent.com/mikolalysenko/lena/master/lena.png",
    "https://people.sc.fsu.edu/~jburkardt/data/png/lena.png",
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

downloaded = False
for i, url in enumerate(urls):
    try:
        print(f"[..] Trying source {i+1}: {url[:60]}...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            img_data = response.read()
        with open('images/lena_tmp.png', 'wb') as f:
            f.write(img_data)
        
        img = io.imread('images/lena_tmp.png')
        if img.ndim == 3:
            img = (rgb2gray(img) * 255).astype(np.uint8)
        
        # Resize to 512x512 if needed
        from skimage.transform import resize
        if img.shape != (512, 512):
            img = (resize(img, (512, 512), anti_aliasing=True) * 255).astype(np.uint8)
        
        io.imsave('images/lena.png', img)
        os.remove('images/lena_tmp.png')
        print(f"[OK] Lena saved: {img.shape}, dtype={img.dtype}")
        downloaded = True
        break
    except Exception as e:
        print(f"[FAIL] Source {i+1} failed: {e}")

if not downloaded:
    # 最后方案: 用 scipy.datasets.face() 替代或生成合成图
    print("[..] All URLs failed. Using skimage built-in 'grass' as fallback, trying another approach...")
    # 尝试直接从 Wikipedia 用不同方式
    try:
        lena_url = "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
        req = urllib.request.Request(lena_url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            img_data = response.read()
        with open('images/lena_tmp.png', 'wb') as f:
            f.write(img_data)
        img = io.imread('images/lena_tmp.png')
        if img.ndim == 3:
            img = (rgb2gray(img) * 255).astype(np.uint8)
        io.imsave('images/lena.png', img)
        os.remove('images/lena_tmp.png')
        print(f"[OK] Lena saved from Wikipedia: {img.shape}")
        downloaded = True
    except Exception as e2:
        print(f"[FAIL] Wikipedia also failed: {e2}")
        print("[INFO] Please manually download Lena image and save to images/lena.png")

if downloaded:
    print("\nAll images ready!")
