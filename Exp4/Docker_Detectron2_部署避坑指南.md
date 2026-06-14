# 🐳 Windows WSL2 + Docker 深度学习部署与避坑指南 (Detectron2)

这份文档是针对第一次接触 WSL + Docker 的新手整理的实战复盘。以配置 `shutu_rcnn` (Detectron2) 为例，记录了从文件解压到最终跑通模型的所有“潜规则”和避坑细节。

---

## 📂 阶段一：本地文件组装与“排雷”

这是整个实验最容易翻车的阶段。
> [!CAUTION]
> **核心原则：** 绝对不能有任何中文路径！所有的零件必须严丝合缝地放在指定的“抽屉”里，否则 Docker 挂载后找不到文件。

1. **路径排雷**：确保你用来存放实验文件的根目录（例如 `D:\Vscode\Project\DIP-Experiment\Exp4\Segmentation\`）是纯英文的。
2. **解压规则**：
   - 必须解压：`detectron2-main.zip`、`weights.zip`、`val2017.zip`、`annotations_trainval2017.zip`。
   - **坚决不解压**：`shutu_rcnn.tar`（这是打包好的 Docker 镜像集装箱，千万别碰它）。
3. **拼装目录结构**：像搭积木一样，把解压出来的零件塞进 `detectron2-main` 文件夹里。最终你的 Windows 目录树必须严格长这样：

```text
Segmentation/
├── shutu_rcnn.tar                 <-- 原封不动放在外面
└── detectron2-main/               <-- 核心工作区
    ├── weights/                   <-- 放预训练的 .pkl 模型文件
    └── datasets/
        └── coco/
            ├── val2017/           <-- 放 5000 张测试图片的文件夹
            └── annotations/       <-- 放 instances_val2017.json 的文件夹
```

---

## 🌉 阶段二：底层基建（WSL2 与 Docker 打通）

这一步是为了让 Windows 上的 Docker 能完美调用 Linux 内核和你的物理显卡（GPU）。

1. **确认 WSL2 状态**：在 Windows cmd 中输入 `wsl -l -v`，确认你的系统拥有 Ubuntu，且版本显示为 `2`。
2. **Docker 跨界授权**：打开 Docker Desktop -> `Settings` -> `Resources` -> `WSL integration`，打开 Ubuntu 的蓝色开关，点击 `Apply & restart`。
3. **安装显卡穿透工具**：进入 Ubuntu 终端（在 cmd 输入 `wsl` 即可进入），依次通过官方命令（`curl` 和 `apt-get`）安装 `nvidia-container-toolkit`，这是打通容器与物理 GPU 的核心桥梁。

---

## 🚢 阶段三：装载镜像与启动挂载

把 25GB 的环境“集装箱”装载进电脑，并把你的代码挂载进去。

1. **加载镜像**：在包含 `.tar` 文件的 Ubuntu 终端路径下运行以下命令，静静等待几十 GB 的环境导入：
   ```bash
   docker load -i shutu_rcnn.tar
   ```

2. **启动容器（魔法指令）**：用绝对路径映射，把 Windows 的代码目录挂载进容器的 `/mnt/data`，并启动交互终端。
   > [!TIP]
   > **路径转换秘籍**：在 WSL 中，Windows 的 `D:\` 盘会被映射为 `/mnt/d/`，斜杠也要换成 Linux 的正斜杠 `/`。
   
   ```bash
   docker run -it --name mask_rcnn -v /mnt/d/Vscode/Project/DIP-Experiment/Exp4/Segmentation/detectron2-main:/mnt/data --gpus all shutu_rcnn:latest /bin/bash
   ```

---

## 🚀 阶段四：执行核心推理与评估任务

当你看到终端前缀变成类似 `root@eddc3dc6932a:/#` 时，恭喜你，你已经身处 Docker 容器内部了！接下来的所有深度学习操作都在这里完成。

### 前置操作（每次进入容器都要做）
```bash
conda activate rcnn    # 激活预装好的深度学习 conda 环境
cd /mnt/data           # 进入我们挂载的代码根目录
```

### 任务 A：单张图像推理（完成实验要求 1 和 2）
- **动作**：使用 `demo.py` 脚本，对指定的图片进行目标检测和像素抠图。
- **💥 硬件避坑 (针对最新显卡)**：如果你的显卡架构极新（例如 RTX 5070 的 `sm_120` 架构），而 Docker 镜像由于年代久远（CUDA 版本老旧）无法识别该架构，强行跑 GPU 会直接报错崩溃。**妥协的解决办法是**：在命令末尾强行追加 `MODEL.DEVICE cpu`，用 CPU 熬十几秒出图，牺牲速度换取不报错。
- **命令示例（处理手机拍的自定义图片）**：
  ```bash
  cd demo
  mkdir results
  python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input input.png --output results --opts MODEL.WEIGHTS ../weights/R50_FPN_3x/model_final_f10217.pkl MODEL.DEVICE cpu
  ```
- **结果**：去 Windows 宿主机的 `results` 文件夹里，你就能直接看到带有高亮蒙版和检测框的成品图了（这正是路径挂载的神奇之处）。

### 任务 B：全集验证与 AP 指标测算（完成实验加分项 3.a）
- **动作**：使用 `train_net.py` 核心脚本，配合标注好的 JSON 标准答案文件，对 5000 张图进行批量跑分。
- **命令**（注意同样追加了 CPU 后缀，如果你是在老显卡上跑，可以删掉 `MODEL.DEVICE cpu`）：
  ```bash
  cd /mnt/data
  python tools/train_net.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --eval-only MODEL.WEIGHTS weights/R50_FPN_3x/model_final_f10217.pkl MODEL.DEVICE cpu
  ```
- **结果**：经过漫长的推理后，终端会输出一个详尽的矩阵表格。找到 `Task: segm` 下方的首列 `AP` 数值，这就是该模型在此验证集上的最终 **Mask-AP** 核心得分！
