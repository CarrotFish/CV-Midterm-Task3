# 从零搭建与损失函数工程：图像分割模型的像素级训练
## 环境搭建
- 根据硬件环境安装Pytorch、torchvision
- 安装requirements
```bash
pip install wandb>=0.12.0 numpy>=1.21.0 Pillow>=8.0.0 tqdm>=4.62.0
```

## 进行训练
```bash
python main.py --config configs.config_ce
python main.py --config configs.config_dice
python main.py --config configs.config_ce_dice
```

## 权重下载
- [Best With CE Loss](https://ricacraft.com/downloads/pt/CV-Midterm-Task3/best_ce.pt)
- [Best With Dice Loss](https://ricacraft.com/downloads/pt/CV-Midterm-Task3/best_dice.pt)
- [Best With CE-Dice Loss](https://ricacraft.com/downloads/pt/CV-Midterm-Task3/best_ce_dice.pt)

训练环境采用 Ubuntu 26.04 + ROCm 7.13-preview，Intel Ultra 7 265K (64G RAM) + AMD Instinct MI50 32G