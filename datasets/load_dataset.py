import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, mode='train', transforms=None):
        """
        Args:
            root_dir (str): 数据集根目录 (包含 'images' 和 'annotations' 文件夹)
            mode (str): 'train' 或 'val'
            transforms (callable, optional): 图像和Mask的联合变换流水线
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'annotations', 'trimaps')
        
        # Oxford Pet 数据集的官方划分文件
        # 我们使用 trainval.txt 作为训练集，test.txt 作为验证/测试集
        split_file = 'trainval.txt' if mode == 'train' else 'test.txt'
        split_path = os.path.join(root_dir, 'annotations', split_file)
        
        self.filenames = []
        with open(split_path, 'r') as f:
            for line in f:
                # 每一行的第一个单词是去掉后缀的文件名
                filename = line.strip().split()[0]
                self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        img_path = os.path.join(self.image_dir, filename + '.jpg')
        mask_path = os.path.join(self.mask_dir, filename + '.png')
        
        # 读取图像并统一转为 RGB (极少数图片可能是灰度或RGBA)
        image = Image.open(img_path).convert("RGB")
        # 读取 Mask，保持其原始模式 (通常为单通道调色板模式或灰度)
        mask = Image.open(mask_path)
        
        # 应用数据增强与预处理
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        
        # ⚠️ 核心操作：Oxford Pet 的 Mask 值域为 1(前景), 2(背景), 3(边缘未分类)
        # 交叉熵损失要求类别从 0 开始，所以我们需要全体减 1 -> 0, 1, 2
        mask = mask - 1
        
        return image, mask