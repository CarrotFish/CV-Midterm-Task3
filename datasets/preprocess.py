import torch
import torchvision.transforms.functional as TF
import random
import numpy as np

class Compose:
    """组合多个联合变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class Resize:
    """调整图像和Mask的尺寸"""
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, image, mask):
        # 图像使用双线性插值 (平滑)
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        # Mask 必须使用最近邻插值，保证像素值只能是类别标签 (0, 1, 2)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

class RandomHorizontalFlip:
    """随机水平翻转 (图像和Mask必须同步)"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

class ToTensorAndNormalize:
    """将图像转为Tensor并归一化，Mask转为LongTensor"""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        # 图像转为 FloatTensor 并归一化到 [0, 1] 然后应用 ImageNet 标准化
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        
        # Mask 转为 Numpy 数组，再转为 PyTorch 的 LongTensor (不用归一化)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask

# 工厂函数，供外部调用
def get_transforms(mode='train', img_size=(256, 256)):
    if mode == 'train':
        return Compose([
            Resize(img_size),
            RandomHorizontalFlip(p=0.5),
            ToTensorAndNormalize()
        ])
    else:
        return Compose([
            Resize(img_size),
            ToTensorAndNormalize()
        ])