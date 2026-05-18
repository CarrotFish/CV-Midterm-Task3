import torch.nn as nn
from .unet import UNet
from .losses import DiceLoss, CEDiceLoss

def build_model(num_classes=3):
    """实例化 U-Net 模型"""
    # 强制从随机初始化开始，不使用预训练权重
    model = UNet(in_channels=3, out_channels=num_classes)
    return model

def build_loss(loss_type):
    """
    根据给定的 loss_type 实例化损失函数
    loss_type: 'ce', 'dice', 或 'ce_dice'
    """
    loss_type = loss_type.lower()
    if loss_type == 'ce':
        print("Using Cross-Entropy Loss")
        return nn.CrossEntropyLoss()
    elif loss_type == 'dice':
        print("Using Dice Loss")
        return DiceLoss()
    elif loss_type == 'ce_dice':
        print("Using Combined CE + Dice Loss")
        return CEDiceLoss()
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}。可选值为 'ce', 'dice', 'ce_dice'")