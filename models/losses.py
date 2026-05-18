import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """手动实现的多分类 Dice Loss"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits 形状: [B, C, H, W]
        # targets 形状: [B, H, W]，且值为 0, 1, 2
        num_classes = logits.shape[1]
        
        # 1. 对网络输出进行 Softmax 激活，得到属于每个类的概率
        probs = F.softmax(logits, dim=1)
        
        # 2. 将真实的类别索引转化为 One-Hot 编码，并调整维度与 probs 匹配
        # [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 3. 计算每个类的交集和分母
        # 只在空间维度 (H, W) 上求和，保留 Batch 和 Channel 维度
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))
        
        # 4. 计算 Dice 分数
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 5. 求所有类的平均 Dice 损失 (1 - Dice_score)
        return 1. - dice_score.mean()

class CEDiceLoss(nn.Module):
    """组合损失 (Cross-Entropy Loss + Dice Loss)"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss