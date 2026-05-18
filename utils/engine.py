import torch
from tqdm import tqdm
from .metrics import SegmentationMetrics

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger=None):
    """单轮训练逻辑"""
    model.train()
    total_loss = 0.0
    
    # 进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播与计算 Loss
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    
    # 记录 wandb
    if logger is not None:
        logger.log({'train/loss': avg_loss, 'epoch': epoch})
        
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes=3, epoch=None, logger=None):
    """验证集推理与评估逻辑"""
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(num_classes)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Eval]" if epoch else "[Evaluate]", dynamic_ncols=True)
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播 (不需要梯度)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # 预测结果: 取 channel 维度上概率最大的索引作为预测类别
        preds = torch.argmax(outputs, dim=1)
        
        # 更新混淆矩阵
        metrics.update(masks.cpu().numpy(), preds.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    iou_per_class, miou = metrics.get_results()

    # 记录 wandb
    if logger is not None:
        log_data = {'val/loss': avg_loss, 'val/mIoU': miou}
        if epoch is not None:
            log_data['epoch'] = epoch
        # 记录每一个类别的单独 IoU
        for i, iou in enumerate(iou_per_class):
            log_data[f'val/iou_class_{i}'] = iou
            
        logger.log(log_data)

    return avg_loss, miou