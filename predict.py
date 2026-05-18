import torch
import matplotlib.pyplot as plt
from datasets.load_dataset import OxfordPetDataset
from datasets.preprocess import get_transforms
from models.unet import UNet
import numpy as np

config = 'ce_dice'

def visualize_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = f"weights/best_{config}.pt" # 替换为你的权重路径
    
    # 1. 组装模型
    model = UNet(in_channels=3, out_channels=3, features=[64, 128, 256, 512]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. 拿几张测试数据 (别打乱，拿前3张看看)
    val_transforms = get_transforms(mode='val', img_size=(256, 256))
    dataset = OxfordPetDataset("./data/oxford-iiit-pet", mode='val', transforms=val_transforms)
    
    num_samples = 3
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i] # 取出一个样本
            image_tensor = image.unsqueeze(0).to(device)
            
            # 推理
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            # 将 Tensor 转回可以 plt 打印的 HWC 格式 (并逆向归一化)
            # 假设你之前用的是 ImageNet 均值方差，这里只是粗略逆转换方便肉眼看
            img_vis = image.permute(1, 2, 0).cpu().numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
            
            mask_vis = mask.squeeze().cpu().numpy() if mask.ndim > 2 else mask.cpu().numpy()

            # 画图
            axes[i, 0].imshow(img_vis)
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_vis, cmap='viridis')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='viridis')
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"segmentation_results_{config}.png")
    print(f"✅ 可视化结果已保存为 segmentation_results_{config}.png")
    plt.show()

if __name__ == '__main__':
    visualize_prediction()