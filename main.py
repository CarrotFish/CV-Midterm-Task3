import os
import argparse
import importlib
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# 导入你的模块
from datasets.load_dataset import OxfordPetDataset
from datasets.preprocess import get_transforms
from models.unet import UNet
from models.losses import DiceLoss, CEDiceLoss
from utils.engine import train_one_epoch, evaluate
from utils.logger import WandbLogger

def parse_args():
    parser = argparse.ArgumentParser(description="CV Midterm Task 3: Terminal > Config Priority")
    
    # ================= 1. 核心配置文件入口 =================
    parser.add_argument('--config', type=str, default='configs.config_ce', 
                        help="配置文件路径。默认 configs.config_ce")
    
    # ================= 2. 终端覆写参数 (默认全是 None) =================
    # 这样我们就能通过判断 value is not None 来确认用户是否在终端显式输入了该参数
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], help="覆盖运行模式")
    
    parser.add_argument('--data_root', type=str, help="覆盖数据集路径")
    parser.add_argument('--weights_save_path', type=str, help="覆盖权重保存/加载路径")
    
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    
    parser.add_argument('--in_channels', type=int)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--features', type=int, nargs='+')
    
    parser.add_argument('--loss_type', type=str, choices=['ce', 'dice', 'ce_dice'])
    parser.add_argument('--dice_smooth', type=float)
    parser.add_argument('--ce_weight', type=float)
    parser.add_argument('--dice_weight', type=float)
    
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epochs', type=int)
    
    parser.add_argument('--project_name', type=str)
    parser.add_argument('--run_name', type=str)

    return parser.parse_args()

def main():
    args = parse_args()
    
    # ==========================================
    # 🌟 核心逻辑：加载 Config 并应用终端覆盖
    # ==========================================
    # 1. 读取配置文件
    try:
        config_module = importlib.import_module(args.config)
        cfg = config_module.config
    except Exception as e:
        raise ImportError(f"❌ 无法加载配置文件 {args.config}，报错信息: {e}")

    print(f"📄 成功加载配置基座: [{args.config}]")

    # 2. 终端参数覆盖 Config (Terminal > Config)
    for key, value in vars(args).items():
        if key == 'config':
            continue
        # 只要终端输入了值 (不为 None)，就覆盖 cfg 里的默认值
        if value is not None:
            print(f"⌨️ 终端参数覆盖 -> {key}: {cfg.get(key, 'None')} => {value}")
            cfg[key] = value

    # 3. 将最终合并的字典重新赋给 args，方便后续代码调用 (比如 args.mode)
    for key, value in cfg.items():
        setattr(args, key, value)

    # ==========================================
    # 🚀 后续业务逻辑 (完全不用修改)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 启动任务 | 模式: {args.mode.upper()} | 设备: {device} | Loss: {args.loss_type.upper()}\n")

    # 1. 实例化模型
    model = UNet(
        in_channels=args.in_channels, 
        out_channels=args.num_classes, 
        features=args.features
    ).to(device)

    # 2. 实例化损失函数
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'dice':
        criterion = DiceLoss(smooth=args.dice_smooth)
    elif args.loss_type == 'ce_dice':
        criterion = CEDiceLoss(ce_weight=args.ce_weight, dice_weight=args.dice_weight)
    else:
        raise ValueError("不支持的 Loss 类型")

    # 3. 数据集与 DataLoader
    if args.mode == 'train':
        train_transforms = get_transforms(mode='train', img_size=(args.img_size, args.img_size))
        train_dataset = OxfordPetDataset(args.data_root, mode='train', transforms=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_transforms = get_transforms(mode='val', img_size=(args.img_size, args.img_size))
    val_dataset = OxfordPetDataset(args.data_root, mode='val', transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ==========================================
    # 🌟 训练模式
    # ==========================================
    if args.mode == 'train':
        os.makedirs(os.path.dirname(args.weights_save_path), exist_ok=True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 将最终合并后的 cfg 存入 Wandb，保证日志记录的是最真实的运行参数
        logger = WandbLogger(
            project_name=args.project_name, 
            config=cfg, 
            run_name=args.run_name
        )

        best_miou = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
            val_loss, val_miou = evaluate(model, val_loader, criterion, device, num_classes=args.num_classes, epoch=epoch, logger=logger)
            
            print(f"Epoch [{epoch}/{args.epochs}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), args.weights_save_path)
                print(f"✅ 保存当前最佳模型！(mIoU: {best_miou:.4f})")

        logger.finish()
        print(f"🎉 训练完成！最佳验证集 mIoU: {best_miou:.4f}")

    # ==========================================
    # 🌟 评估模式
    # ==========================================
    elif args.mode == 'eval':
        if not os.path.exists(args.weights_save_path):
            raise FileNotFoundError(f"❌ 找不到权重文件: {args.weights_save_path}")
            
        print(f"📥 加载权重: {args.weights_save_path}")
        model.load_state_dict(torch.load(args.weights_save_path, map_location=device))
        
        print("开始在验证集上进行评估...")
        val_loss, val_miou = evaluate(model, val_loader, criterion, device, num_classes=args.num_classes)
        print(f"\n📊 最终评估结果 -> Loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")

if __name__ == '__main__':
    main()