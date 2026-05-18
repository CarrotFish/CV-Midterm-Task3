# configs/config_ce.py

config = {
    # 🌟 运行模式 (现在也被整合进来了)
    "mode": "train",  # 默认在配置文件里设为 train
    
    # 基础与路径配置
    "data_root": "./data/oxford-iiit-pet",
    "weights_save_path": "./weights/best_ce.pt",
    
    # 数据集配置
    "img_size": 256,
    "batch_size": 16,
    "num_workers": 4,
    
    # 模型配置
    "in_channels": 3,
    "num_classes": 3,
    "features": [64, 128, 256, 512],
    
    # 损失函数配置
    "loss_type": "ce",
    "dice_smooth": 1e-5,
    "ce_weight": 0.5,
    "dice_weight": 0.5,
    
    # 训练配置
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    
    # Wandb 日志配置
    "project_name": "CV_Midterm_Task3",
    "run_name": "Experiment_A_CE_Only"
}