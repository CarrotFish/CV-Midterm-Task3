# configs/config_dice.py
from .config_ce import config as base_config

# 深拷贝基础配置，防止污染原字典
config = base_config.copy()

# 覆盖 Dice 实验特有的参数
config.update({
    "weights_save_path": "./weights/best_dice.pt",
    "loss_type": "dice",
    "lr": 5e-4,  # 单纯使用 Dice Loss 训练时，稍微降低一点学习率有助于稳定收敛
    "run_name": "Experiment_B_Dice_Only"
})