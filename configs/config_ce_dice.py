# configs/config_ce_dice.py
from .config_ce import config as base_config

# 深拷贝基础配置，防止污染原字典
config = base_config.copy()

# 覆盖组合损失实验特有的参数
config.update({
    "weights_save_path": "./weights/best_ce_dice.pt",
    "loss_type": "ce_dice",
    "ce_weight": 0.5,     # 交叉熵权重 (你可以随时在这里调整比例，如 0.4)
    "dice_weight": 0.5,   # Dice 权重 (如 0.6)
    "epochs": 40,         # 组合损失有时需要更长的时间来寻找最优解，这里可以单独加长 epochs
    "run_name": "Experiment_C_Combined"
})