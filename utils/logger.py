import wandb

class WandbLogger:
    """Wandb 日志记录器的简单封装"""
    def __init__(self, project_name="CV_Midterm_Task3", config=None, run_name=None):
        """
        初始化 Wandb
        Args:
            project_name: 项目名称 (在 Wandb 后台显示的文件夹名)
            config: 存放超参数的字典 (会被保存到后台便于复现)
            run_name: 本次实验的名字 (如 'experiment_ce', 'experiment_dice')
        """
        self.run = wandb.init(
            project=project_name, 
            config=config, 
            name=run_name
        )

    def log(self, metrics_dict):
        """记录指标 (loss, mIoU 等)"""
        wandb.log(metrics_dict)

    def finish(self):
        """结束本次记录"""
        wandb.finish()