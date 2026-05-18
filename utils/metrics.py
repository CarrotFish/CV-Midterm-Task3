import numpy as np

class SegmentationMetrics:
    """计算图像分割评价指标 (如 mIoU) 的工具类"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        """利用 bincount 快速计算混淆矩阵"""
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], 
            minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        """
        在每个 Batch 传入真实的 mask 和预测的 mask 以更新混淆矩阵
        label_trues: numpy array, shape [B, H, W]
        label_preds: numpy array, shape [B, H, W]
        """
        for lt, lp in zip(label_trues, label_preds):
            self.hist += self._fast_hist(lt.flatten(), lp.flatten(), self.num_classes)

    def get_results(self):
        """计算各个类别的 IoU 以及最终的 mIoU"""
        # IoU = TP / (TP + FP + FN)
        # 对应混淆矩阵: 对角线元素 / (行和 + 列和 - 对角线元素)
        diag = np.diag(self.hist)
        
        # 加上 1e-10 防止分母为 0
        iou = diag / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - diag + 1e-10)
        miou = np.nanmean(iou)
        
        return iou, miou

    def reset(self):
        """每个 Epoch 结束后重置混淆矩阵"""
        self.hist = np.zeros((self.num_classes, self.num_classes))