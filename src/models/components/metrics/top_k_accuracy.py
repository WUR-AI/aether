from typing import Dict, override

import torch

from src.models.components.metrics.base_metrics import BaseMetrics


class TopKAccuracy(BaseMetrics):
    def __init__(self, k_list=None) -> None:
        super().__init__()
        self.k_list = k_list or [1, 5, 10]

    @override
    def forward(
        self,
        pred: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.float]:

        labels = batch.get("target")

        inds_sorted_preds = torch.argsort(
            pred, dim=1, descending=True
        )  # dim =1; sort along 2nd dimension (ie per sample)
        inds_sorted_target = torch.argsort(labels, dim=1, descending=True)
        len_batch = pred.shape[0]

        accs = {}

        for k in self.k_list:
            # Calculate top-k accuracy using tmp binary vectors that are 1 for the top-k predictions
            tmp_pred_greater_th = torch.zeros_like(pred)
            tmp_label_greater_th = torch.zeros_like(labels)
            for row in range(len_batch):
                tmp_pred_greater_th[row, inds_sorted_preds[row, :k]] = 1
                tmp_label_greater_th[row, inds_sorted_target[row, :k]] = 1

            tmp_joint = tmp_pred_greater_th * tmp_label_greater_th
            n_present = torch.sum(tmp_joint, dim=1)  # sum per batch sample
            top_k_acc = n_present.float() / k  # accuracy per batch sample
            accs[f"top_{k}_acc"] = top_k_acc.mean()
        return accs
