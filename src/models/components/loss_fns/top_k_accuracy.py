from typing import override

import torch
import torch.nn.functional as F

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class TopKAccuracy(BaseLossFn):
    def __init__(self, k_list: list[int] = [1, 5, 10]) -> None:
        super().__init__()
        self.k_list = k_list

    @override
    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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

            assert tmp_pred_greater_th.sum() <= k * len_batch, tmp_pred_greater_th.sum()
            assert tmp_label_greater_th.sum() <= k * len_batch, tmp_label_greater_th.sum()
            tmp_joint = tmp_pred_greater_th * tmp_label_greater_th
            n_present = torch.sum(tmp_joint, dim=1)  # sum per batch sample

            for n in n_present:
                assert n <= k, n_present

            top_k_acc = n_present.float() / k  # accuracy per batch sample
            accs[k] = top_k_acc.mean()

        return accs


if __name__ == "__main__":
    _ = TopKAccuracy()
