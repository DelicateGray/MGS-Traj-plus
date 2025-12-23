import torch
import torch.nn as nn
from typing import Any, Callable, Optional
from torchmetrics import Metric


class RMSE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
            self,
            # compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ) -> None:
        super(RMSE, self).__init__(
            # compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.pred, self.target = 0, 0
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred, target):
        self.pred = pred
        self.target = target
        with torch.no_grad():
            mse_func = nn.MSELoss()
            rmse = torch.sqrt(mse_func(self.pred, self.target))
            self.sum += rmse
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
