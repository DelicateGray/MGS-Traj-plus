import torch
from typing import Any, Callable, Optional
from torchmetrics import Metric


class InferenceTime(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
            self,
            # compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ) -> None:
        super(InferenceTime, self).__init__(
            # compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, inference_time):
        with torch.no_grad():
            self.sum += inference_time
            self.count += 1

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
