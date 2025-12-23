import os
import time
from typing import Union, Dict, Any
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchinfo import summary

from config import *
from model.pred_model import PredNet

# ✅ 新增：结构化剪枝 + 重建 linear 层
def prune_and_rebuild_linear(layer: nn.Linear, amount: float):
    prune.ln_structured(layer, name="weight", amount=amount, n=1, dim=0)
    mask = layer.weight_mask.detach().cpu().numpy()
    kept_idx = np.where(mask[:, 0] == 1)[0]
    new_out_features = len(kept_idx)
    in_features = layer.in_features
    new_weight = layer.weight.data[kept_idx, :]
    new_bias = layer.bias.data[kept_idx] if layer.bias is not None else None
    new_layer = nn.Linear(in_features, new_out_features, bias=(layer.bias is not None))
    new_layer.weight.data.copy_(new_weight)
    if new_bias is not None:
        new_layer.bias.data.copy_(new_bias)
    return new_layer, kept_idx

class Pruner:
    def __init__(self, original_model_path: str, ts_pruning_args: Dict, social_pruning_args: Dict):
        self.original_model_path = original_model_path
        self.ts_pruning_args = ts_pruning_args
        self.social_pruning_args = social_pruning_args
        self.pruning_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.log_dir = f"../log/pruning/{self.pruning_time}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.config = load_config("../config.yaml")
        self.B, self.N, self.Th, self.Tf, self.D = self.config['common']["B"], self.config['common']["N"], \
            self.config['common']['Th'], self.config['common']['Tf'], self.config['common']["D"]

    def evaluate_model(self, model):
        traj_example = torch.randn((self.B, self.N, self.Th, self.D), dtype=torch.float32)
        traj_mask_example = torch.zeros((self.B, self.N, self.Th), dtype=torch.bool)
        traj_mask_example[:, 6:, :] = 1
        summary(model, input_data=(traj_example, traj_mask_example))

    def do_pruning(self) -> nn.Module:
        model = PredNet(self.config)
        model.load_state_dict(torch.load(self.original_model_path).state_dict())
        model.eval()
        print("Before pruning:")
        self.evaluate_model(model)

        # ✅ 替换 ts_net 中的 q/k/v
        for i, layer in enumerate(model.ts_net.att_layer):
            for name in ['q_linear', 'k_linear', 'v_linear']:
                old_layer = getattr(layer.self_att, name)
                new_layer, _ = prune_and_rebuild_linear(old_layer, self.ts_pruning_args['amount'])
                setattr(layer.self_att, name, new_layer)

        # ✅ 替换 social_net 中的 q/k/v
        for i, layer in enumerate(model.social_net.mh_selfatt):
            for name in ['q_linear', 'k_linear', 'v_linear']:
                old_layer = getattr(layer.self_att, name)
                new_layer, _ = prune_and_rebuild_linear(old_layer, self.social_pruning_args['amount'])
                setattr(layer.self_att, name, new_layer)

        print("After pruning:")
        self.evaluate_model(model)

        # ✅ 保存模型
        torch.save(model, f"{self.log_dir}/pruned_model.pth")
        return model

if __name__ == '__main__':
    pruner = Pruner(
        original_model_path="../log/pred/11/2024-12-22-15_39_37_k_dwa/checkpoint/checkpoint_49_0.004433_0.004190_.pth",
        ts_pruning_args={"amount": 0.2},
        social_pruning_args={"amount": 0.2}
    )
    model = pruner.do_pruning()
