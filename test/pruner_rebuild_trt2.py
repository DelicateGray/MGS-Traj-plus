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


def prune_and_rebuild_linear(layer: nn.Linear, amount: float, num_heads: int):
    # 获取权重和偏置
    weight = layer.weight.data
    bias = layer.bias.data if layer.bias is not None else None

    # 计算每行L1范数
    norms = torch.norm(weight, p=1, dim=1)
    n = len(norms)

    # 计算保留数量（确保是num_heads的整数倍）
    num_kept = n - int(n * amount)
    num_kept = (num_kept // num_heads) * num_heads
    if num_kept < num_heads:
        num_kept = num_heads

    # 选择最重要的行
    _, indices = torch.topk(norms, k=num_kept, largest=True)
    kept_indices, _ = torch.sort(indices)  # 保持原始顺序

    # 构建新权重和偏置
    new_weight = weight[kept_indices, :]
    new_bias = bias[kept_indices] if bias is not None else None

    # 创建新层
    new_layer = nn.Linear(layer.in_features, num_kept, bias=layer.bias is not None)
    new_layer.weight.data.copy_(new_weight)
    if new_bias is not None:
        new_layer.bias.data.copy_(new_bias)

    return new_layer, kept_indices


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

        # 时空网络剪枝
        for i, layer in enumerate(model.ts_net.att_layer):
            self_att = layer.self_att
            num_heads = self_att.num_heads

            # 剪枝Q/K/V层
            for name in ['q_linear', 'k_linear', 'v_linear']:
                old_layer = getattr(self_att, name)
                new_layer, kept_idx = prune_and_rebuild_linear(
                    old_layer,
                    self.ts_pruning_args['amount'],
                    num_heads
                )
                setattr(self_att, name, new_layer)

            # 剪枝输出投影层
            old_out = self_att.out_linear
            new_out = nn.Linear(len(kept_idx), old_out.out_features, bias=old_out.bias is not None)
            new_weight = old_out.weight.data[:, kept_idx]
            new_out.weight.data.copy_(new_weight)
            if old_out.bias is not None:
                new_out.bias.data.copy_(old_out.bias.data)
            self_att.out_linear = new_out

            # 更新超参数
            self_att.d_model = len(kept_idx)
            self_att.head_dim = len(kept_idx) // num_heads

        # 交互网络剪枝（与上面逻辑相同）
        for i, layer in enumerate(model.social_net.mh_selfatt):
            self_att = layer
            num_heads = self_att.num_heads

            for name in ['q_linear', 'k_linear', 'v_linear']:
                old_layer = getattr(self_att, name)
                new_layer, kept_idx = prune_and_rebuild_linear(
                    old_layer,
                    self.social_pruning_args['amount'],
                    num_heads
                )
                setattr(self_att, name, new_layer)

            old_out = self_att.out_linear
            new_out = nn.Linear(len(kept_idx), old_out.out_features, bias=old_out.bias is not None)
            new_weight = old_out.weight.data[:, kept_idx]
            new_out.weight.data.copy_(new_weight)
            if old_out.bias is not None:
                new_out.bias.data.copy_(old_out.bias.data)
            self_att.out_linear = new_out

            self_att.d_model = len(kept_idx)
            self_att.head_dim = len(kept_idx) // num_heads

        print("After pruning:")
        self.evaluate_model(model)

        save_path = f"{self.log_dir}/pruned_model.pth"
        torch.save(model, save_path)
        print(f"Pruned model saved to {save_path}")
        return model


if __name__ == '__main__':
    pruner = Pruner(
        original_model_path="../log/pred/11/2024-12-22-15_39_37_k_dwa/checkpoint/checkpoint_49_0.004433_0.004190_.pth",
        ts_pruning_args={"amount": 0.2},
        social_pruning_args={"amount": 0.2}
    )
    model = pruner.do_pruning()
