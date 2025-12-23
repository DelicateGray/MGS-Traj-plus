import os
import time
from typing import Dict
import torch
import torch.nn as nn
from torchinfo import summary

from config import load_config
from model.pred_model import PredNet


def prune_and_rebuild_linear(layer: nn.Linear, amount: float, num_heads: int):
    # 计算保留维度（确保是num_heads的整数倍）
    n = layer.out_features
    num_kept = n - int(n * amount)
    num_kept = max((num_kept // num_heads) * num_heads, num_heads)

    # 构建修剪后的新层
    new_layer = nn.Linear(layer.in_features, num_kept, bias=layer.bias is not None)
    return new_layer, num_kept


class Pruner:
    def __init__(self, original_model_path: str, ts_pruning_args: Dict, social_pruning_args: Dict):
        self.original_model_path = original_model_path
        self.ts_pruning_args = ts_pruning_args
        self.social_pruning_args = social_pruning_args
        self.pruning_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.log_dir = f"../log/pruning/{self.pruning_time}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.config = load_config("../config.yaml")
        self.B, self.N, self.Th, self.Tf, self.D = (
            self.config['common']["B"],
            self.config['common']["N"],
            self.config['common']['Th'],
            self.config['common']['Tf'],
            self.config['common']["D"]
        )

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

        # 修改自注意力模块的前向传播
        def create_new_forward(self_att, num_heads, original_dim):
            def new_forward(x):
                B, L, _ = x.shape
                q = self_att.q_linear(x).view(B, L, num_heads, -1).transpose(1, 2)
                k = self_att.k_linear(x).view(B, L, num_heads, -1).transpose(1, 2)
                v = self_att.v_linear(x).view(B, L, num_heads, -1).transpose(1, 2)

                # 注意力计算（在低维空间）
                attn = (q @ k.transpose(-2, -1)) * self_att.scale
                attn = attn.softmax(dim=-1)
                attn_output = (attn @ v).transpose(1, 2).reshape(B, L, -1)

                # 维度恢复
                return self_att.recover_linear(attn_output)

            return new_forward

        # 时空网络剪枝
        for i, layer in enumerate(model.ts_net.att_layer):
            self_att = layer.self_att
            num_heads = self_att.num_heads
            original_dim = self_att.d_model  # 保存原始维度

            # 剪枝Q/K/V层（输出维度减少）
            for name in ['q_linear', 'k_linear', 'v_linear']:
                old_layer = getattr(self_att, name)
                new_layer, d_reduced = prune_and_rebuild_linear(
                    old_layer,
                    self.ts_pruning_args['amount'],
                    num_heads
                )
                # 复制权重（随机初始化后微调）
                setattr(self_att, name, new_layer)

            # 添加维度恢复层
            self_att.recover_linear = nn.Linear(d_reduced, original_dim)

            # 移除原始输出层（不再需要）
            if hasattr(self_att, 'out_linear'):
                del self_att.out_linear

            # 初始化恢复层（恒等初始化）
            with torch.no_grad():
                # 近似恒等初始化
                identity = torch.eye(original_dim, d_reduced)
                self_att.recover_linear.weight.copy_(identity)

                # 零初始化偏置
                if self_att.recover_linear.bias is not None:
                    self_att.recover_linear.bias.zero_()

            # 更新前向传播逻辑
            self_att.forward = create_new_forward(self_att, num_heads, original_dim)

        # 交互网络剪枝（同样逻辑）
        for i, layer in enumerate(model.social_net.mh_selfatt):
            self_att = layer.self_att
            num_heads = self_att.num_heads
            original_dim = self_att.d_model

            for name in ['q_linear', 'k_linear', 'v_linear']:
                old_layer = getattr(self_att, name)
                new_layer, d_reduced = prune_and_rebuild_linear(
                    old_layer,
                    self.social_pruning_args['amount'],
                    num_heads
                )
                setattr(self_att, name, new_layer)

            self_att.recover_linear = nn.Linear(d_reduced, original_dim)

            if hasattr(self_att, 'out_linear'):
                del self_att.out_linear

            with torch.no_grad():
                identity = torch.eye(original_dim, d_reduced)
                self_att.recover_linear.weight.copy_(identity)
                if self_att.recover_linear.bias is not None:
                    self_att.recover_linear.bias.zero_()

            self_att.forward = create_new_forward(self_att, num_heads, original_dim)

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