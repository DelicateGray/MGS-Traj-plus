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


class Pruner:
    """
    非结构化及结构化剪枝类，使用torch自带的nn.utils.prune工具进行简单的剪枝操作
    参数
        original_model_path：原始pytorch预测模型路径
        ts_pruning_args：时空网络剪枝参数
        social_pruning_args：交互网络剪枝参数
    """

    def __init__(self, original_model_path: str, ts_pruning_args: Dict, social_pruning_args: Dict):
        self.original_model_path = original_model_path
        self.ts_pruning_args = ts_pruning_args
        self.social_pruning_args = social_pruning_args
        self.pruning_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.log_dir = f"../log/pruning/{self.pruning_time}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.config = load_config("../config.yaml")
        self.B, self.N, self.Th, self.Tf, self.D = self.config['common']["B"], self.config['common']["N"], \
            self.config['common']['Th'], self.config['common']['Tf'], self.config['common']["D"]

    def get_ts_net_layers(self, ts_net: nn.Module) -> Dict:
        """
        获取时空网络待剪枝的网络层
        参数
            ts_net：时空网络的pytorch模型
        返回值：
            ts_att_layers：包含时空网络待剪枝网络层的列表
        """
        ts_att_layers = {}
        for i, layer in enumerate(ts_net.att_layer):
            ts_att_layers.update({i: [layer.self_att.q_linear, layer.self_att.k_linear, layer.self_att.v_linear]})
        return ts_att_layers

    def get_social_net_layers(self, social_net: nn.Module) -> Dict:
        """
        获取交互网络待剪枝的网络层
        参数
            social_net：交互网络的pytorch模型
        返回值：
            social_att_layers：包含交互网络待剪枝网络层的列表
        """
        social_att_layers = {}
        for i, layer in enumerate(social_net.mh_selfatt):
            social_att_layers.update({i: [layer.self_att.q_linear, layer.self_att.k_linear, layer.self_att.v_linear]})
        return social_att_layers

    def unstructured_pruning(self, layer: nn.Module, pruning_args: Dict) -> Union[Any, np.ndarray]:
        """
        非结构化剪枝函数
        参数
            layer：待剪枝的网络层
            pruning_args：剪枝参数
        返回值：
            mask：非结构化剪枝的掩码数组，维度与被剪枝的权重参数一致，其中被剪枝的神经元对应的mask元素为0,未被剪枝的为1
        """
        prune.l1_unstructured(layer, "weight", pruning_args["amount"])
        mask = layer.weight_mask.detach().cpu().numpy().astype(np.bool_)
        prune.remove(layer, 'weight')

        return mask

    def structured_pruning(self, layer: nn.Module, pruning_args: Dict) -> Union[Any, np.ndarray]:
        """
        结构化剪枝函数
        参数
            layer：待剪枝的网络层
            pruning_args：剪枝参数
        返回值：
            mask：结构化剪枝的掩码数组，维度与被剪枝的权重参数一致，其中被剪枝的神经元对应的mask元素为0,未被剪枝的为1
            score：被结构化剪枝的权重参数行的L1范数评分，代表该行神经元的重要性，此处没用到，所以赋值了全零数组用来占位
        """
        prune.ln_structured(layer, "weight", pruning_args["amount"], pruning_args["n"], pruning_args["dim"])
        mask = layer.weight_mask.detach().cpu().numpy().astype(np.bool_)
        prune.remove(layer, 'weight')
        score = np.array([0]).astype(np.float32)

        return mask, score

    def evaluate_model(self, model):
        """
        用来评价模型的大小，如参数量等
        参数
            model：待评价的模型
        """
        traj_example = torch.randn((self.B, self.N, self.Th, self.D), dtype=torch.float32)
        traj_mask_example = torch.zeros((self.B, self.N, self.Th), dtype=torch.bool)
        traj_mask_example[:, 6:, :] = 1
        summary(model, input_data=(traj_example, traj_mask_example))

    def do_pruning(self) -> None:
        """
        主函数，执行剪枝操作
        """
        model = PredNet(self.config)
        model.load_state_dict(torch.load(self.original_model_path).state_dict())
        model.eval()
        # 模型规模评估
        self.evaluate_model(model)
        ts_layers = self.get_ts_net_layers(model.ts_net)
        social_layers = self.get_social_net_layers(model.social_net)
        ts_para = {}
        social_para = {}
        # 时空网络剪枝
        for level, layers in ts_layers.items():
            layer_para = {"weight": [], "unstructured_mask": [], "structured_mask": [], "structured_score": []}
            for j, layer in enumerate(layers):
                weight = layer.weight.data.detach().cpu().numpy().astype(np.float32)
                print(f"ts_net_{level}{j} original weight num: {np.count_nonzero(weight)}")
                layer_para["weight"].append(weight)
                # 非结构化剪枝
                if self.ts_pruning_args["unstructured_pruing"]:
                    mask = self.unstructured_pruning(layer, self.ts_pruning_args)
                    layer_para["unstructured_mask"].append(mask)
                    print(f"ts_net_{level}{j} original weight num: {np.count_nonzero(mask)}")
                # 结构化剪枝
                if self.ts_pruning_args["structured_pruing"]:
                    mask, score = self.structured_pruning(layer, self.ts_pruning_args)
                    layer_para["structured_mask"].append(mask)
                    layer_para["structured_score"].append(score)
                    print(f"ts_net_{level}{j} original weight num: {np.count_nonzero(mask)}")
            print(
                "====================================================================================================")
            ts_para.update({level: layer_para})
        # 交互网络剪枝
        for level, layers in social_layers.items():
            layer_para = {"weight": [], "unstructured_mask": [], "structured_mask": [], "structured_score": []}
            for j, layer in enumerate(layers):
                weight = layer.weight.data.detach().cpu().numpy().astype(np.float32)
                print(f"social_net_{level}{j} original weight num: {np.count_nonzero(weight)}")
                layer_para["weight"].append(weight)
                # 非结构化剪枝
                if self.social_pruning_args["unstructured_pruing"]:
                    mask = self.unstructured_pruning(layer, self.social_pruning_args)
                    layer_para["unstructured_mask"].append(mask)
                    print(f"social_net_{level}{j} unstructured-pruned weight num: {np.count_nonzero(mask)}")
                # 结构化剪枝
                if self.social_pruning_args["structured_pruing"]:
                    mask, score = self.structured_pruning(layer, self.social_pruning_args)
                    layer_para["structured_mask"].append(mask)
                    layer_para["structured_score"].append(score)
                    print(f"social_net_{level}{j} structured-pruned weight num: {np.count_nonzero(mask)}")
            print(
                "====================================================================================================")
            social_para.update({level: layer_para})

        torch.save(model,
                   f"{self.log_dir}/pred_model_{self.ts_pruning_args['unstructured_pruing']}{self.ts_pruning_args['structured_pruing']}{self.social_pruning_args['unstructured_pruing']}{self.social_pruning_args['structured_pruing']}.pkl")
        joblib.dump(ts_para, f"{self.log_dir}/ts_para.pkl")
        joblib.dump(social_para, f"{self.log_dir}/social_para.pkl")
        with open(f"{self.log_dir}/pruning_args.txt", "w") as file:
            file.write(f"ts_pruning_args: {self.ts_pruning_args}")
            file.write(f"\r\nsocial_pruning_args: {self.social_pruning_args}")


if __name__ == '__main__':
    original_para_num = 3008178
    pruner = Pruner(
        original_model_path="../log/pred/11/2024-12-22-15_39_37_k_dwa/checkpoint/checkpoint_49_0.004433_0.004190_.pth",
        ts_pruning_args={"unstructured_pruing": 1, "structured_pruing": 0, "amount": 0.2, "n": 1, "dim": 0},
        social_pruning_args={"unstructured_pruing": 0, "structured_pruing": 1, "amount": 0.2, "n": 1, "dim": 0})
    pruner.do_pruning()
