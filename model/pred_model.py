import torch
import torch.nn as nn

from model.utils.decoder import PredDecoder
from model.utils.encoder import SpatiotemporalNet, AggregationNet, SocialNet


class PredNet(nn.Module):
    """
        轨迹预测网络模型，编码器包含时空网络、聚合网络和交互网络， 解码器为多层感知机
        参数
            config：配置参数
    """

    def __init__(self, config):
        super(PredNet, self).__init__()
        self.task = config["pred"]["task"]
        self.pretrained_model_path = config["pred"]["pretrained_model_path"]
        self.ts_net = SpatiotemporalNet(
            config['common']["D"],
            config["common"]["De"],
            config["common"]["head_num"],
            config["pred"]["max_drop_prob"],
            config["common"]["stnet_depth"],
        )
        self.agg_net = AggregationNet(
            config["common"]["De"],
            config["common"]["De"],
            config["common"]["aggnet_depth"],
        )
        self.social_net = SocialNet(
            config["common"]["De"],
            config["common"]["head_num"],
            config["pred"]["max_drop_prob"],
            config["common"]["socialnet_depth"],
        )
        self.decoder = PredDecoder(
            config["common"]["De"],
            config['common']["Tf"] * 2,
        )
        # 如果预训练模型路径不为空，那么就对其进行微调，用预训练编码器的参数初始化预测模型的编码器
        if self.pretrained_model_path is not None:
            self.init_weights_from_model()

    def init_weights_from_model(self):
        pretrain_model = torch.load(self.pretrained_model_path)
        if self.task["RTR"]:
            self.ts_net.load_state_dict(pretrain_model.ts_net.state_dict())
        if self.task["STR"]:
            self.agg_net.load_state_dict(pretrain_model.agg_net.state_dict())
            self.social_net.load_state_dict(pretrain_model.social_net.state_dict())

    def forward(self, traj, traj_mask):
        """
            轨迹预测网络前向函数，执行轨迹预测
            参数
                traj：历史轨迹，维度为(B,N,Th,D)
                traj_mask：历史轨迹的掩码，维度为(B,N,Th)
            返回值
                pred：预测轨迹，维度为(B,N,Tf,2)
        """
        traj = self.ts_net(traj, traj_mask)
        traj, traj_mask = self.agg_net(traj, traj_mask)
        traj = self.social_net(traj, traj_mask)
        # 注意，轨迹预测任务中，仅对目标车辆（每个场景中的第一辆车）的交互特征进行解码
        pred = self.decoder(traj[:, 0, :])

        return pred
