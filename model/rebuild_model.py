import torch.nn as nn

from model.utils.decoder import RTR_Decoder, STR_Decoder
from model.utils.encoder import SpatiotemporalNet, AggregationNet, SocialNet
from model.utils.model_utils import TrajMasking


class RebuildNet(nn.Module):
    """
        轨迹重构网络模型，RTR任务采用时空网络为编码器，STR任务采用和RTR任务共用的时空网络以及额外的聚合网络和交互网络作为编码器，两个任务的
        解码器都是多层感知机，但是参数相互独立
        参数
            config：参数配置
    """

    def __init__(self, config):
        super(RebuildNet, self).__init__()
        self.task = config["rebuild"]["task"]
        self.alpha = config["rebuild"]["alpha"]
        self.beta = config["rebuild"]["beta"]
        self.Trtr = int(self.alpha * config['common']["Th"])
        self.Tstr = int(self.beta * config['common']["Th"])
        self.ts_net = SpatiotemporalNet(
            config['common']["D"],
            config["common"]["De"],
            config["common"]["head_num"],
            config["rebuild"]["max_drop_prob"],
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
            config["rebuild"]["max_drop_prob"],
            config["common"]["socialnet_depth"],
        )
        self.rtr_decoder = RTR_Decoder(config["common"]["De"], 2 * self.Trtr)
        self.str_decoder = STR_Decoder(config["common"]["De"], 2 * self.Tstr)

    def rtr_forward(self, traj, traj_mask, target):
        """
            RTR网络前向函数，执行随机轨迹重构
            参数
                traj：历史轨迹，维度为(B,N,Th,D)
                traj_mask：历史轨迹的掩码，维度为(B,N,Th)
                target：重构目标，即历史轨迹，但是维度为(B,N,Th,2)，即只包括两个位置特征
            返回值
                rtr_rebuild：RTR重构轨迹，维度为(Nsum,Trtr,2)
                rtr_kinematic：RTR运动学轨迹，维度为(Nsum,Trtr,2)
                rtr_target：RTR目标轨迹，维度为(B,N,Th,2)
                rtr_target_mask：RTR目标轨迹的掩码，维度为(B,N,Th)
                rtr_meta_index：RTR元数据的索引,长度为Nsum
                注意：Nsum是当前批量中进行掩码的车辆的总数，且有0<Nsum<B*N,因为一个批量中很多车辆数据都是填充的0，用来使输入的轨迹序列等
                长，而不是每个场景都有N辆车
        """
        traj_masker = TrajMasking(traj_mask, alpha=self.alpha)
        rtr_traj_mask, rtr_rebuild_mask, rtr_target_mask, rtr_kinematic, rtr_meta_index = traj_masker.random_masking(traj)
        rtr_traj = self.ts_net(traj, rtr_traj_mask)
        rtr_rebuild = self.rtr_decoder(rtr_traj)
        rtr_rebuild = rtr_rebuild[~rtr_rebuild_mask].reshape(-1, self.Trtr, 2)
        rtr_target = target[~rtr_target_mask].reshape(-1, self.Trtr, 2)

        return rtr_rebuild, rtr_kinematic, rtr_target, rtr_target_mask, rtr_meta_index

    def str_forward(self, traj, traj_mask, target):
        """
            STR网络前向函数，执行交互轨迹重构
            参数
                traj：历史轨迹，维度为(B,N,Th,D)
                traj_mask：历史轨迹的掩码，维度为(B,N,Th)
                target：重构目标，即历史轨迹，但是维度为(B,N,Th,2)，即只包括两个位置特征
            返回值
                str_rebuild：STR重构轨迹，维度为(Nsum,Tstr,2)
                str_target：STR目标轨迹，维度为(B,N,Th,2)
                str_target_mask：STR目标轨迹的掩码，维度为(B,N,Th)
                str_meta_index：STR元数据的索引,长度为Nsum
                注意：Nsum同上
        """
        traj_masker = TrajMasking(traj_mask, beta=self.beta)
        str_traj_mask, str_rebuild_mask, str_target_mask, str_meta_index = traj_masker.social_masking()
        str_traj = self.ts_net(traj, str_traj_mask)
        str_traj, str_traj_mask = self.agg_net(str_traj, str_traj_mask)
        str_traj = self.social_net(str_traj, str_traj_mask)
        str_rebuild = self.str_decoder(str_traj)
        str_rebuild = str_rebuild[~str_rebuild_mask].reshape(-1, self.Tstr, 2)
        str_target = target[~str_target_mask].reshape(-1, self.Tstr, 2)

        return str_rebuild, str_target, str_target_mask, str_meta_index

    def forward(self, traj, traj_mask, traj_target):
        """
            轨迹重构网络前向函数，执行轨迹重构
            参数
                traj：历史轨迹，维度为(B,N,Th,D)
                traj_mask：历史轨迹的掩码，维度为(B,N,Th)
                traj_target：重构目标，即历史轨迹，但是维度为(B,N,Th,2)，即只包括两个位置特征
            返回值
                outputs：重构轨迹字典，其中"RTR"对应的值为随机重构轨迹，"STR"对应的值为交互重构轨迹，维度均为(B,N,Th,2)
        """
        outputs = {"RTR": [], "STR": []}
        if self.task["RTR"] == 1:
            outputs["RTR"] = list(self.rtr_forward(traj, traj_mask, traj_target))
        if self.task["STR"] == 1:
            outputs["STR"] = list(self.str_forward(traj, traj_mask, traj_target))
        return outputs
