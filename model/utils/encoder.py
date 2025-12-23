import torch
import torch.nn as nn

from config import *
from model.utils.model_utils import (
    AggregationBlock,
    PositionEmbedding,
    TransformerEncoderBlock,
    repadding,
)


class SpatiotemporalNet(nn.Module):
    def __init__(self, insize, out_size, head_num, max_drop_prob=0.2, att_depth=2):
        super(SpatiotemporalNet, self).__init__()
        self.proj_layer = nn.Linear(insize, out_size)
        self.config = load_config("../../config.yaml")
        self.pos_embed = PositionEmbedding(self.config['common']['Lmax'], out_size)
        drop_prob = [
            prob.item() for prob in torch.linspace(0, max_drop_prob, att_depth)
        ]
        self.att_layer = nn.ModuleList(
            [
                TransformerEncoderBlock(out_size, head_num, drop_prob[i])
                for i in range(att_depth)
            ]
        )

    def forward(self, x, x_mask):
        B, N, L, D = x.shape
        x = self.proj_layer(x.reshape(B * N * L, D)).reshape(B, N, L, -1)
        x_embs = []
        for i in range(B):
            scene_emb = self.pos_embed(x[i, :, :, :])
            scene_mask = x_mask[i, :, :]
            for attlayer in self.att_layer:
                scene_emb = attlayer(scene_emb, scene_mask)
            x_embs.append(scene_emb.unsqueeze(0))
        x_embs = torch.cat(x_embs, dim=0).to(x.device)

        return x_embs


class AggregationNet(nn.Module):
    def __init__(self, insize, outsize, depth=1):
        super(AggregationNet, self).__init__()
        self.agg = AggregationBlock(insize, outsize, depth)

    def forward(self, x, x_mask):
        B, N, L, D = x.shape
        output = self.agg(x.view(B * N, L, D), x_mask.view(B * N, L)).view(B, N, -1)
        output_mask = torch.all(x_mask, dim=-1).view(B, N)
        output = repadding(output, output_mask, 0)

        return output, output_mask


class SocialNet(nn.Module):
    def __init__(self, hidden_size, head_num, max_drop_prob, depth=3):
        super(SocialNet, self).__init__()
        drop_prob = [prob.item() for prob in torch.linspace(0, max_drop_prob, depth)]
        self.mh_selfatt = nn.ModuleList(
            [
                TransformerEncoderBlock(hidden_size, head_num, drop_prob[i])
                for i in range(depth)
            ]
        )

    def forward(self, x, x_mask):
        for att_layer in self.mh_selfatt:
            x = att_layer(x, x_mask)
        output = x
        return output
