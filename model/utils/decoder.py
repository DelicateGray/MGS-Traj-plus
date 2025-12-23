import torch.nn as nn

from config import *


class RTR_Decoder(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(RTR_Decoder, self).__init__()
        self.config = load_config("../../config.yaml")
        self.mlp_layer1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.mlp_layer2 = nn.Linear(embed_dim * self.config['common']['Th'], out_dim)

    def forward(self, x):
        B, N, L, D = x.shape
        output = self.mlp_layer1(x.reshape(B * N * L, D)).view(B, N, -1)
        output = self.mlp_layer2(output).view(B, N, -1, 2)
        return output


class STR_Decoder(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(STR_Decoder, self).__init__()
        self.mlp_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x):
        B, N, D = x.shape
        output = self.mlp_layer(x).view(B, N, -1, 2)
        return output


class PredDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PredDecoder, self).__init__()
        self.pred_net = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        B, D = x.shape
        pred = self.pred_net(x).view(B, -1, 2)

        return pred
