from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from pfhedge.nn import BlackScholes, MultiLayerPerceptron, Clamp


class NoTransactionBandNet(Module):
    def __init__(self, derivative):
        super().__init__()

        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def inputs(self):
        return self.delta.inputs() + ["prev_hedge"]

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.delta(input[..., :-1])
        out = self.mlp(torch.cat([input[..., :-1], delta], dim=-1))
        out = self.clamp(prev_hedge + out[..., [0]], prev_hedge + out[..., [1]])
        return out


class TwoHeadHedgeNet(nn.Module):
    def __init__(self, in_features=7, hidden_dim=32, n_layers=3, init_price=0.01):
        super().__init__()
        layers = [nn.Linear(in_features, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.shared_mlp = nn.Sequential(*layers)

        self.hedge_head = nn.Linear(hidden_dim, 1)
        self.price_head = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.price_head.bias.fill_(init_price)

    def forward(self, x):
        h = self.shared_mlp(x)
        hedge = self.hedge_head(h)
        price = self.price_head(h)
        return hedge, price
