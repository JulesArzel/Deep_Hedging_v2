from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential

from pfhedge.nn import BlackScholes
from pfhedge.nn import Clamp
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.features import LogMoneyness, TimeToMaturity, Volatility, PrevHedge


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
        width = self.mlp(input[..., :-1])

        min = delta - fn.leaky_relu(width[..., [0]])
        max = delta + fn.leaky_relu(width[..., [1]])

        return self.clamp(prev_hedge, min=min, max=max)
    
class TwoHeadHedgeNet(Module):
    def __init__(self, in_features=7, hidden_dim=32, n_layers=3, init_price=0.01):
        super().__init__()
        layers = [torch.nn.Linear(in_features, hidden_dim), torch.nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()]
        self.shared_mlp = torch.nn.Sequential(*layers)

        self.head_with = torch.nn.Linear(hidden_dim, 1)
        self.head_no = torch.nn.Linear(hidden_dim, 1)

        self.price = torch.nn.Parameter(torch.tensor([init_price], dtype=torch.float32))

    def forward(self, x):
        h = self.shared_mlp(x)            # x: (n_paths, in_features)
        hedge_with = self.head_with(h)    # (n_paths, 1)
        hedge_no = self.head_no(h)        # (n_paths, 1)
        return hedge_with, hedge_no

class MultiLayerPerceptron(Sequential):

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 1,
        n_layers: int = 4,
        n_units: Union[int, Sequence[int]] = 32,
        activation: Module = ReLU(),
        out_activation: Module = Identity(),
    ):
        n_units = (n_units,) * n_layers if isinstance(n_units, int) else n_units

        layers: List[Module] = []
        for i in range(n_layers):
            if i == 0:
                if in_features is None:
                    layers.append(LazyLinear(n_units[0]))
                else:
                    layers.append(Linear(in_features, n_units[0]))
            else:
                layers.append(Linear(n_units[i - 1], n_units[i]))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units[-1], out_features))
        layers.append(deepcopy(out_activation))

        super().__init__(*layers)

class LSTMHedger(Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(x)   
        out = self.fc(h)      
        return out

class GatedHedgeMLP(torch.nn.Module):
    """
    Two-head model:
      - gate head outputs p in (0,1)
      - action head outputs delta_y
    Output is y_t (hedge ratio), compatible with pfhedge Hedger.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes=(32, 32),
        action_scale: float = 2.0,
        gate_temperature: float = 1.0,
        prev_hedge_index: int = -1,  # assumes prev_hedge is last feature
    ):
        super().__init__()
        self.prev_hedge_index = prev_hedge_index
        self.action_scale = float(action_scale)
        self.gate_temperature = float(gate_temperature)

        layers = []
        d = in_features
        for h in hidden_sizes:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        self.backbone = torch.nn.Sequential(*layers)

        self.gate_head = torch.nn.Linear(d, 1)    # outputs logit
        self.action_head = torch.nn.Linear(d, 1)  # outputs unconstrained delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1, F) as provided by pfhedge features.
        returns y_t: (N, 1, 1)
        """
        # extract prev hedge y_{t^-}
        y_prev = x[..., self.prev_hedge_index:self.prev_hedge_index+1]  # (N,1,1)

        h = self.backbone(x)  # (N,1,H)

        # gate probability p_t in (0,1)
        logits = self.gate_head(h) / self.gate_temperature
        p = torch.sigmoid(logits)  # (N,1,1)

        # proposed action delta_y (bounded a bit for stability)
        delta_raw = self.action_head(h)
        delta_y = self.action_scale * torch.tanh(delta_raw)  # (N,1,1)

        y = y_prev + p * delta_y
        return y