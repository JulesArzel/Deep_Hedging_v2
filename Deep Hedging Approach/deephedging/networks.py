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
    r"""Creates a multilayer perceptron.

    Number of input features is lazily determined.

    Args:
        in_features (int, optional): Size of each input sample.
            If ``None`` (default), the number of input features will be
            will be inferred from the ``input.shape[-1]`` after the first call to
            ``forward`` is done. Also, before the first ``forward`` parameters in the
            module are of :class:`torch.nn.UninitializedParameter` class.
        out_features (int, default=1): Size of each output sample.
        n_layers (int, default=4): The number of hidden layers.
        n_units (int or tuple[int], default=32): The number of units in
            each hidden layer.
            If ``tuple[int]``, it specifies different number of units for each layer.
        activation (torch.nn.Module, default=torch.nn.ReLU()):
            The activation module of the hidden layers.
            Default is a :class:`torch.nn.ReLU` instance.
        out_activation (torch.nn.Module, default=torch.nn.Identity()):
            The activation module of the output layer.
            Default is a :class:`torch.nn.Identity` instance.

    Shape:
        - Input: :math:`(N, *, H_{\text{in}})` where
          :math:`*` means any number of additional dimensions and
          :math:`H_{\text{in}}` is the number of input features.
        - Output: :math:`(N, *, H_{\text{out}})` where
          all but the last dimension are the same shape as the input and
          :math:`H_{\text{out}}` is the number of output features.

    Examples:

        By default, ``in_features`` is lazily determined:

        >>> import torch
        >>> from pfhedge.nn import MultiLayerPerceptron
        >>>
        >>> m = MultiLayerPerceptron()
        >>> m
        MultiLayerPerceptron(
          (0): LazyLinear(in_features=0, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ReLU()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): ReLU()
          (8): Linear(in_features=32, out_features=1, bias=True)
          (9): Identity()
        )
        >>> _ = m(torch.zeros(3, 2))
        >>> m
        MultiLayerPerceptron(
          (0): Linear(in_features=2, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ReLU()
          (6): Linear(in_features=32, out_features=32, bias=True)
          (7): ReLU()
          (8): Linear(in_features=32, out_features=1, bias=True)
          (9): Identity()
        )

        Specify different number of layers for each layer:

        >>> m = MultiLayerPerceptron(1, 1, n_layers=2, n_units=(16, 32))
        >>> m
        MultiLayerPerceptron(
          (0): Linear(in_features=1, out_features=16, bias=True)
          (1): ReLU()
          (2): Linear(in_features=16, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=1, bias=True)
          (5): Identity()
        )
    """

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
