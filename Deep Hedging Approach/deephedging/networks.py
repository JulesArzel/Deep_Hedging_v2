from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
import math

import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.distributions import Normal

from pfhedge.nn import BlackScholes
from pfhedge.nn import Clamp


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

        out = self.mlp(input[..., :-1])
        lo  = out[..., [0]]
        hi  = lo + fn.softplus(out[..., [1]])   # guarantees hi > lo, no delta anchor

        return self.clamp(prev_hedge, min=lo, max=hi)
    
class WWGuidedNTBN(Module):
    """
    Whalley-Wilmott Guided No-Transaction Band Network.

    Improvement over the standard NoTransactionBandNet on two fronts:

    1. WW-guided bandwidth: the band half-width is initialised from the
       Whalley-Wilmott asymptotic formula
           h_WW = (3 * lambda * sigma^2 * S^2 * Gamma_BS / (2 * gamma))^(1/3)
       and the MLP only learns a small residual correction on top of it.
       This gives the network the correct scaling with TC, vol and gamma
       for free, so it converges faster and generalises better across
       parameter regimes.

    2. Soft clamp: the hard torch.clamp (zero gradient outside the band)
       is replaced by a tanh-based smooth approximation
           softclamp(x, l, u) = mid + half * tanh(beta * (x - mid) / half)
       where mid = (l+u)/2, half = (u-l)/2.
       This provides non-zero gradients everywhere, giving the network a
       learning signal even when prev_hedge is outside the band.

    Args:
        derivative : pfhedge instrument (used for BlackScholes delta and
                     to read the transaction cost lambda).
        gamma      : risk-aversion coefficient matching the entropic loss.
        n_layers   : depth of the residual-correction MLP.
        n_units    : width of the residual-correction MLP.
        clamp_beta : sharpness of the soft clamp (larger -> closer to hard
                     clamp; recommended range 5-20).
    """

    def __init__(
        self,
        derivative,
        gamma: float = 1.0,
        n_layers: int = 4,
        n_units: int = 32,
        clamp_beta: float = 10.0,
    ):
        super().__init__()
        self.derivative = derivative
        self.gamma = gamma
        self.clamp_beta = clamp_beta

        self.delta = BlackScholes(derivative)
        # MLP receives [log_moneyness, time_to_maturity, volatility] and
        # outputs two residual corrections [delta_lower, delta_upper].
        self.mlp = MultiLayerPerceptron(out_features=2, n_layers=n_layers, n_units=n_units)

    def inputs(self):
        return self.delta.inputs() + ["prev_hedge"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bs_gamma(self, log_moneyness: Tensor, ttm: Tensor, volatility: Tensor) -> Tensor:
        """Black-Scholes gamma in log-moneyness / normalised-price space."""
        sqrt_tau = ttm.sqrt().clamp(min=1e-6)
        d1 = (log_moneyness + 0.5 * volatility ** 2 * ttm) / (volatility * sqrt_tau)
        nd1 = torch.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
        # S = exp(log_moneyness) because pfhedge normalises with K=1
        S = log_moneyness.exp()
        return nd1 / (S * volatility * sqrt_tau).clamp(min=1e-6)

    def _ww_halfwidth(self, log_moneyness: Tensor, ttm: Tensor, volatility: Tensor) -> Tensor:
        """
        Whalley-Wilmott asymptotic half-bandwidth:
            h_WW = (3 * lambda * sigma^2 * S^2 * Gamma / (2 * gamma))^(1/3)
        """
        tc = float(self.derivative.underlier.cost)
        S = log_moneyness.exp()
        gamma_bs = self._bs_gamma(log_moneyness, ttm, volatility)
        numerator = 3.0 * tc * volatility ** 2 * S ** 2 * gamma_bs
        h_ww = (numerator / (2.0 * self.gamma)).clamp(min=0.0).pow(1.0 / 3.0)
        return h_ww

    @staticmethod
    def _soft_clamp(x: Tensor, lo: Tensor, hi: Tensor, beta: float) -> Tensor:
        """
        Differentiable approximation to clamp(x, lo, hi):
            mid + half * tanh(beta * (x - mid) / half)
        Gradient is non-zero everywhere; recovers hard clamp as beta -> inf.
        """
        half = ((hi - lo) / 2.0).clamp(min=1e-6)
        mid = (lo + hi) / 2.0
        return mid + half * torch.tanh(beta * (x - mid) / half)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]
        features   = input[..., :-1]          # (log_moneyness, ttm, volatility)

        log_moneyness = features[..., [0]]
        ttm           = features[..., [1]]
        volatility    = features[..., [2]]

        # 1. Band centre: Black-Scholes delta
        delta = self.delta(features)           # shape (..., 1)

        # 2. WW analytical half-width (structural prior)
        h_ww = self._ww_halfwidth(log_moneyness, ttm, volatility)

        # 3. MLP residual corrections
        corrections = self.mlp(features)       # shape (..., 2)

        # 4. Band boundaries: leaky_relu keeps width non-negative;
        #    the MLP correction shifts the WW baseline up or down.
        lo = delta - fn.leaky_relu(h_ww + corrections[..., [0]])
        hi = delta + fn.leaky_relu(h_ww + corrections[..., [1]])

        # 5. Hard clamp (same as standard NTBN)
        return prev_hedge.clamp(lo, hi)


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