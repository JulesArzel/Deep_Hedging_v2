from __future__ import annotations

import torch
from pfhedge.instruments import BaseDerivative, EuropeanOption, BrownianStock


class NegativeEuropeanOption(BaseDerivative):
    def __init__(self, underlier, strike=1.0, maturity=1.0):
        super().__init__()
        self.underlier = underlier
        self.strike = strike
        self.maturity = maturity

    def payoff_fn(self):
        spot = self.underlier.spot
        return -torch.clamp(spot[..., -1] - self.strike, min=0)

    def moneyness(self, time_step=None, log=False):
        spot = self.underlier.spot
        index = ... if time_step is None else [time_step]
        output = spot[..., index] / self.strike
        if log:
            return torch.log(output)
        return output

    def log_moneyness(self, time_step=None):
        return self.moneyness(time_step=time_step, log=True)

    def time_to_maturity(self, time_step=None):
        spot = self.underlier.spot
        n_paths, n_steps = spot.size()
        if time_step is None:
            t = torch.arange(n_steps).to(spot) * self.underlier.dt
            return (t[-1] - t).unsqueeze(0).expand(n_paths, -1)
        else:
            time = n_steps - (time_step % n_steps) - 1
            t = torch.tensor([[time]]).to(spot) * self.underlier.dt
            return t.expand(n_paths, -1)

    def volatility(self, time_step=None):
        return self.underlier.volatility(time_step=time_step)

    def inputs(self):
        return ["log_moneyness", "time_to_maturity", "volatility"]

    def init(self, spot: torch.Tensor, n_steps: int, dt: float):
        self.underlier.init(spot, n_steps, dt)
        return self

    def list(self, pricer):
        self.pricer = pricer
        return self

    def ul(self):
        return self.underlier


class CallSpread(BaseDerivative):
    def __init__(self, underlier, strike_long, strike_short, maturity=1.0):
        super().__init__()
        self.underlier = underlier
        self.strike_long = strike_long
        self.strike_short = strike_short
        self.maturity = maturity

    def payoff_fn(self):
        spot = self.underlier.spot
        return torch.clamp(spot[..., -1] - self.strike_long, min=0) - torch.clamp(spot[..., -1] - self.strike_short, min=0)

    def moneyness(self, time_step=None, log=False):
        spot = self.underlier.spot
        index = ... if time_step is None else [time_step]
        # For a spread, define moneyness relative to lower strike
        output = spot[..., index] / self.strike_long
        if log:
            return torch.log(output)
        return output

    def log_moneyness(self, time_step=None):
        return self.moneyness(time_step=time_step, log=True)

    def time_to_maturity(self, time_step=None):
        spot = self.underlier.spot
        n_paths, n_steps = spot.size()
        if time_step is None:
            t = torch.arange(n_steps).to(spot) * self.underlier.dt
            return (t[-1] - t).unsqueeze(0).expand(n_paths, -1)
        else:
            time = n_steps - (time_step % n_steps) - 1
            t = torch.tensor([[time]]).to(spot) * self.underlier.dt
            return t.expand(n_paths, -1)

    def volatility(self, time_step=None):
        return self.underlier.volatility(time_step=time_step)

    def inputs(self):
        return ["log_moneyness", "time_to_maturity", "volatility"]

    def init(self, spot: torch.Tensor, n_steps: int, dt: float):
        self.underlier.init(spot, n_steps, dt)
        return self

    def list(self, pricer):
        self.pricer = pricer
        return self

    def ul(self):
        return self.underlier
