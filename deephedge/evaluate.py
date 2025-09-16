# deephedge/evaluate.py
# Minimal helpers for pricing, expected utility, NT bands, trade stats, and plots.
# This file is designed to match the structure you already use in the notebook.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------- Pricing -------------------------------------- #

def indifference_price(
    hedger: Any,
    derivative: Any,
    *,
    n_paths: int = 50_000,
    init_state: Optional[Tuple] = None,
    **price_kwargs: Any,
) -> float:
    """
    Wrapper over pfhedge.nn.Hedger.price.
    Exactly what you already do in the notebook.

    Example in notebook:
        price = indifference_price(hedger, derivative, n_paths=10000, init_state=(S0,))
    """
    return float(hedger.price(derivative, n_paths=n_paths, init_state=init_state, **price_kwargs))


# -------------------------- Expected Utility -------------------------------- #

def expected_utility_from_wealth(
    W_T: torch.Tensor,
    *,
    utility: str = "exp",
    gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute expected utility from terminal wealth samples W_T.

    utility:
      - "exp":    U(w) = -exp(-gamma * w)
      - "crra":   U(w) = (w^(1-gamma) - 1) / (1-gamma)   (log utility as limit gamma->1)

    NOTE: This function only turns {W_T} into E[U(W_T)].
          You still produce W_T exactly as in your notebook (from your PnL/payoff routine).
    """
    if utility == "exp":
        return (-torch.exp(-gamma * W_T)).mean()
    elif utility == "crra":
        if abs(gamma - 1.0) < 1e-8:
            return torch.log(W_T.clamp_min(eps)).mean()
        return ((W_T.clamp_min(eps).pow(1.0 - gamma) - 1.0) / (1.0 - gamma)).mean()
    else:
        raise ValueError(f"Unknown utility: {utility!r}")


# ---------------------- Trade frequency / turnover -------------------------- #

def trade_frequency(
    hedge: torch.Tensor,
    *,
    count_when_nonzero: bool = True,
) -> torch.Tensor:
    """
    Number of trades per path.

    Input
      hedge: shape (n_paths, n_steps) or (n_paths, n_steps, 1)
    Returns
      freq:  shape (n_paths,) = number of times Δhedge != 0 along the path

    This mirrors your 'compute_trade_frequency_pfhedge' logic.
    """
    if hedge.dim() == 3:
        hedge = hedge[..., 0]
    dhedge = hedge[:, 1:] - hedge[:, :-1]
    if count_when_nonzero:
        trades = (dhedge.abs() > 0).sum(dim=1)
    else:
        # count sign flips only (stricter notion)
        trades = ((dhedge != 0).int().diff(dim=1).abs() > 0).sum(dim=1)
    return trades


def average_shares_traded(
    hedge: torch.Tensor,
) -> torch.Tensor:
    """
    Average absolute shares per trade (per path).

    If a path has 0 trades, returns 0 for that path.
    Mirrors your 'compute_avg_num_shares_traded_pfhedge' logic.
    """
    if hedge.dim() == 3:
        hedge = hedge[..., 0]
    dhedge = (hedge[:, 1:] - hedge[:, :-1]).abs()
    num_trades = (dhedge > 0).sum(dim=1).clamp_min(1)  # avoid divide-by-zero
    total_abs = dhedge.sum(dim=1)
    return total_abs / num_trades


# -------------------------- Empirical NT bands ------------------------------ #

@dataclass
class NTBands:
    lower: torch.Tensor  # shape (n_steps, n_bins)
    upper: torch.Tensor  # shape (n_steps, n_bins)
    s_grid: torch.Tensor # centers of price bins, shape (n_bins,)
    t_index: torch.Tensor  # 0..n_steps-1


def empirical_nt_bands_from_hedge(
    spot: torch.Tensor,
    hedge: torch.Tensor,
    *,
    n_bins: int = 41,
    quantile: float = 0.05,
) -> NTBands:
    """
    Empirical NT bands estimated from (spot, hedge) paths.

    Intuition:
      - Bucket S by bins at each time step.
      - For each (t, bin), look at distribution of Δhedge around that bucket.
      - The inner 1-2*quantile fraction defines an empirical 'inaction' region.
      - Return lower/upper band of hedge (NOT prices), i.e., where positions tend not to change.

    Inputs
      spot:  (n_paths, n_steps)
      hedge: (n_paths, n_steps) or (n_paths, n_steps, 1)

    Returns
      NTBands with tensors:
        lower, upper: (n_steps, n_bins)  -- in hedge units (shares)
        s_grid:       (n_bins,)          -- bin centers for spot
        t_index:      (n_steps,)
    """
    if hedge.dim() == 3:
        hedge = hedge[..., 0]

    n_paths, n_steps = spot.shape
    s_min = float(spot.min())
    s_max = float(spot.max())
    s_edges = torch.linspace(s_min, s_max, n_bins + 1, device=spot.device)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    # Δhedge per step
    dhedge = torch.zeros_like(hedge)
    dhedge[:, 1:] = hedge[:, 1:] - hedge[:, :-1]

    lowers = torch.full((n_steps, n_bins), float("nan"), device=spot.device)
    uppers = torch.full((n_steps, n_bins), float("nan"), device=spot.device)

    # loop over time (n_steps is typically << n_paths; OK)
    for t in range(n_steps):
        S_t = spot[:, t]
        H_t = hedge[:, t]
        dH_t = dhedge[:, t] if t > 0 else torch.zeros_like(H_t)

        # assign bins
        # bin index in [0, n_bins-1], points on the rightmost edge go to last bin
        idx = torch.bucketize(S_t, s_edges) - 1
        idx = idx.clamp(0, n_bins - 1)

        for b in range(n_bins):
            mask = idx == b
            if mask.sum() < 5:  # too few samples → skip
                continue
            Hvals = H_t[mask]
            dHvals = dH_t[mask].abs()

            # points with small changes are "inaction"
            # pick the 1-quantile mass around zero-change as the NT region
            thresh = torch.quantile(dHvals, quantile)
            stable = (dHvals <= thresh)
            if stable.sum() < 3:
                continue

            band_vals = Hvals[stable]
            low = torch.quantile(band_vals, 0.05)
            up  = torch.quantile(band_vals, 0.95)
            lowers[t, b] = low
            uppers[t, b] = up

    return NTBands(lower=lowers, upper=uppers, s_grid=s_centers, t_index=torch.arange(n_steps, device=spot.device))


# ------------------------------- Plots -------------------------------------- #

def plot_pnl_hist(
    pnl: torch.Tensor,
    *,
    bins: int = 50,
    title: str = "PnL distribution",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Quick histogram; keep your notebook cell tiny."""
    data = pnl.detach().cpu().numpy().ravel()
    ax = ax or plt.gca()
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Count")
    return ax


def plot_trade_frequency_per_time(
    hedge: torch.Tensor,
    *,
    title: str = "Trade activity over time",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot average |Δhedge| per time step (a simple activity proxy).
    """
    if hedge.dim() == 3:
        hedge = hedge[..., 0]
    dhedge = (hedge[:, 1:] - hedge[:, :-1]).abs()
    avg_per_t = dhedge.mean(dim=0).detach().cpu().numpy()
    ax = ax or plt.gca()
    ax.plot(avg_per_t)
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("avg |Δhedge|")
    return ax


def plot_nt_bands(
    bands: NTBands,
    *,
    t_indices: Optional[Sequence[int]] = None,
    title: str = "Empirical no-transaction bands",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot NT bands (lower/upper hedge) vs spot bins for selected time indices.
    """
    ax = ax or plt.gca()
    ts = t_indices or [0, len(bands.t_index)//2, len(bands.t_index)-1]
    for t in ts:
        lo = bands.lower[t].detach().cpu().numpy()
        up = bands.upper[t].detach().cpu().numpy()
        s  = bands.s_grid.detach().cpu().numpy()
        ax.plot(s, lo, alpha=0.8, label=f"t={t} lower")
        ax.plot(s, up, alpha=0.8, label=f"t={t} upper")
    ax.set_title(title)
    ax.set_xlabel("spot")
    ax.set_ylabel("hedge")
    ax.legend()
    return ax


# ------------------------------ Convenience --------------------------------- #

def summarize_trading_stats(
    hedge: torch.Tensor,
) -> Dict[str, float]:
    """
    Return a small dict you can print in one line.
    """
    freq = trade_frequency(hedge)               # (n_paths,)
    avg_shares = average_shares_traded(hedge)   # (n_paths,)
    out = {
        "trades_per_path_mean": float(freq.float().mean()),
        "trades_per_path_median": float(freq.median()),
        "avg_abs_shares_per_trade_mean": float(avg_shares.mean()),
        "avg_abs_shares_per_trade_median": float(avg_shares.median()),
    }
    return out
