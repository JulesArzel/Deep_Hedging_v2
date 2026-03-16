"""
sc_functions.py
---------------
Helper functions for the Stochastic Control pricing notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------------
# DP solving & tracking
# ---------------------------------------------------------------------------

def run_and_track(pricer, N=500, TYPE="writer"):
    """
    Runs the DP with tracking turned on.
    Returns: price, Q_slices, A_slices, x_grids, y_grid
    """
    price = pricer.price(N=N, TYPE=TYPE, track_policy=True, Time=False)
    Q_slices = pricer.Q_slices[TYPE]
    A_slices = pricer.action_slices[TYPE]

    dt = pricer.T / N
    dx = pricer.sig * np.sqrt(dt)
    M  = int(0.8 * np.floor(N / 2))
    y_grid = np.linspace(-M * dx, M * dx, 2 * M + 1)

    x0 = np.log(pricer.S0)
    x_grids = []
    for k in range(N + 1):
        xk = x0 + (pricer.mu - 0.5 * pricer.sig**2) * (dt * k) \
             + (2 * np.arange(k + 1) - k) * dx
        x_grids.append(xk)

    return price, Q_slices, A_slices, x_grids, y_grid


# ---------------------------------------------------------------------------
# Policy-level analytics
# ---------------------------------------------------------------------------

def compute_trade_frequency(A_slices):
    """
    Returns the fraction of (t, x, y) states at which the agent trades (A != 0).
    Skips t=0 and t=N (boundary nodes).
    """
    N = len(A_slices) - 1
    traded = 0
    total  = 0
    for t in range(1, N):
        A = A_slices[t]
        traded += np.count_nonzero(A != 0)
        total  += A.size
    return traded / total


def compute_num_shares(A_slices, dy):
    """
    Average traded shares per decision-point:
      sum_{t,i} |A_{t,i}| * dy  /  total_nodes
    """
    N = len(A_slices) - 1
    total_moved = 0.0
    total_nodes = 0
    for t in range(1, N):
        A = A_slices[t]
        total_moved += np.abs(A).sum() * dy
        total_nodes += A.size
    return total_moved / total_nodes


def extract_dp_target(A_slice, y_grid):
    """
    Given the action map A_slice.shape = (len(x_k), len(y_grid)),
    returns the midpoint of the no-trade region at each x-node.
    """
    y_dp = np.empty(A_slice.shape[0])
    for j in range(A_slice.shape[0]):
        no_trade_inds = np.nonzero(A_slice[j] == 0)[0]
        y_lo = y_grid[no_trade_inds.min()]
        y_hi = y_grid[no_trade_inds.max()]
        y_dp[j] = 0.5 * (y_lo + y_hi)
    return y_dp


# ---------------------------------------------------------------------------
# Theoretical reference curves
# ---------------------------------------------------------------------------

def compute_phi_w(xk, K, r, mu, sig, T, t, gamma):
    """
    Compute the modified optimal holding:
      phi_w = delta_BS + exp(-r*tau) * (mu-r) / (gamma * S * sig^2)
    Returns (S_k, delta_bs, phi_w).
    """
    tau = T - t
    S_k = np.exp(xk)
    d1  = (np.log(S_k / K) + (r + 0.5 * sig**2) * tau) / (sig * np.sqrt(tau))
    delta_bs = norm.cdf(d1)
    delta_t  = np.exp(-r * tau)
    phi_w    = delta_bs + delta_t * (mu - r) / (gamma * S_k * sig**2)
    return S_k, delta_bs, phi_w


def phi_w_along_x(x_k, K, r, mu, sigma, T, t, gamma):
    """
    Same as compute_phi_w but returns only the phi_w array.
    """
    tau = T - t
    S_k = np.exp(x_k)
    d1  = (np.log(S_k / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    delta_bs  = norm.cdf(d1)
    disc      = np.exp(-r * tau)
    drift_adj = (mu - r) / (sigma**2)
    return delta_bs + disc / (gamma * S_k) * drift_adj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_policy(xk, y_grid, A_slice, title="Policy"):
    """Colour-map of the DP action on the (log-price, position) grid."""
    X, Y = np.meshgrid(xk, y_grid, indexing='xy')
    plt.figure(figsize=(6, 4))
    cmap = plt.get_cmap('RdYlBu', 3)
    plt.pcolormesh(X, Y, A_slice.T, cmap=cmap, vmin=-1, vmax=1, shading='auto')
    plt.colorbar(ticks=[-1, 0, 1], label='Action')
    plt.xlabel('log-price $x$')
    plt.ylabel('position $y$')
    plt.ylim((-0.5, 1.2))
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_policy_and_phi(xk, y_grid, A_slice, phi_w, title="Policy", K=1.0):
    """Action colour-map with the modified-delta overlay.

    Parameters
    ----------
    K : float
        Strike used to compute log-moneyness  x - log(K).
        Defaults to 1.0 (normalised problem).
    """
    lm = xk - np.log(K)          # log-moneyness axis
    X, Y = np.meshgrid(lm, y_grid, indexing='xy')
    plt.figure(figsize=(6, 4))
    cmap = plt.get_cmap('RdYlBu', 3)
    plt.pcolormesh(X, Y, A_slice.T, cmap=cmap, vmin=-1, vmax=1, shading='auto')
    plt.plot(lm, phi_w, 'k--',
             label=r'$\Delta_{BS}+\frac{\delta(T,s)(\alpha-r)}{\gamma S \sigma^2}$',
             linewidth=1.5)
    plt.colorbar(ticks=[-1, 0, 1], label='Action')
    plt.xlabel('log-moneyness $x - \\ln K$')
    plt.ylabel('position $y$')
    plt.ylim((-0.5, 1.2))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_wealth_and_policy(TC2, As, xgrids, ygrid,
                                S0, T, mu, sigma, r, cost_b, cost_s,
                                N, B0=0.0, seed=None):
    """
    Simulate one GBM price path and apply the DP-derived policy As.

    At t=0, allows instantaneous rebalancing to the optimal starting position
    (applying the k=0 policy repeatedly until the no-trade zone is reached).

    Returns: t_vec, S_path, y_path, B_path, W_path  (all length N+1)
    """
    dt = T / N
    t_vec = np.linspace(0, T, N + 1)
    if seed is not None:
        np.random.seed(seed)
    dW = np.sqrt(dt) * np.random.randn(N)
    W  = np.concatenate([[0], np.cumsum(dW)])
    S_path = S0 * np.exp((mu - 0.5 * sigma**2) * t_vec + sigma * W)

    dy = ygrid[1] - ygrid[0]
    Ny = len(ygrid)

    y_path = np.zeros(N + 1)
    B_path = np.zeros(N + 1)
    B_path[0] = B0

    # Instantaneous initial rebalancing at t=0
    idx_price_0 = 0
    idx_y = np.argmin(np.abs(ygrid - y_path[0]))
    for _ in range(Ny):
        a0 = As[0][idx_price_0, idx_y]
        if a0 == 0 or idx_y <= 0 or idx_y >= Ny - 1:
            break
        delta_y = a0 * dy
        y_path[0] += delta_y
        if a0 > 0:
            B_path[0] -= (1 + cost_b) * S_path[0] * delta_y
        else:
            B_path[0] += (1 - cost_s) * S_path[0] * abs(delta_y)
        idx_y = np.argmin(np.abs(ygrid - y_path[0]))

    # Main time loop
    for k in range(N):
        B_path[k] *= np.exp(r * dt)
        idx_price = np.argmin(np.abs(xgrids[k] - np.log(S_path[k])))
        idx_y     = np.argmin(np.abs(ygrid - y_path[k]))
        a = As[k][idx_price, idx_y]
        y_path[k + 1] = y_path[k] + a * dy
        if a > 0:
            B_path[k + 1] = B_path[k] - (1 + cost_b) * S_path[k] * (a * dy)
        elif a < 0:
            B_path[k + 1] = B_path[k] + (1 - cost_s) * S_path[k] * abs(a * dy)
        else:
            B_path[k + 1] = B_path[k]

    W_path = B_path + y_path * S_path
    return t_vec, S_path, y_path, B_path, W_path


def simulate_pnl(actions, pricer, payoff_func, M_paths=10_000, seed=None):
    """
    Monte-Carlo P&L for a single-leg hedge using the DP policy `actions`.

    Parameters
    ----------
    actions     : list of arrays (output of action_slices)
    pricer      : TC_pricer instance (provides S0, mu, sig, T, cost_b, cost_s)
    payoff_func : callable(S_T) -> float, short-option payoff to the hedger
    M_paths     : number of Monte-Carlo paths
    seed        : optional random seed

    Returns
    -------
    pnl : np.ndarray of shape (M_paths,)
    """
    if seed is not None:
        np.random.seed(seed)

    N_steps = len(actions) - 1
    dt = pricer.T / N_steps
    dx = pricer.sig * np.sqrt(dt)
    dy = dx
    x0 = np.log(pricer.S0)
    med = int(0.8 * np.floor(N_steps / 2))   # central y-index

    pnl = np.zeros(M_paths)
    for m in range(M_paths):
        S = np.empty(N_steps + 1)
        S[0] = pricer.S0
        for k in range(N_steps):
            dW = np.random.randn() * np.sqrt(dt)
            S[k + 1] = S[k] * np.exp(
                (pricer.mu - 0.5 * pricer.sig**2) * dt + pricer.sig * dW)

        cash  = 0.0
        y_pos = 0.0

        for k in range(N_steps):
            x_grid = x0 + (pricer.mu - 0.5 * pricer.sig**2) * dt * k \
                     + (2 * np.arange(k + 1) - k) * dx
            i = np.abs(x_grid - np.log(S[k])).argmin()
            delta_shares = actions[k][i, med] * dy

            if delta_shares > 0:
                cash -= S[k] * delta_shares * (1 + pricer.cost_b)
            elif delta_shares < 0:
                cash += -delta_shares * S[k] * (1 - pricer.cost_s)

            y_pos += delta_shares

        # Unwind at maturity
        if y_pos > 0:
            cash += y_pos * S[-1] * (1 - pricer.cost_s)
        elif y_pos < 0:
            cash += y_pos * S[-1] * (1 + pricer.cost_b)

        pnl[m] = cash - payoff_func(S[-1])

    return pnl


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))
