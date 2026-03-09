import numpy as np


def no_opt(x, y, cost_b, cost_s):
    """Terminal liquidation wealth for a portfolio with no option."""
    S = np.exp(x)[:, None]   # (Nx, 1)
    Y = y[None, :]            # (1,  Ny)
    return np.where(Y < 0, (1 + cost_b) * Y * S, (1 - cost_s) * Y * S)


def writer(x, y, cost_b, cost_s, K):
    """
    Terminal liquidation wealth for the option writer (short call).

    Exercise is triggered when (1+cost_b)*S > K (DPZ framework).
    On exercise the writer delivers 1 share and receives K.
    """
    S        = np.exp(x)[:, None]                    # (Nx, 1)
    Y        = y[None, :]                             # (1,  Ny)
    exercise = ((1 + cost_b) * S > K)                # (Nx, 1) → broadcasts to (Nx, Ny)
    Y_net    = Y - exercise.astype(float)             # Y-1 on exercise, Y otherwise
    stock    = np.where(Y_net < 0, (1 + cost_b) * Y_net * S,
                                   (1 - cost_s) * Y_net * S)
    return stock + exercise.astype(float) * K        # +K received on exercise


def buyer(x, y, cost_b, cost_s, K):
    """
    Terminal liquidation wealth for the option buyer (long call).

    Exercise is triggered when (1+cost_b)*S > K (DPZ framework).
    On exercise the buyer receives 1 share and pays K.
    """
    S        = np.exp(x)[:, None]
    Y        = y[None, :]
    exercise = ((1 + cost_b) * S > K)
    Y_net    = Y + exercise.astype(float)             # Y+1 on exercise, Y otherwise
    stock    = np.where(Y_net < 0, (1 + cost_b) * Y_net * S,
                                   (1 - cost_s) * Y_net * S)
    return stock - exercise.astype(float) * K        # -K paid on exercise


def callspread_writer(x, y, cost_b, cost_s, K1, K2):
    """Terminal liquidation wealth for the call spread writer."""
    S          = np.exp(x)[:, None]
    Y          = y[None, :]
    payoff     = np.maximum(S - K1, 0) - np.maximum(S - K2, 0)  # buyer's payoff
    stock      = np.where(Y < 0, (1 + cost_b) * Y * S, (1 - cost_s) * Y * S)
    return stock - payoff                            # writer pays the spread payoff


def callspread_buyer(x, y, cost_b, cost_s, K1, K2):
    """Terminal liquidation wealth for the call spread buyer."""
    S          = np.exp(x)[:, None]
    Y          = y[None, :]
    payoff     = np.maximum(S - K1, 0) - np.maximum(S - K2, 0)
    stock      = np.where(Y < 0, (1 + cost_b) * Y * S, (1 - cost_s) * Y * S)
    return stock + payoff                            # buyer receives the spread payoff
