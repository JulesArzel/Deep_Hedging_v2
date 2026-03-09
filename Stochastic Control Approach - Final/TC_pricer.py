"""
Solver for the option pricing model of Davis-Panas-Zariphopoulou,
with extraction of the indirect utility and optimal trading policy.
"""

from time import time
import numpy as np
import cost_utils as cost


class TC_pricer:
    """
    Solver for the option pricing model of Davis-Panas-Zariphopoulou.
    Exponential utility: U(W) = -exp(-gamma * W).

    Parameters
    ----------
    Option_info : Option_param
        Contains (S0, K, T).
    Process_info : Diffusion_process
        Contains (r, mu, sig).
    cost_b : float
        Proportional buy cost (lambda).
    cost_s : float
        Proportional sell cost (mu).
    gamma : float
        Risk aversion coefficient.
    """

    def __init__(self, Option_info, Process_info, cost_b=0, cost_s=0, gamma=0.001):
        if Option_info.payoff == "put":
            raise ValueError("Not implemented for Put Options")

        self.r      = Process_info.r
        self.mu     = Process_info.mu
        self.sig    = Process_info.sig
        self.S0     = Option_info.S0
        self.K      = Option_info.K
        self.T      = Option_info.T
        self.cost_b = cost_b
        self.cost_s = cost_s
        self.gamma  = gamma

        self.Q_slices      = None
        self.action_slices = None

    def price(self, N=500, TYPE="writer", track_policy=False, Time=False):
        """
        Parameters
        ----------
        N : int
            Number of time steps.
        TYPE : str
            "writer" or "buyer".
        track_policy : bool
            Whether to record Q and optimal actions at each step.
        Time : bool
            Whether to return runtime.

        Returns
        -------
        price : float  (and elapsed if Time=True)
        """
        t_start = time()
        np.seterr(all="ignore")

        x0        = np.log(self.S0)
        T_vec, dt = np.linspace(0, self.T, N + 1, retstep=True)
        delta     = np.exp(-self.r * (self.T - T_vec))
        dx        = self.sig * np.sqrt(dt)
        dy        = dx
        M         = int(0.8 * np.floor(N / 2))
        y_grid    = np.linspace(-M * dy, M * dy, 2 * M + 1)
        N_y       = len(y_grid)
        med       = np.where(y_grid == 0)[0].item()

        def F(x_vals, k):
            return np.exp(self.gamma * (1 + self.cost_b) * np.exp(x_vals) * dy / delta[k])

        def G(x_vals, k):
            return np.exp(-self.gamma * (1 - self.cost_s) * np.exp(x_vals) * dy / delta[k])

        if track_policy:
            Q_store      = {}
            action_store = {}

        for portfolio in ["no_opt", TYPE]:
            xN = x0 + (self.mu - 0.5 * self.sig**2) * dt * N + (2 * np.arange(N + 1) - N) * dx

            if portfolio == "no_opt":
                Q = np.exp(-self.gamma * cost.no_opt(xN, y_grid, self.cost_b, self.cost_s))
            elif portfolio == "writer":
                Q = np.exp(-self.gamma * cost.writer(xN, y_grid, self.cost_b, self.cost_s, self.K))
            else:
                Q = np.exp(-self.gamma * cost.buyer(xN, y_grid, self.cost_b, self.cost_s, self.K))

            if track_policy:
                Q_k      = [None] * (N + 1)
                action_k = [None] * (N + 1)
                Q_k[N]      = Q
                action_k[N] = np.zeros_like(Q, dtype=int)

            for k in range(N - 1, -1, -1):
                Q_new = 0.5 * (Q[:-1, :] + Q[1:, :])

                xk    = x0 + (self.mu - 0.5 * self.sig**2) * dt * k + (2 * np.arange(k + 1) - k) * dx
                multF = F(xk, k).reshape(-1, 1)
                multG = G(xk, k).reshape(-1, 1)

                Buy          = Q_new.copy()
                Buy[:, :-1] = multF * Q_new[:, 1:]

                Sell         = Q_new.copy()
                Sell[:, 1:] = multG * Q_new[:, :-1]

                Q = np.minimum.reduce([Q_new, Buy, Sell])

                if track_policy:
                    A             = np.zeros_like(Q, dtype=int)
                    A[(Buy  < Q_new) & (Buy  < Sell)] = +1
                    A[(Sell < Q_new) & (Sell < Buy)]  = -1
                    Q_k[k]      = Q
                    action_k[k] = A

            if portfolio == "no_opt":
                Q_no  = Q[0, med]
            else:
                Q_yes = Q[0, med]

            if track_policy:
                Q_store[portfolio]      = Q_k
                action_store[portfolio] = action_k

        price = (delta[0] / self.gamma) * np.log(
            (Q_yes / Q_no) if TYPE == "writer" else (Q_no / Q_yes)
        )

        if track_policy:
            self.Q_slices      = Q_store
            self.action_slices = action_store

        if Time:
            return price, time() - t_start
        return price
