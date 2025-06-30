"""
Solver for the option pricing model of Davis-Panas-Zariphopoulou,
with extraction of the indirect utility and optimal trading policy.
"""

from time import time
import numpy as np
import numpy.matlib
import cost_utils as cost

class TC_pricer_CS:
    """
    Solver for the option pricing model of Davis-Panas-Zariphopoulou.
    """

    def __init__(self, Option_info, Process_info, K1, K2, cost_b=0, cost_s=0, gamma=0.001):
        """
        Option_info:  Option_param containing (S0,K,T)
        Process_info: Diffusion_process containing (r,mu,sig)
        cost_b:       (lambda) BUY cost
        cost_s:       (mu)     SELL cost
        gamma:        risk aversion coefficient
        """
        if Option_info.payoff == "put":
            raise ValueError("Not implemented for Put Options")

        self.r      = Process_info.r
        self.mu     = Process_info.mu
        self.sig    = Process_info.sig
        self.S0     = Option_info.S0
        self.K1 = K1
        self.K2 = K2
        self.T      = Option_info.T
        self.cost_b = cost_b
        self.cost_s = cost_s
        self.gamma  = gamma

        # Will be filled if track_policy=True
        self.Q_slices      = None
        self.action_slices = None

    def price(self, N=500, TYPE="callspread", track_policy=False, Time=False):
        """
        N = number of time steps
        TYPE = "writer" or "buyer" or "callspread"
        track_policy = whether to record the entire Q and optimal actions
        Time = whether to return runtime
        """
        t_start = time()
        np.seterr(all="ignore")

        # Precompute grids & steps
        x0       = np.log(self.S0)
        T_vec, dt = np.linspace(0, self.T, N+1, retstep=True)
        delta    = np.exp(-self.r * (self.T - T_vec))
        dx       = self.sig * np.sqrt(dt)
        dy       = dx
        M        = int(0.8*np.floor(N / 2))
        y_grid   = np.linspace(-M*dy, M*dy, 2*M+1)
        N_y      = len(y_grid)
        med      = np.where(y_grid == 0)[0].item()

        # Transaction multipliers
        def F(x_vals, dy, k):
            return np.exp(self.gamma * (1 + self.cost_b) * np.exp(x_vals) * dy / delta[k])

        def G(x_vals, dy, k):
            return np.exp(-self.gamma * (1 - self.cost_s) * np.exp(x_vals) * dy / delta[k])

        # Storage if tracking
        if track_policy:
            Q_store      = {}
            action_store = {}

        # Loop over baseline and writer/buyer
        for portfolio in ["no_opt", TYPE]:
            # Terminal layer k = N
            xN = x0 + (self.mu - 0.5*self.sig**2)*dt*N + (2*np.arange(N+1) - N)*dx
            if portfolio == "no_opt":
                Q = np.exp(-self.gamma * cost.no_opt(xN, y_grid, self.cost_b, self.cost_s))
            elif portfolio == "callspread_writer":
                Q = np.exp(-self.gamma * cost.callspread_writer(xN, y_grid, self.cost_b, self.cost_s, self.K1,self.K2))
            elif portfolio == "callspread_buyer":
                Q = np.exp(-self.gamma * cost.callspread_buyer(xN, y_grid, self.cost_b, self.cost_s, self.K1,self.K2))

            # Initialize slices if needed
            if track_policy:
                Q_k      = [None] * (N+1)
                action_k = [None] * (N+1)
                Q_k[N]      = Q
                action_k[N] = np.zeros_like(Q, dtype=int)

            # Backward induction
            for k in range(N-1, -1, -1):
                # Continuation value (diffusion expectation)
                Q_new = 0.5 * (Q[:-1, :] + Q[1:, :])

                # Price grid at time k
                xk = x0 + (self.mu - 0.5*self.sig**2)*dt*k + (2*np.arange(k+1) - k)*dx

                # BUY branch: move up in y
                Buy = Q_new.copy()
                multF = F(xk, dy, k).reshape(-1,1)
                Buy[:, :-1] = multF * Q_new[:, 1:]

                # SELL branch: move down in y
                Sell = Q_new.copy()
                multG = G(xk, dy, k).reshape(-1,1)
                Sell[:, 1:] = multG * Q_new[:, :-1]

                # Choose minimum: continuation, buy, or sell
                Q = np.minimum.reduce([Q_new, Buy, Sell])

                if track_policy:
                    # Determine which action was optimal
                    A = np.zeros_like(Q, dtype=int)
                    mask_buy  = (Buy  < Q_new) & (Buy  < Sell)
                    mask_sell = (Sell < Q_new) & (Sell < Buy)
                    A[mask_buy]  = +1   # buy
                    A[mask_sell] = -1   # sell
                    # store
                    Q_k[k]      = Q
                    action_k[k] = A

            # extract central node
            if portfolio == "no_opt":
                Q_no = Q[0, med]
            else:
                Q_yes = Q[0, med]

            if track_policy:
                Q_store[portfolio]      = Q_k
                action_store[portfolio] = action_k

        # Final price calculation
        price = (delta[0]/self.gamma)*np.log(
            (Q_yes/Q_no) if TYPE=="callspread_writer" else (Q_no/Q_yes)
        )

        # Save tracking if requested
        if track_policy:
            self.Q_slices      = Q_store
            self.action_slices = action_store

        elapsed = time() - t_start
        if Time:
            return price, elapsed
        else:
            return price

# Usage example:
# pricer = TC_pricer(opt_info, proc_info, cost_b=0.002, cost_s=0.002, gamma=0.01)
# price = pricer.price(N=500, TYPE="writer", track_policy=True)
# # then inspect pricer.Q_slices and pricer.action_slices
