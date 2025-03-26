import numpy as np
import pandas as pd

class DP2:
    def __init__(self, So, K, v, mu, r, T, n, h, c1, c2, c3, a, option_type=None, utype=None):
        """
        Initialize the OptionPricer with the required parameters.
        
        Parameters:
          So         : Initial stock price
          K          : Strike price
          v          : Volatility
          mu         : Drift of the stock
          r          : Risk-free interest rate
          T          : Time to expiration
          n          : Number of timesteps
          h          : Number of hedge units (discretization level)
          c1         : Transaction cost as a percentage of the stock price
          c2         : Fixed cost per share traded
          c3         : Flat fee per trade
          a          : Risk aversion parameter (for the exponential utility function)
          option_type: Extra parameter for utility (unused here, but kept for consistency)
          utype      : Extra parameter for utility (unused here, but kept for consistency)
        """
        self.So = So
        self.K = K
        self.v = v
        self.mu = mu
        self.r = r
        self.T = T
        self.n = n
        self.h = h
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.a = a
        self.option_type = option_type
        self.utype = utype
        
        # Basic derived parameters.
        self.dt = self.T / self.n
        self.u = np.exp(self.v * np.sqrt(self.dt))
        self.d = np.exp(-self.v * np.sqrt(self.dt))
        self.p = (np.exp(self.mu * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        
        # Build the binomial stock price tree using vectorized computation.
        j_idx = np.arange(self.n+1).reshape(-1, 1)  # time index (n+1, 1)
        i_idx = np.arange(self.n+1).reshape(1, -1)    # down moves index (1, n+1)
        S_full = self.So * (self.u ** (j_idx - i_idx)) * (self.d ** i_idx)
        self.S = np.tril(S_full)  # only lower-triangular part is relevant.
        
        # DP arrays (with time dimension!).
        self.fb = np.zeros((self.n+1, self.n+1, self.h+1))
        self.dh = np.zeros((self.n+1, self.n+1, self.h+1))
        self.hedg = np.zeros((self.n+1, self.n+1, self.h+1))
        
        # Will be set after pricing:
        self.option_value = None

    @staticmethod
    def utility_fn(x, a, option_type, utype, flag):
        """
        Exponential utility function:
          U(W) = -exp(-a * W).
        If flag == 1, return U(x).
        If flag == -1, return the inverse utility: W such that U(W) = x.
        
        Here, option_type and utype are not used but kept for generality.
        """
        if flag == 1:
            return -np.exp(-a * x)
        elif flag == -1:
            return -np.log(-x) / a
        else:
            raise ValueError("flag must be 1 or -1")

    def price_option(self):
        """
        Perform backward induction to compute the DP arrays fb and dh.
        Stores the final option price in self.option_value.
        """
        n, h = self.n, self.h
        S = self.S
        fb = self.fb
        dh = self.dh
        hedg = self.hedg
        
        # At maturity j = n, payoff for each node i in [0..n].
        payoff = np.maximum(S[n, :n+1] - self.K, 0.0)
        
        # Fill terminal condition:
        for i in range(n+1):
            for db in range(h+1):
                fb[n, i, db] = payoff[i]
                dh[n, i, db] = ((1.0 if S[n, i] > self.K else 0.0) * h) - db
        
        # Work backward in time from j = n-1 down to 0.
        for j in range(n-1, -1, -1):
            for i in range(j+1):
                for db in range(h+1):
                    # Create vector for all possible new hedge states da.
                    da = np.arange(h+1)
                    dS = (da / h) * S[j, i]
                    cost = (np.abs(da - db) / h) * S[j, i] * self.c1
                    
                    # Value in the up state: at time j+1, node i, hedge da.
                    up_value = (fb[j+1, i, :] - (da / h) * S[j+1, i]) * self.discount - dS + cost
                    # Value in the down state: at time j+1, node i+1, hedge da.
                    down_value = (fb[j+1, i+1, :] - (da / h) * S[j+1, i+1]) * self.discount - dS + cost
                    
                    util_up = self.utility_fn(up_value, self.a, self.option_type, self.utype, 1)
                    util_down = self.utility_fn(down_value, self.a, self.option_type, self.utype, 1)
                    utilities = self.p * util_up + (1 - self.p) * util_down
                    
                    best_da = np.argmax(utilities)
                    max_util = utilities[best_da]
                    
                    fb[j, i, db] = self.utility_fn(max_util, self.a, self.option_type, self.utype, -1)
                    dh[j, i, db] = best_da - db
                    hedg[j, i, db] = best_da
        
        self.option_value = fb[0, 0, 0]
        self.fb = fb
        self.dh = dh 
        self.hedg = hedg
        return self.option_value, self.fb, self.dh, self.hedg

    def simulate_forward(self):
        """
        Simulate one forward path from time 0 to n, using the DP hedge strategy in dh,
        and record at each time step:
          time, stock_price, option_value, hedge_state, hedge_adjustment,
          transaction_cost, portfolio_value, hedging_error.
        
        Returns:
          A pandas DataFrame of the simulation steps.
        """
        if self.option_value is None:
            raise ValueError("Must call price_option() before simulate_forward().")
        
        records = []
        n, h = self.n, self.h
        
        # Start at time 0, node i=0 (no down moves yet), hedge=0.
        t = 0
        i = 0
        current_hedge = 0
        
        option_value = self.fb[0, 0, int(current_hedge)]
        portfolio_value = option_value
        cash = portfolio_value
        
        records.append({
            "time": t,
            "stock_price": self.S[t, i],
            "option_value": option_value,
            "hedge_state": current_hedge,
            "hedge_adjustment": 0,
            "transaction_cost": 0.0,
            "portfolio_value": portfolio_value,
            "hedging_error": abs(portfolio_value - option_value)
        })
        
        # Move forward through time steps 0..n-1.
        for t in range(n):
            adjustment = self.dh[t, i, int(current_hedge)]
            new_hedge = current_hedge + adjustment
            
            # Compute transaction cost:
            if adjustment != 0:
                trans_cost = (abs(adjustment) / h) * self.S[t, i] * self.c1
            else:
                trans_cost = 0.0
            
            # Cost of buying/selling the stock portion.
            stock_trade_cost = (adjustment / h) * self.S[t, i]
            
            # Grow cash at the risk-free rate then pay for the stock trade and transaction cost.
            cash *= np.exp(self.r * self.dt)
            cash -= stock_trade_cost
            cash -= trans_cost
            
            # Update the portfolio.
            portfolio_value = cash + (new_hedge / h) * self.S[t, i]
            option_value = self.fb[t, i, int(current_hedge)]
            hedging_error = abs(portfolio_value - option_value)
            
            records.append({
                "time": t,
                "stock_price": self.S[t, i],
                "option_value": option_value,
                "hedge_state": new_hedge,
                "hedge_adjustment": adjustment,
                "transaction_cost": trans_cost,
                "portfolio_value": portfolio_value,
                "hedging_error": hedging_error
            })
            
            # Move to next node randomly.
            if np.random.rand() < self.p:
                i_new = i
            else:
                i_new = i + 1
            
            i = i_new
            current_hedge = new_hedge
        
        # At maturity time n.
        t = n
        final_stock_price = self.S[t, i]
        option_value = self.fb[n, i, int(current_hedge)]
        portfolio_value = cash + (current_hedge / h) * final_stock_price
        hedging_error = abs(portfolio_value - option_value)
        
        records.append({
            "time": t,
            "stock_price": final_stock_price,
            "option_value": option_value,
            "hedge_state": current_hedge,
            "hedge_adjustment": 0,
            "transaction_cost": 0.0,
            "portfolio_value": portfolio_value,
            "hedging_error": hedging_error
        })
        
        return pd.DataFrame(records)
