import numpy as np
import pandas as pd

class PricerDP_no_tc:
    def __init__(self, So, K, v, mu, r, T, n, h, a, option_type=None, utype=None):
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
        self.a = a
        self.option_type = option_type
        self.utype = utype
        
        # Basic derived parameters.
        self.dt = self.T / self.n
        self.u = np.exp(self.v * np.sqrt(self.dt))
        self.d = np.exp(-self.v * np.sqrt(self.dt))
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        
        # Build the binomial stock price tree S[j, i]:
        # j = time index (0..n), i = node index (0..j).
        self.S = np.zeros((self.n+1, self.n+1))
        self.S[0, 0] = self.So
        for j in range(1, self.n+1):
            for i in range(j):
                self.S[j, i] = self.S[j-1, i] * self.u
            self.S[j, j] = self.S[j-1, j-1] * self.d
        
        # DP arrays (with time dimension!).
        self.value = np.zeros((self.n+1, self.n+1))
        self.hedge = np.zeros((self.n+1, self.n+1))
        
        # Will be set after pricing:
        self.option_value = None
        
    @staticmethod
    def utility_fn(x, a, option_type, utype, flag):
        """
        Exponential utility function:
          U(W) = -exp(-a * W).
        If flag == 1, return U(x).
        If flag == -1, return the inverse utility: W s.t. U(W) = x.
        
        Here, option_type and utype are not used but kept for generality.
        """
        if flag == 1:
            return -np.exp(-a * x)
        elif flag == -1:
            # Solve -exp(-a*W) = x => W = -log(-x)/a.
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
        value = self.value
        hedge = self.hedge
        
        # At maturity j = n, payoff for each node i in [0..n].
        payoff = np.maximum(S[n, :n+1] - self.K, 0.0)
        
        # Fill terminal condition:
        for i in range(n+1):
            value[n, i] = payoff[i]
            # Hedge adjustment at maturity (if S > K, go to full hedge; else 0).
            hedge[n, i] = (1.0 if S[n, i] > self.K else 0.0)

        # Work backward in time from j = n-1 down to 0.
        for j in range(n-1, -1, -1):
            for i in range(j+1):
                hedge[j,i] = (value[j+1,i]-value[j+1,i+1])/(S[j+1,i]-S[j+1,i+1])
                up_value = ( value[j+1, i] - (hedge[j,i])*S[j+1, i] ) * self.discount - (hedge[j,i])*S[j, i]
                down_value = ( value[j+1, i+1] - (hedge[j,i])*S[j+1, i+1] ) * self.discount - (hedge[j,i])*S[j, i]
                util_up = self.utility_fn(up_value, self.a, self.option_type, self.utype, 1)
                util_down = self.utility_fn(down_value, self.a, self.option_type, self.utype, 1)
                    
                # Convert utility back to monetary value:
                value[j, i] = self.utility_fn(self.p * util_up + (1 - self.p) * util_down, self.a, self.option_type, self.utype, -1)
                
                
        # The option's fair value at time 0 is at (j=0, i=0).
        self.option_value = value[0, 0]
        self.value = value
        self.hedge = hedge
        return self.option_value, self.value, self.hedge