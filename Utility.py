import numpy as np

def utility_fn(x, a, type_, utype, flag):
    """
    Example utility function using negative exponential utility.
    
    Parameters:
      x     : the argument (e.g. a monetary outcome)
      a     : risk aversion parameter
      type_ : a flag for the utility type (not used in this simple example)
      utype : an additional utility type flag (not used here)
      flag  : if 1, compute the utility; if -1, compute the certainty equivalent
      
    For negative exponential utility:
      U(x) = -exp(-a*x)
      and the certainty equivalent is given by: -1/a * log(-U)
      
    (In a complete implementation, these choices would be adjusted.)
    """
    if flag == 1:
        return -np.exp(-a * x)
    elif flag == -1:
        # Return the certainty equivalent
        return - (1.0 / a) * np.log(-x)
    else:
        return x

def option_pricer(So, K, v, mu, r, T, n, h, c1, c2, c3, a, type_, utype):
    """
    Prices a call option in a two-period (or n-period) binomial model with transaction costs
    using dynamic programming. Transaction costs are incorporated as three components:
       - c1: proportional cost (percentage of the traded amount)
       - c2: cost proportional to the number of shares traded
       - c3: a flat fee per trade.
       
    The investor's utility is assumed to be given by utility_fn() and the optimal hedge is chosen
    at each node to maximize the expected utility.
    
    Parameters:
      So    : initial stock price
      K     : strike price of the call option
      v     : volatility parameter used to compute the up and down factors
      mu    : drift of the stock
      r     : risk-free interest rate
      T     : time to expiration
      n     : number of time steps in the binomial tree
      h     : number of discrete hedge levels (e.g., discretization units for delta)
      c1,c2,c3 : transaction cost parameters
      a     : risk aversion parameter for the utility function
      type_, utype : additional parameters for the utility function
       
    Returns:
      value : the computed option price
      dh    : a three-dimensional array holding optimal hedge adjustments at each node.
    """
    dt = T / n
    u = np.exp(v * np.sqrt(dt))
    d = np.exp(-v * np.sqrt(dt))
    p = (np.exp(mu * dt) - d) / (u - d)
    
    # Build the stock price tree: S is a lower-triangular (n+1)x(n+1) array.
    S = np.zeros((n+1, n+1))
    S[0, 0] = So
    for t in range(1, n+1):
        for s in range(t):
            S[s, t] = S[s, t-1] * u
        S[t, t] = S[t-1, t-1] * d

    # Preallocate arrays for option values (fb) and hedge adjustments (dh)
    fb = np.zeros((n+1, h+1))
    dh = np.zeros((n+1, n+1, h+1))
    
    # Terminal condition at time n:
    # For each discrete hedge level db = 0,1,...,h, set fb(:,db) to the terminal payoff.
    for db in range(h+1):
        # For all states at time n, the option payoff is max(S - K, 0)
        fb[:n+1, db] = np.maximum(S[:n+1, n] - K, 0)
        # Set the hedge adjustment at terminal time:
        # In MATLAB: dh(:, n+1, db+1) = (S(:, n) > K)*h - db.
        # In Python, we use index n for time n (0-indexed) and simply:
        dh[:n+1, n, db] = (S[:n+1, n] > K).astype(float) * h - db

    # Backward induction: loop from time n-1 down to time 0.
    for t in range(n-1, -1, -1):
        for s in range(t+1):
            # Save previous option values (fa) for convenience.
            fa = fb.copy()
            for db in range(h+1):
                # For each possible current hedge level (discrete value db),
                # we choose a new hedge level da (from 0 to h).
                utility = np.zeros(h+1)
                for da in range(h+1):
                    # Compute the incremental amount as a fraction of S(s,t)
                    dS = (da / h) * S[s, t] if h != 0 else 0
                    # Transaction cost for changing the hedge from db to da.
                    cost = (abs(da - db) / h * S[s, t] * c1 +
                            c2 * abs(da - db) / h + c3)
                    if da == db:
                        cost = 0
                    # Compute outcomes in the two possible states at time t+1.
                    up = (fa[s, da] - (da / h) * S[s, t+1]) * np.exp(-r * dt) + dS + cost
                    down = (fa[s+1, da] - (da / h) * S[s+1, t+1]) * np.exp(-r * dt) + dS + cost
                    # Expected utility at this node for choice da.
                    utility[da] = (utility_fn(up, a, type_, utype, 1) * p +
                                   utility_fn(down, a, type_, utype, 1) * (1-p))
                # The value at the node is obtained by converting the maximum expected utility
                # back into monetary units (using flag=-1).
                fb[s, db] = utility_fn(np.max(utility), a, type_, utype, -1)
                # Store the optimal hedge adjustment: the difference between the optimal da and current db.
                opt_da = np.argmax(utility)
                dh[s, t, db] = opt_da - db
    value = fb[0, 0]
    return value, dh

# Example usage:
if __name__ == "__main__":
    # Define parameters:
    So = 100.0    # initial stock price
    K = 100.0     # strike price
    v = 0.2       # volatility
    mu = 0.1      # drift
    r = 0.05      # risk-free rate
    T = 1.0       # time to expiration (1 year)
    n = 50        # number of time steps
    h = 10        # discretization levels for the hedge
    c1 = 0.01     # proportional transaction cost on value traded
    c2 = 0.005    # fixed cost component per unit traded
    c3 = 0.001    # flat fee per trade
    a = 0.1       # risk-aversion parameter
    type_ = 1     # utility type indicator (example)
    utype = 1     # additional utility type indicator
    
    option_value, hedge_adjustments = option_pricer(So, K, v, mu, r, T, n, h, c1, c2, c3, a, type_, utype)
    print("Option Price:", option_value)