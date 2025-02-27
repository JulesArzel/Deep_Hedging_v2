import numpy as np

# Parameters
T = 5  # Number of time steps
S0 = 100  # Initial stock price
u = 1.1  # Up factor
d = 0.9  # Down factor
k = 0.02  # Transaction cost
strike = 100  # Strike price (for European call)
r = 0.01  # Risk-free rate (not directly used)
grid_size = 50  # Grid size for discretization

# Generate Stock Price Tree
def generate_stock_tree(S0, u, d, T):
    tree = np.zeros((T+1, T+1))
    for t in range(T+1):
        for j in range(t+1):
            tree[j, t] = S0 * (u ** (t-j)) * (d ** j)
    return tree

S_tree = generate_stock_tree(S0, u, d, T)

# Payoff function at maturity (European call option)
def option_payoff(S):
    return max(S - strike, 0)

# Discretization of portfolio positions
delta_vals = np.linspace(-1, 1, grid_size)  # Possible stock positions
beta_vals = np.linspace(-100, 100, grid_size)  # Possible cash positions

# Initialize terminal value function at T
J_T = np.zeros((grid_size, grid_size, T+1))

for i, delta in enumerate(delta_vals):
    for j, beta in enumerate(beta_vals):
        S_T = S_tree[:, T]  # Possible stock values at T
        portfolio_value = delta * S_T + beta
        J_T[i, j, T] = np.max(option_payoff(S_T) - portfolio_value, 0)

# Backward recursion
J = np.copy(J_T)  # Store dynamic programming values

for t in range(T-1, -1, -1):  # Iterate backwards
    for i, delta in enumerate(delta_vals):
        for j, beta in enumerate(beta_vals):
            S_t = S_tree[:, t]
            expected_cost = 0
            
            for idx, S_next in enumerate([S_t * u, S_t * d]):  # Up and down moves
                # Find closest indices in the discretization grid
                closest_delta_idx = np.argmin(np.abs(delta_vals - delta))
                closest_beta_idx = np.argmin(np.abs(beta_vals - beta))
                
                # Compute transaction cost adjustment
                delta_new = delta_vals[closest_delta_idx]
                beta_new = beta_vals[closest_beta_idx]
                cost = np.abs(delta_new - delta) * k * S_next
                
                expected_cost += 0.5 * (J[closest_delta_idx, closest_beta_idx, t+1] + cost)
            
            J[i, j, t] = expected_cost  # Store in DP table

# Extract optimal hedging strategy
w_star = np.min([w for w, val in enumerate(J[:, :, 0]) if val == 0])

print(f"Optimal initial wealth required: {w_star}")
