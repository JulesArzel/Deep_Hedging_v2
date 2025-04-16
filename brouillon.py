# ------------------------------
# Parameters
# ------------------------------
T = 1.0
N_t = 20
dt = T / N_t

S0 = 100.0
sigma = 0.2
alpha = 0.05
r = 0.03
K = 100.0
gamma = 0.5
lambda_tc = 0.01

pi_vals = np.linspace(0, 1, 30)

u = np.exp(sigma * np.sqrt(dt))
d = np.exp(-sigma * np.sqrt(dt))
p = (np.exp(alpha * dt) - d) / (u - d)

# ------------------------------
# Wealth Grid
# ------------------------------
W_min, W_max = 1.0, 200.0
N_W = 400
W_grid = np.linspace(W_min, W_max, N_W)

# ------------------------------
# Utility Function
# ------------------------------
def utility(x):
    return max(x,1e-8)**gamma / gamma if x > 0 else -1e12

# ------------------------------
# Interpolation Helper
# ------------------------------
def interp_value(W_new, grid, values):
    return np.interp(W_new, grid, values)


# ==============================
# Part 1: No Transaction Costs
# ==============================

# ------------------------------
# Terminal Utility without Option
# ------------------------------
def terminal_utility_no_option(S):
    return np.array([utility(W) for W in W_grid])

# ------------------------------
# Dynamic Programming Solver (No Transaction Costs, No Option)
# ------------------------------
def solve_dp_no_transaction_cost(terminal_utility_func):
    U = {}
    Pi = {}
    for i in range(N_t + 1):
        S_val = S0 * (u**i) * (d**(N_t - i))
        U[(N_t, i)] = terminal_utility_func(S_val)
        Pi[(N_t, i)] = np.zeros(N_W)

    for n in reversed(range(N_t)):
        for i in range(n + 1):
            S_val = S0 * (u**i) * (d**(n - i))
            U_val = np.zeros(N_W)
            Pi_val = np.zeros(N_W)
            for j, W in enumerate(W_grid):
                best_val = -1e12
                best_pi = -1e12
                for pi in pi_vals:
                    W_up = W * (1 + r * dt + pi * ((alpha - r) * dt + sigma * np.sqrt(dt)))
                    W_down = W * (1 + r * dt + pi * ((alpha - r) * dt - sigma * np.sqrt(dt)))
                    U_up = interp_value(W_up, W_grid, U[(n+1, i+1)])
                    U_down = interp_value(W_down, W_grid, U[(n+1, i)])
                    U_pi = p * U_up + (1 - p) * U_down
                    if U_pi > best_val:
                        best_val = U_pi
                        best_pi = pi
                U_val[j] = best_val
                Pi_val[j] = best_pi
            U[(n, i)] = U_val
            Pi[(n, i)] = Pi_val
    return U, Pi

# ------------------------------
# Solve and Plot (No Option)
# ------------------------------
U_no_opt, Pi_no_opt = solve_dp_no_transaction_cost(terminal_utility_no_option)

plt.figure(figsize=(8, 6))
plt.plot(W_grid, U_no_opt[(0, 0)], 'b-', label="U(0,S0,W) no option")
plt.xlabel("Wealth, W")
plt.ylabel("Value Function U")
plt.title("Value Function at t=0 for S=S0 (No Option)")
plt.legend()
plt.grid(True)
plt.show()