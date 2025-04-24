# ------------------------------
# Parameters
# ------------------------------
T = 1.0
N_t = 40
dt = T / N_t

S0 = 100.0
sigma = 0.3
alpha = 0.05
r = 0.03
K = 100.0
gamma = 0.5
lambda_tc = 0.01

pi_vals = np.linspace(-2, 2, 100)

u = np.exp(sigma * np.sqrt(dt))
d = np.exp(-sigma * np.sqrt(dt))
#p = (np.exp(alpha * dt) - d) / (u - d)
p=0.5

# ------------------------------
# Wealth Grid
# ------------------------------
W_min, W_max = 1e-3, 200.0
N_W = 500
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
                def objective_pi(pi):
                    W_up = W * (1 + r * dt + pi * ((alpha - r) * dt + sigma * np.sqrt(dt)))
                    W_down = W * (1 + r * dt + pi * ((alpha - r) * dt - sigma * np.sqrt(dt)))
                    U_up = interp_value(W_up, W_grid, U[(n+1, i+1)])
                    U_down = interp_value(W_down, W_grid, U[(n+1, i)])

                    expected_U = p * U_up + (1 - p) * U_down
                    return -expected_U  

                res = minimize_scalar(
                    objective_pi,
                    bounds=(-2, 2), 
                    method='bounded'
                )

                Pi_val[j] = res.x
                U_val[j] = -res.fun  

            U[(n, i)] = U_val
            Pi[(n, i)] = Pi_val
        print(f"Processing time step {n} of {N_t-1}...")

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

Pi_theo = np.full((N_W,N_t),(alpha-r)/(sigma**2*(1-gamma))) 


plt.figure(figsize=(8, 6))
plt.plot(W_grid, Pi_no_opt[(0, 0)], 'b-', label="$\pi$(0,S0,W) no option")
plt.plot(W_grid,Pi_theo[:,0],'r-', label='$\pi$ theoretical')
plt.xlabel("Wealth, W")
plt.ylabel("Value Function U")
plt.title("Value Function at t=0 for S=S0 (No Option)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Terminal Utility with Option
# ------------------------------

def terminal_utility_with_option(S):
    payoff = max(S - K, 0.0)
    return np.array([utility(max(W - payoff, 1e-8)) for W in W_grid])

# ------------------------------
# Solve and Plot (With Option)
# ------------------------------
U_with_opt, Pi_with_opt = solve_dp_no_transaction_cost(terminal_utility_with_option)

plt.figure(figsize=(8, 6))
plt.plot(W_grid, U_with_opt[(0, 0)], 'r--', label="U(0,S0,W) with option")
plt.xlabel("Wealth, W")
plt.ylabel("Value Function U")
plt.title("Value Function at t=0 for S=S0 (With Short Call Option)")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Theoretical pi values when there is a position in option
# ------------------------------

Pi_theo_opt = {}
for n in reversed(range(N_t)):
    for i in range(n+1): 
        S_val = S0 * (u**i) * (d**(n - i))
        Price = call_price(0, S_val, (N_t-n)/N_t, r, sigma, K)  
        Delt = Delta(0, S_val, (N_t-n)/N_t, r, sigma, K)
        Pi_vals = np.zeros(N_W)
        for j, W in enumerate(W_grid):
            Pi_vals[j] = (alpha-r)/(sigma**2*(1-gamma))*((W-Price)/W)+(Price/W)*Delt
        Pi_theo_opt[(n,i)] = Pi_vals
        
# ------------------------------
# Proportion for BS Delta Hedging
# ------------------------------

Delt = Delta(0,S0,T,r,sigma,K)
Pi_delta_0 = [min((Delt*S0)/W,4) for W in W_grid] 
print('Delta Black-Scholes =', Delt)

plt.figure(figsize=(10, 6))
plt.plot(W_grid, Pi_with_opt[(0, 0)], label='$\pi$ with option', lw=2)
plt.plot(W_grid, Pi_theo_opt[(0,0)], label="$\pi$ with option (theoretical)")
plt.plot(W_grid, Pi_delta_0, label = '$\pi$ from BS Delta Hedging')
plt.xlabel("Wealth, W")
plt.ylabel("Optimal proportion")
plt.ylim((-2,3))
plt.title("Optimal theoretical proportion vs real proportion")
plt.legend()
plt.grid(True)
plt.show()