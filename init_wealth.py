import numpy as np
import itertools

# Parameters
T = 3  # Number of periods
S0 = 100  # Initial stock price
K = 100  # Strike price
u = 1.1  # Up factor
d = 0.9  # Down factor
theta = 0.05  # Transaction cost rate

# Generate Stock Price Tree
stock_tree = np.zeros((T+1, T+1))
for t in range(T+1):
    for j in range(t+1):
        stock_tree[t, j] = S0 * (u ** j) * (d ** (t - j))

# Compute Option Payoff at Maturity
payoff = np.maximum(stock_tree[T] - K, 0)

# Define Discretized Hedge Positions
alpha_values = np.linspace(0, 1, 11)  # Possible stock holdings (e.g., 0 to 1 in 0.1 steps)

# Initialize Storage for Value Function and Trading Decisions
J = {}  # Cost-to-go function (Expected future hedging cost)
strategy = {}  # Optimal trading decisions

# Step 1: Initialize at Maturity (Final Payoff)
for j in range(T+1):
    for alpha in alpha_values:
        b = payoff[j] - alpha * stock_tree[T, j]  # Solve for bond holdings
        J[(T, j, alpha)] = max(payoff[j] - alpha * stock_tree[T, j] - b, 0)
        strategy[(T, j, alpha)] = (alpha, b)

# Step 2: Backward Recursion from t=T-1 to t=0
for t in reversed(range(T)):
    for j in range(t+1):
        S_t = stock_tree[t, j]  # Stock price at node (t, j)
        J_next = []
        
        for alpha in alpha_values:
            min_cost = float('inf')
            best_decision = None
            
            # Loop over all possible next-period holdings (alpha')
            for alpha_prime in alpha_values:
                S_up = stock_tree[t+1, j+1]
                S_down = stock_tree[t+1, j]
                
                # Compute bond holdings to satisfy self-financing constraint
                b_prime = - (alpha_prime * S_t - alpha * S_t + theta * abs(alpha_prime - alpha) * S_t)
                
                # Compute expected cost-to-go
                cost = 0.5 * (J[(t+1, j+1, alpha_prime)] + J[(t+1, j, alpha_prime)])
                
                # Add transaction cost
                cost += theta * abs(alpha_prime - alpha) * S_t
                
                # Update if this decision is better
                if cost < min_cost:
                    min_cost = cost
                    best_decision = (alpha_prime, b_prime)
            
            # Store best decision for (t, j, alpha)
            J[(t, j, alpha)] = min_cost
            strategy[(t, j, alpha)] = best_decision

# Step 3: Find Minimum Initial Wealth w*
w_star = min(J[(0, 0, alpha)] for alpha in alpha_values)

# Print Results
print(f"Minimum Initial Wealth Required: {w_star}")
print("\nOptimal Trading Strategy at t=0:")
for alpha in alpha_values:
    print(f"Hold {alpha:.2f} stocks -> Next Decision: {strategy[(0, 0, alpha)]}")
