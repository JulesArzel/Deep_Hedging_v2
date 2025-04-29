tc2 = TC2.price(N=500, TYPE="writer", track_policy=True, Time=False)
print(f"{cost} TC price : ",tc2)
intensity = compute_trading_intensity(TC2)
print("Trading‐intensity metric:", intensity)

t_plot = 0.1 # Time we want to plot in percentage 

N = 500
k = int(round(t_plot / TC2.T * N))

# Grid of prices from the tree 
dt = TC2.T / N
dx = TC2.sig * np.sqrt(dt)
x0 = np.log(TC2.S0)
x_k = x0 + (TC2.mu - 0.5 * TC2.sig**2)*dt*k + (2*np.arange(k+1) - k)*dx

# Grid for possible stock holdings
M = int(0.8*np.floor(N/2))
dy = dx
y = np.linspace(-M*dy, M*dy, 2*M+1)

A = TC2.action_slices["writer"][k]    # shape (k+1, len(y))
Q = TC2.Q_slices["writer"][k]         # = E[e^{–γW_T}]
U = 1 - Q                             # indirect utility U_t

X, Y = np.meshgrid(x_k, y, indexing='xy')  # Y-axis = shares, X-axis = log price

from scipy.stats import norm

# a) Policy plot
plt.figure(figsize=(6,4))
cmap = plt.get_cmap('RdYlBu', 3) 
im = plt.pcolormesh(X, Y, A.T, cmap=cmap, vmin=-1, vmax=1)
plt.colorbar(im, ticks=[-1,0,1], label='Policy')
plt.xlabel('log price')
plt.ylabel('shares y')
plt.title(f'Optimal Policy at t={t_plot}')

# convert log‐grid back to real-price if you want S‐axis, 
# or keep x_k for log‐price axis
tau    = TC2.T - t_plot
S_k    = np.exp(x_k)
d1      = (np.log(S_k/TC2.K) + (TC2.r + 0.5*TC2.sig**2)*tau) \
            / (TC2.sig * np.sqrt(tau))
delta_bs = norm.cdf(d1)
# if your x‐axis is log‐price:
#plt.plot(x_k, delta_bs, 'k--', linewidth=1, label='BS Δ')
# if you had converted X axis to real price S_grid, instead use:
# plt.plot(S_k, delta_bs, 'k--', linewidth=2, label='BS Δ')

# Optimal trading policy in theory, with no transaction costs
mu, r, sigma = diff_param.mu, diff_param.r, diff_param.sig
K = opt_param.K
gamma = 0.5

tau = TC2.T - t_plot
d1 = (np.log(S_k/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
delta_bs = norm.cdf(d1)

disc = np.exp(-r * tau)                 # δ(T,s)
drift_adj = (mu - r) / (sigma**2)
phi_w = delta_bs + disc/(gamma * S_k) * drift_adj

plt.plot(x_k, phi_w, 'k--', linewidth=1, label=r'$\Delta_{BS}+\frac{\delta(T,s)(\alpha-r)}{\gamma S \sigma^2}$')

plt.legend(loc='upper left')
plt.show()

# b) Indirect utility plot
'''plt.figure(figsize=(6,4))
im2 = plt.pcolormesh(X, Y, U.T, shading='auto')
plt.colorbar(im2, label='U_t = 1 - E[e^{-γW_T}]')
plt.xlabel('log price')
plt.ylabel('shares y')
plt.title(f'Indirect Utility at t={t_plot}')
plt.show()
'''




gamma = 0.0001
pricer0 = TC_pricer2(opt_param, diff_param, cost_b=0.0, cost_s=0.0, gamma=gamma)

# --- 1) Solve with zero transaction costs ---
price0 = pricer0.price(N=500, TYPE="writer", track_policy=True)

bs = BS.closed_formula()

print("Zero TC price: ", price0)
change = round(((tc2-price0)/price0)*100,2)
print(f"Sensitivity to Transaction Costs : {change}%")
print("Black Scholes price:", bs)
print("Difference:", np.abs(price0 - bs))


# --- 2) Extract model parameters ---
mu, r, sigma = diff_param.mu, diff_param.r, diff_param.sig
K = opt_param.K

# --- 3) Pick a time slice t_plot ---
t_plot = 0.5
N = 500
k = int(round(t_plot / pricer0.T * N))
dt = pricer0.T / N
dx = sigma * np.sqrt(dt)

# --- 4) Rebuild your (x,y) grids at time k ---
# log‐price grid
x0 = np.log(pricer0.S0)
x_k = x0 + (mu - 0.5 * sigma**2) * dt * k + (2 * np.arange(k+1) - k) * dx
S_k = np.exp(x_k)

# share‐position grid
M = int(0.8*np.floor(N / 2))
dy = dx
y = np.linspace(-M*dy, M*dy, 2*M+1)

# --- 5) Extract the DP policy at slice k ---
A = pricer0.action_slices["writer"][k]   # shape (k+1, len(y))

# --- 6) Compute Black–Scholes delta and φ_w(s,S) from eq. (4.31) ---
tau = pricer0.T - t_plot
d1 = (np.log(S_k/K) + (r + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
delta_bs = norm.cdf(d1)

disc = np.exp(-r * tau)                 # δ(T,s)
drift_adj = (mu - r) / (sigma**2)
phi_w = delta_bs + disc/(gamma * S_k) * drift_adj
phi_w = np.clip(phi_w, y[0], y[-1])

# --- 7) Plot the DP action map and overlay φ_w ---
X, Y = np.meshgrid(x_k, y, indexing='xy')

plt.figure(figsize=(8,4))
cmap = plt.get_cmap('RdYlBu', 3)
plt.pcolormesh(X, Y, A.T, cmap=cmap, vmin=-1, vmax=1, shading='auto')

# dashed line = theoretical φ_w(s,S)
plt.plot(x_k, phi_w, 'k--', linewidth=1, label=r'$\varphi_w(s,S)$ (eq 4.31)')

plt.colorbar(ticks=[-1,0,1], label='Action: -1=sell, 0=hold, +1=buy')
plt.xlabel('log‐price $x$')
plt.ylabel('position $y$')
plt.title(f'Writer policy at t={t_plot:.2f}, tc=0, γ={gamma}')
plt.legend()
plt.tight_layout()
plt.show()
