import torch
from math import log
from torch.distributions.normal import Normal

def crra_utility(w, gamma=0.5):
    return (w.clamp(min=1e-6) ** gamma) / gamma

def exp_utility(w, gamma=0.5, clamp_min=-100.0, clamp_max=100.0):
    w_clamped = w.clamp(min=clamp_min, max=clamp_max)
    return -torch.exp(-gamma * w_clamped)

def notransactionband(K,T,t,sigma, model):
    
    # Black–Scholes delta
    def bs_delta(S, K, T, t, sigma):
        tau = torch.tensor(T - t, dtype=S.dtype, device=S.device)
        d1 = (S / K).log().add(0.5 * sigma**2 * tau) / (sigma * tau.sqrt())
        normal = Normal(0.0, 1.0)
        return normal.cdf(d1)

    # Grid of spot prices and corresponding log-moneyness
    S_vals = torch.linspace(0.65 * K, 1.5 * K, 1000)
    log_moneyness = (S_vals / K).log()
    t_vals = torch.full_like(S_vals, t)

    # Prepare model input
    prev_hedge = torch.zeros_like(S_vals)
    x_mlp_input = torch.stack([S_vals, t_vals, prev_hedge], dim=-1)  # shape [100, 3]

    # Forward pass
    with torch.no_grad():
        delta_vals = bs_delta(S_vals, K, T, t, sigma)
        width_vals = model.mlp(x_mlp_input)
        lower = delta_vals - torch.nn.functional.leaky_relu(width_vals[:, 0])
        upper = delta_vals + torch.nn.functional.leaky_relu(width_vals[:, 1])
        
    return log_moneyness, delta_vals, lower, upper

def compute_trade_frequency_pfhedge(hedge: torch.Tensor) -> float:
    """
    hedge: Tensor of shape (n_paths, n_steps)
    Returns: fraction of time steps where the hedge changes
    """
    hedge_diff = hedge[:, 1:] - hedge[:, :-1]
    traded = (hedge_diff != 0).sum().item()
    total = hedge_diff.numel()
    return traded / total

def compute_avg_num_shares_traded_pfhedge(hedge: torch.Tensor) -> float:
    """
    hedge: Tensor of shape (n_paths, n_steps)
    Returns: average number of shares traded per time step, per path
    """
    hedge_diff = hedge[:, 1:] - hedge[:, :-1]
    total_traded = hedge_diff.abs().sum().item()
    total_steps = hedge_diff.numel()
    return total_traded / total_steps

def compute_pnl(spot: torch.Tensor, hedge: torch.Tensor, cost: float = 0.0, initial_cash: torch.Tensor = None) -> torch.Tensor:
    n_paths, n_steps = hedge.shape

    dS = spot[:, 1:] - spot[:, :-1] 
    pnl_risky = (hedge * dS).sum(dim=1)  
    dhedge = torch.zeros_like(hedge)
    dhedge[:, 0] = hedge[:, 0]  
    dhedge[:, 1:] = hedge[:, 1:] - hedge[:, :-1] 
    trade_cost = cost * torch.abs(dhedge * spot[:, :-1]).sum(dim=1)
    terminal_hedge_value = hedge[:, -1] * spot[:, -1]

    if initial_cash is None:
        initial_cash = torch.zeros(n_paths, device=spot.device)

    pnl = initial_cash + pnl_risky - trade_cost + terminal_hedge_value

    return pnl

def fit_step_by_step(model, derivative, compute_pnl, utility_fn,
                     n_paths=2048, n_epochs=200, cost=1e-4,
                     gamma=0.5, lambda_duality=10.0, lr=1e-3, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        derivative.simulate(n_paths=n_paths)
        spot = derivative.underlier.spot  # (n_paths, n_steps+1)
        payoff = derivative.payoff()
        log_moneyness = derivative.log_moneyness()
        time_to_maturity = derivative.time_to_maturity()
        volatility = derivative.underlier.volatility

        n_steps = spot.shape[1] - 1
        device = spot.device

        W_with = model.price.expand(n_paths)
        W_no = torch.zeros(n_paths, device=device)
        prev_hedge_with = torch.zeros(n_paths, device=device)
        prev_hedge_no = torch.zeros(n_paths, device=device)

        for t in range(n_steps):
            x_t = torch.stack([
                log_moneyness[:, t],
                time_to_maturity[:, t],
                volatility[:, t],
                prev_hedge_with,
                prev_hedge_no,
                W_with,
                W_no
            ], dim=-1)


            hedge_with, hedge_no = model(x_t)
            hedge_with = hedge_with.squeeze(-1)
            hedge_no = hedge_no.squeeze(-1)

            dS = spot[:, t + 1] - spot[:, t]

            dhedge_with = hedge_with - prev_hedge_with
            dhedge_no = hedge_no - prev_hedge_no

            cost_with = cost * torch.abs(dhedge_with * spot[:, t])
            cost_no = cost * torch.abs(dhedge_no * spot[:, t])

            W_with = W_with + hedge_with * dS - cost_with
            W_no = W_no + hedge_no * dS - cost_no

            prev_hedge_with = hedge_with
            prev_hedge_no = hedge_no

        W_with = W_with + prev_hedge_with * spot[:, -1] - payoff
        W_no = W_no + prev_hedge_no * spot[:, -1]

        U_with = utility_fn(W_with, gamma)
        U_no = utility_fn(W_no, gamma)

        loss_duality = (U_with.mean() - U_no.mean())**2
        loss_risk = -U_with.mean()
        loss = loss_risk + lambda_duality * loss_duality

        loss.backward()
        optimizer.step()

        if verbose and epoch % 10 == 0:
            print(f"[{epoch:3d}] Loss={loss.item():.4f} | Price={model.price.item():.5f} | Duality Gap={loss_duality.item():.5f}")

    return model.price.item()
