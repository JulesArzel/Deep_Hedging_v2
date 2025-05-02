import numpy as np


def no_opt(x, y, cost_b, cost_s):
    cost = np.zeros((len(x), len(y)))

    for i in range(len(y)):
        if y[i] <= 0:
            cost[:, i] = (1 + cost_b) * y[i] * np.exp(x)
        else:
            cost[:, i] = (1 - cost_s) * y[i] * np.exp(x)

    return cost


def writer(x, y, cost_b, cost_s, K):
    cost = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            if y[j] < 0 and (1 + cost_b) * np.exp(x[i]) <= K:
                cost[i][j] = (1 + cost_b) * y[j] * np.exp(x[i])

            elif y[j] >= 0 and (1 + cost_b) * np.exp(x[i]) <= K:
                cost[i][j] = (1 - cost_s) * y[j] * np.exp(x[i])

            elif y[j] - 1 >= 0 and (1 + cost_b) * np.exp(x[i]) > K:
                cost[i][j] = ((1 - cost_s) * (y[j] - 1) * np.exp(x[i])) + K

            elif y[j] - 1 < 0 and (1 + cost_b) * np.exp(x[i]) > K:
                cost[i][j] = ((1 + cost_b) * (y[j] - 1) * np.exp(x[i])) + K

    return cost


def buyer(x, y, cost_b, cost_s, K):
    cost = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            if y[j] < 0 and (1 + cost_b) * np.exp(x[i]) <= K:
                cost[i][j] = (1 + cost_b) * y[j] * np.exp(x[i])

            elif y[j] >= 0 and (1 + cost_b) * np.exp(x[i]) <= K:
                cost[i][j] = (1 - cost_s) * y[j] * np.exp(x[i])

            elif y[j] + 1 >= 0 and (1 + cost_b) * np.exp(x[i]) > K:
                cost[i][j] = ((1 - cost_s) * (y[j] + 1) * np.exp(x[i])) - K

            elif y[j] + 1 < 0 and (1 + cost_b) * np.exp(x[i]) > K:
                cost[i][j] = ((1 + cost_b) * (y[j] + 1) * np.exp(x[i])) - K

    return cost



def callspread_writer(x, y, cost_b, cost_s, K1, K2):
    S = np.exp(x)[:, None]        # shape (Nx,1)
    Y = y[None, :]                # shape (1,Ny)

    # cash‐flow from spread for the writer = -(buyer‐payoff)
    payoff_buyer = np.maximum(S - K1, 0) - np.maximum(S - K2, 0)
    cash = - payoff_buyer         # writer’s cash-in = – buyer’s cash-out

    # liquidation cost of Y shares
    stock_cost = np.where(
        Y < 0,
        (1 + cost_b) * Y * S,
        (1 - cost_s) * Y * S
    )                             # shape (Nx,Ny)

    # total cost = stock_cost MINUS cash you receive
    return stock_cost + cash      # shape (Nx,Ny)

def callspread_buyer(x, y, cost_b, cost_s, K1, K2):
    S = np.exp(x)[:, None]
    Y = y[None, :]

    # buyer’s cash‐flow = max(S-K1,0) - max(S-K2,0)
    cash =   np.maximum(S - K1, 0) - np.maximum(S - K2, 0)

    stock_cost = np.where(
        Y < 0,
        (1 + cost_b) * Y * S,
        (1 - cost_s) * Y * S
    )

    # cost = liquidation + (–cash you pay)
    return stock_cost + cash
