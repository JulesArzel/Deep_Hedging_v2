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


def callspread(x, y, cost_b, cost_s, K1, K2):
    cost = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        for j in range(len(y)):
            if y[j] < 0 and np.max(np.exp(x[i])-K1,0)-np.max(np.exp(x[i])-K2) <= 0:
                cost[i][j] = (1 + cost_b) * y[j] * np.exp(x[i])

            elif y[j] >= 0 and np.max(np.exp(x[i])-K1,0)-np.max(np.exp(x[i])-K2) <= 0:
                cost[i][j] = (1 - cost_s) * y[j] * np.exp(x[i])

            elif y[j] >= 0 and np.max(np.exp(x[i])-K1,0)-np.max(np.exp(x[i])-K2) > 0:
                cost[i][j] = ((1 - cost_s) * (y[j]) * np.exp(x[i])) + np.max(np.exp(x[i])-K1,0)-np.max(np.exp(x[i])-K2)

            elif y[j] < 0 and np.max(np.exp(x[i])-K1,0)-np.max(np.exp(x[i])-K2) > 0:
                cost[i][j] = ((1 + cost_b) * (y[j]) * np.exp(x[i])) + np.max(np.exp(x[i])-K1,0)-np.max(np.exp(x[i])-K2)

    return cost