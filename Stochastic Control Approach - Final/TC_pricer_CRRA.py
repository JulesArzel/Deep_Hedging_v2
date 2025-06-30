#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TC_pricerCRRA.py

4-D dynamic programming for European option pricing with proportional
transaction costs, CRRA utility.
"""

import numpy as np
from time import time

# import or adapt your cost/no_opt, writer, buyer functions here:
import cost_utils as cost

class TC_pricerCRRA:
    """
    4-state DP for CRRA utility:
      state = (t_k, B_l, y_i, S_j)
      utility U(W) = W^(1-alpha)/(1-alpha)
    """

    def __init__(self, Option_info, Process_info,
                 cost_b=0.0, cost_s=0.0, alpha=2.0,
                 B_min=-10.0, B_max=50.0, NB=40):
        # market params
        self.r      = Process_info.r
        self.mu     = Process_info.mu
        self.sig    = Process_info.sig
        self.S0     = Option_info.S0
        self.K      = Option_info.K
        self.T      = Option_info.T

        # transaction costs
        self.cost_b = cost_b
        self.cost_s = cost_s

        # CRRA exponent
        self.alpha  = alpha

        # cash grid
        self.B_min  = B_min
        self.B_max  = B_max
        self.NB     = NB

    def price(self, N=200, M_scale=0.5, TYPE="writer", track_policy=False):
        """
        N           = time steps
        M_scale     = fraction of N/2 for y-grid half-size
        TYPE        = "writer" or "buyer"
        track_policy = whether to store actions
        """
        t0 = time()
        # --- 1) build time and space grids ---
        dt   = self.T / N
        tgrid = np.linspace(0, self.T, N+1)

        # x->S grids: at time step k, x has length k+1
        dx    = self.sig * np.sqrt(dt)
        x0    = np.log(self.S0)
        Sgrid = [ None ] * (N+1)
        for k in range(N+1):
            xk = x0 + (self.mu - 0.5*self.sig**2)*dt*k + (2*np.arange(k+1)-k)*dx
            Sgrid[k] = np.exp(xk)

        # y-grid (size Ny)
        M  = int(np.floor(M_scale * N/2))
        dy = dx
        ygrid = np.linspace(-M*dy, M*dy, 2*M+1)
        Ny    = ygrid.size
        y0_ix = np.argmin(np.abs(ygrid))  # index of y=0

        # B-grid (size NB)
        Bgrid = np.linspace(self.B_min, self.B_max, self.NB)

        # --- 2) allocate value & policy arrays ---
        # V[k,ℓ,i,j] ~ V(t_k, B_ℓ, y_i, S_j)
        V = np.zeros((N+1, self.NB, Ny), dtype=float)
        if track_policy:
            A = np.zeros((N+1, self.NB, Ny), dtype=int)

        # helper for stock up/down indices
        def up_down(j,k):
            # simple symmetric binomial: j->j or j+1 in next step of length k+1
            return j, j+1

        # helper to find nearest B-index after cash change
        def find_B_index(newB):
            idx = np.searchsorted(Bgrid, newB)
            if idx<0:   idx=0
            if idx>=self.NB: idx=self.NB-1
            return idx

        # --- 3) terminal condition at k=N (no continuation, only payoff) ---
        SN = Sgrid[N]
        for ℓ in range(self.NB):
            B = Bgrid[ℓ]
            for i,y in enumerate(ygrid):
                for j,S in enumerate(SN):
                    # total wealth + payoff
                    W = B + y*S
                    if TYPE=="writer":
                        payoff = cost.writer([S], [y], self.cost_b, self.cost_s, self.K)[0]
                    elif TYPE=="buyer":
                        payoff = cost.buyer([S], [y], self.cost_b, self.cost_s, self.K)[0]
                    else:
                        payoff = cost.no_opt([S], [y], self.cost_b, self.cost_s)[0]
                    WT = W + payoff
                    # CRRA utility
                    V[N,ℓ,i,j] = WT**(1-self.alpha)/(1-self.alpha)
                # policy = hold at terminal
                if track_policy:
                    A[N,ℓ,i,:] = 0

        # --- 4) backward induction ---
        for k in range(N-1, -1, -1):
            Sk   = Sgrid[k]
            Skp1 = Sgrid[k+1]
            for ℓ in range(self.NB):
                Bcur = Bgrid[ℓ]
                for i,y in enumerate(ygrid):
                    for j,S in enumerate(Sk):
                        # 1) hold: no trade => same (ℓ,i), expectation over S up/down
                        j_up, j_dn = up_down(j,k)
                        cont = 0.5*( V[k+1,ℓ,i,j_up] + V[k+1,ℓ,i,j_dn] )

                        # 2) buy one share (i+1), pay (1+λ)S, move B->B_- 
                        buy_val = -1e300
                        if i+1<Ny:
                            cost_buy = (1+self.cost_b)*S
                            if Bcur >= cost_buy:
                                ℓm = find_B_index(Bcur - cost_buy)
                                buy_val = 0.5*( V[k+1,ℓm,i+1,j_up] + V[k+1,ℓm,i+1,j_dn] )

                        # 3) sell one share (i-1), receive (1-μ)S, move B->B_+
                        sell_val = -1e300
                        if i-1>=0:
                            gain_sell = (1-self.cost_s)*S
                            ℓp = find_B_index(Bcur + gain_sell)
                            sell_val = 0.5*( V[k+1,ℓp,i-1,j_up] + V[k+1,ℓp,i-1,j_dn] )

                        # pick max under CRRA
                        best = cont
                        act  = 0
                        if buy_val > best:
                            best, act = buy_val, +1
                        if sell_val > best:
                            best, act = sell_val, -1

                        V[k,ℓ,i,j] = best
                        if track_policy:
                            A[k,ℓ,i,j] = act

        # --- 5) read off price at initial state (t=0, ℓ0, i0, j0) ---
        # ℓ0: cash needed so B0 >= 0
        ℓ0 = find_B_index(0.0)
        i0 = y0_ix
        # j0: S0 -> index in Sgrid[0]
        j0 = 0  # Sgrid[0] has only one point
        price = V[0,ℓ0,i0,j0]

        out = (price,)
        if track_policy:
            self.V = V
            self.A = A
            out += (V,A,)
        out += (time()-t0,)
        return out

# --- usage example ---
# from TC_pricerCRRA import TC_pricerCRRA
# pricer = TC_pricerCRRA(opt_param, diff_param, cost_b=0.002, cost_s=0.002,
#                        alpha=2.0, B_min=-5, B_max=20, NB=40)
# price, (V,A), elapsed = pricer.price(N=300, track_policy=True, TYPE="writer")
# print("CRRA price:", price, "took", elapsed, "s")
