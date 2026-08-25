# Deep_Hedging_WW-NTBN

Comparison of various hedging strategies in the presence of transaction costs. The project is now complete, and the full write-up is available both as a pdf in this repo (WW-NTBN.pdf) and as a paper on arXiv, *Bridging Stochastic Control and Deep Hedging: Structural Priors for No-Transaction Band Networks* (joint work with Noureddine Lehdili) https://arxiv.org/abs/2603.29994.

The goal of the project is to compare two very different ways of solving the same hedging problem, an investor holding a stock and a short position in a call option (or call spread), who wants to hedge as well as possible once transaction costs are introduced. On one side we have stochastic control, which gives an exact but computationally heavy solution via dynamic programming. On the other we have deep hedging, which trains a neural network to hedge directly. The main contribution of the project is the WW-NTBN, a network architecture that injects the Whalley-Wilmott asymptotic result from the stochastic control side directly into the deep hedging network, so that the two approaches inform each other instead of being treated separately.

**BS-Leland** :  
Introductory results on Black-Scholes and why it becomes inefficient once transaction costs are introduced, and results on the first historical approach to transaction costs, Leland's.

**Stochastic Control Approach** :   
All results exposed either in the pdf or in the notebook.  
Contains the deep hedging free approach, based on stochastic control and dynamic programming. We consider a portfolio made of a risk free asset, a risky asset (the stock), and a short position in a call option (or call spread) written on that stock, and the goal is to maximize the expected utility of the terminal wealth. We solve this by backward dynamic programming, which gives us both the hedging strategy and the price of the option, obtained as the smallest amount to add to the portfolio so that the investor is indifferent between holding and not holding the option.  
This lets us compare price and hedging strategy with and without transaction costs, and observe results such as the appearance of a no-transaction band once costs are introduced, or the non-linearity of the price and hedging strategy of a call spread in their presence. All of this, along with the underlying theory, is detailed in the pdf.

**Deep Reinforcement Learning Approach** :  
All results exposed either in the pdf or in the notebook.  
Contains the deep hedging approach, where we compare three architectures: a plain MLP, an NTBN, and the WW-NTBN. The setting is the same as in the stochastic control approach, we maximize expected utility by choosing the position in the risky asset, only here that position is output by a trained network acting as our policy.  
The NTBN centers its no-transaction band around the Black-Scholes delta, an implementation choice we make explicit and that isn't spelled out in the original paper. The WW-NTBN goes further and initializes the band's width directly from the Whalley-Wilmott formula, so the network starts training with the right order of magnitude for the band and converges faster, while also producing bands that match the stochastic control solution more closely than the plain NTBN across all transaction cost levels tested.

Between the two approaches, stochastic control gives rigorous optimality guarantees and exact indifference prices under its model assumptions, while deep hedging is more flexible and scales to complex payoffs and portfolios. WW-NTBN sits in between, it trades some of the model agnosticism of pure deep hedging for the structural accuracy of stochastic control. All the numerical comparisons, including the divergence between the two approaches at high transaction costs, are laid out in the pdf.
