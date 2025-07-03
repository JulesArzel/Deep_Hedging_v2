# Deep_Hedging_v2
Comparison of various hedging strategies in the presence of transaction costs, includes a written document to explain the strategies (Deep_Hedging.pdf).

**BS-Leland** :  

Introductory results on Black-Scholes and why it becomes inefficient when transaction costs are introduced, and results on the first approach to transaction costs : Leland.

**Stochastic Control Approach** :   

All results exposed either in the pdf or in the notebook 

Contains the *deep hedging free* approach, based on stochastic control and dynamic programming.
We consider a portfolio consisting of a risk free asset and one risky asset (the stock), and a short position in a call option (or Call Spread) (the underlying being the stock in the portfolio), and the goal is to maximize the expected utility of the terminal wealth in the portfolio. 
We solve this by backwards dynamic programming and get as output the hedging strategy and the price of the option, obtained as the smallest quantity to add to the portfolio with a position in the option, so that the investor is indifferent between holding and not holding the option.
This allows us to compare the price and hedging strategy with and without transaction costs, observing interesting results such as the apparition of a no-transaciton band in presence of tc, or the non-linearity of the price and hedging strategy of a Call Spreas in presence of tc. All this results + the theoretical ones are illustrated in the pdf 

**Deep Reinforcement Learning Approach** :

All results exposed either in the pdf or in the notebook 

Contains the deep hedging approach, where we look at results for two different architectures, one which leverages results exposed in the stochastic control approach, presented in 'No-Transaction Band Network: A Neural Network
Architecture for Efficient Deep Hedging'.
The setting is the same as in the first approach, we maximize expected utility by choosing the quantity of risky asset in the portfolio, which is chosen by a trained network, acting as our policy.
