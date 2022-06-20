# Portfolio Optimization

## Background

In algorithmic trading, as in investing, it is important to diversify. Each strategy may only be profitable under certain market conditions. Even if your strategies all have positive expected value over time, if they are highly correlated, you may lose a substantial fraction of your capital in an unfavorable market.

For example, suppose that strategy *A* results in a 30% loss from January to March, followed by a 60% gain from April to June, while strategy *B* results in a 20% gain from January to March followed by a 10% loss from April to June.
Consider the following possible allocations:

| Allocation | Initial Capital | Q1 End | Q2 End  |
| ---------- | --------------- | ------ | ------- |
| 100% *A*   | $1000           | $700   | $1120   |
| 100% *B*   | $1000           | $1200  | $1080   |
| 50/50      | $1000           | $950   | $1187.5 |

In this example, the allocations are rebalanced at the end of each quarter.

### Measuring Risk

The issue with strategy *A* is that, even though it has higher expected returns than strategy *B*, it's also quite risky, so we may lose a lot of our starting capital if our timing is poor.

There are several ways to quantify risk, but the metric we will consider is [maximum drawdown](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp), which is defined as the greatest percentage loss from any peak.

Risk should be considered at the portfolio level - it's okay to take risks if they are balanced out.

## Local Search

When considering a large set of securities (or trading strategies), the question of which subset to choose lends itself well to local search. 

A simple objective function is a [risk/reward ratio](https://www.investopedia.com/terms/r/riskrewardratio.asp); here, we will use its inverse, a _return/risk_ ratio. We will consider "return over maximum drawdown", where both are expressed in percentage points. For example, a 15% return with a 5% maximum drawdown yields a return/risk ratio of 3. This is better than a 16% return with an 8% maximum drawdown, which has a ratio of only 2.

In `main.py`, implement local search for portfolio optimization. Leave a comment which clearly defines the neighborhood of the current solution. You should use a random starting point.

### Random Restarts

One technique to improve the results of local search is to start over at a new random starting point when your algorithm reaches a local optimum. This can be repeated several times, with the best solution from all runs chosen as the final answer. Implement this technique in `main.py`.
