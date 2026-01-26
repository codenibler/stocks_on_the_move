# Stocks on the Move (Trading212 Demo)

## DISCLAIMER (READ FIRST)
Before anything else, use this at your own discretion. I highly recommend you to play with the parameters in ./env and test on a practice Trading212 account before deploying on your Invest / Stocks ISA account, to make sure that it meets your risk profile and strategy behavior needs. 

## What this is
Ok. With that out of the way, welcome! This is a live implementation of a strategy strongly mirroring Andreas Clenow's [Stocks on the Move](https://www.clenow.com/books) equity momentum strategy. Oversimplified, the strategy collects all tickers in the S&P500 and computes a momentum score for each. This score consists of a linear regression of a natural log of the daily price series for the last 90 days, multiplied by R² to penalize choppy trends. 

```python
# Momentum calculation 
from scipy.stats import linregress
import numpy as np

log_prices = np.log(close_prices)
regression = linregress(range(len(log_prices)), log_prices)
score = (regression.slope * 252) * (regression.rvalue ** 2)
```

With scores for all stocks, we first check our regime filter. If the S&P500 is >= 200SMA, we consider opening new positions, otherwise, we hold our cash, and close positions only if they fulfill our exit conditions, being, a gap >= 15%, price < SMA100, or a Momentum Ranking outside of the top 100. Almost all of these parameters are available in the .env file you must create with dotenvstructure.txt (copy into a new .env file and tweak as needed) to tune the strategy to your liking. 

We then narrow down our list by dropping stocks that:
- Have a latest close < their SMA100
- Have a gap close-open >= 15% in the last 90 days
- Are no longer in the top 100 momentum scores. 
This list is then ranked by descending momentum. In the case that the S&P500 is above its 200SMA, we purchase stocks down the list with the following formula for position size: 

```python
position_size = (risk_factor * total_equity) / ATR20 
# risk_factor is a parameter in .env, and ATR20 is the Average True Range indicator over the last 20 1d candles for the given stock. 
```
The rest of the logic is more case-specific, and if you even made it this far, just read the book at this point... The way I personally manage my portfolio with this strategy is by hosting on [Railway](https://railway.com/?referralCode=QJ3qb6&gad_source=1) and setting a scheduler to execute at a random moment during regular market hours, a kind of wild form of [Staggered Rebalancing](https://www.thinknewfound.com/rebalance-timing-luck) to avoid rebalance timing luck.. Now, however, are the main differences from the original implementation of [Stocks on the Move](https://www.clenow.com/books). 
- This implementation considers the S&P1500, a composite of the S&P500 Large Caps, S&P400 Mid Caps, and S&P600 Small Caps. 
- I have a default risk factor of 0.02, compared to the book's 0.01. 
Note that there are several other smaller variations. I just don't have the time to read the book before finishing this readme.

## How a rebalance runs (sequential)
### 1) Load configuration & initialize
When running python3 main.py, which I recommend you set up with some sort of scheduler at the intervals you so desire, the script overrides all env vars you might have set and uses those in .env. I like things this way.

### 2) Scrape index constituents, cross-reference with Trading212 Instruments
The script proceeds to scrape all constituents from the Wikipedia pages for the [S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies), [S&P 400](https://en.wikipedia.org/wiki/List_of_S%26P_400_companies), and [S&P 600](https://en.wikipedia.org/wiki/List_of_S%26P_600_companies). We then request all tradeable instruments from Trading212 and filter for type=='STOCK' and currency=='USD'. We then check, through an algorithm you can find in the codebase, and match all possible instruments. 

### 3) Pull market data (yfinance) for momentum window
With this filtered stock universe, we batch request the last 6mo of 1d candles for stocks, and retry on failed requests. 

### 6) Compute momentum & apply filters
When collected, we compute momentum scores and the ATR20 with the algorithm mentioned above. Stocks are dropped from our rankings if they don't comply with SMA100 and gap requirements.

### 7) Rank momentum & generate momentum charts
Stocks are then ranked by descending momentum, and bar charts are generated with top100 stocks and their scores. 

### 8) Risk gate: S&P 500 vs 200‑SMA
Risk gate is executed.

### 9) Snapshot portfolio before rebalance
<!-- Pre‑rebalance holdings pie -->

### 10) Sell positions not in top 100
<!-- Sell orders for dropouts -->

### 11) Calculate cash + total equity
<!-- availableToTrade + holdings value -->

### 12) Size positions & build rebalance orders
<!-- ATR position sizing + rebalance threshold -->

### 13) Submit rebalance orders
<!-- Rebalance sells, rebalance buys, new buys -->

### 14) Snapshot portfolio after rebalance
<!-- Post‑rebalance holdings pie + index exposure bar -->

### 15) Generate rebalance report PDF
<!-- Includes charts, order summaries, and index price action -->

## Outputs & artifacts
<!-- logs/{today}, report files, charts, symbols snapshots -->

## Configuration
<!-- .env keys, defaults, and examples -->

## Running locally
<!-- Setup steps and how to run main.py -->

## Testing
<!-- How to run tests (if any) -->

## Known limitations / TODO
<!-- Open items, known gaps, future improvements -->
