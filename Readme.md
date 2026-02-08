# Stocks on the Move (Trading212 Demo)

## DISCLAIMER (READ FIRST)
Use this at your own discretion. I recommend you to play with the parameters in .env and test on a practice Trading212 account before deploying on your Invest / Stocks ISA account, to make sure that it meets your risk profile and strategy behavior needs. 

## What this is
Ok. With that out of the way, welcome! This is a live implementation strongly mirroring Andreas Clenow's [Stocks on the Move](https://www.clenow.com/books) equity momentum strategy. Oversimplified, the strategy collects all tickers in the S&P500 and computes a momentum score for each. This score consists of a linear regression of a natural log of the daily price series for the last 90 days, multiplied by R² to penalize choppy trends. 

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
# risk_factor is a parameter in .env, and ATR20 is the Average True Range 
# indicator over the last 20 1d candles for the given stock. 
```

The rest of the logic is more case-specific, and if you even made it this far, just read the book at this point... The way I personally manage my portfolio with this strategy is by hosting on a VPS and setting a scheduler to execute at a random moment during regular market hours, a form of [Staggered Rebalancing](https://www.thinknewfound.com/rebalance-timing-luck) to avoid rebalance timing luck. Now, however, are the main differences from the original implementation of [Stocks on the Move](https://www.clenow.com/books). 
- This implementation considers the S&P1500, a composite of the S&P500 Large Caps, S&P400 Mid Caps, and S&P600 Small Caps. 
- I have a default risk factor of 0.02, compared to the book's 0.01. 
Note that there are several other smaller variations. I just don't have the time to re-read the book before finishing this readme.

## Configuration
Everything is configured through .env. Copy dotenvstructure.txt and rename it to .env. I’d start by editing:
- Trading212 credentials and environment
- Top N, SMA, gap threshold, and risk fraction
- SP400/SP600 tickers (used for index charts)
- Chart font / chart font path
- Telegram bot settings (if you want notifications) and easily accessible artifacts.

## Telegram Setup (Optional)
If you want Telegram alerts, you’ll need to create your own bot via @botfather, copy the API token it returns, and then get your Telegram user ID through @userinfobot. Paste both values into the .env (TELEGRAM_API_TOKEN and TELEGRAM_USER_ID), set TELEGRAM_ENABLED=true, and you should be good to go.

## Running locally
Typical setup:
1) python3 -m venv venv
2) source venv/bin/activate
3) pip install -r requirements.txt
4) copy dotenvstructure.txt -> .env and fill it out
5) python3 main.py
