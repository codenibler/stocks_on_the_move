# Stocks on the Move (Trading212 Demo)

## What this is
DISCLAIMER: Before anything else, use this at your own discretion. I highly recommend you to play with the parameters in ./env and test on a practice Trading212 account before deploying on your Invest / Stocks ISA account, to make sure that it meets your risk profile and strategy behavior needs. 

Ok. With that out of the way, welcome! This is a a live implementation of a strategy strongly mirroring Andreas Clenow's [Stocks on the Move](https://www.clenow.com/books) equity momentum strategy. Oversimplified, the strategy collects all tickers in the S&P500, computes a momentum score for each, as a linear regression of the ln(price_series), and * by R² to penalize choppy trends. 

## How a rebalance runs (sequential)
### 1) Load configuration & initialize
<!-- Env vars, logging setup, Trading212 client, runtime config -->

### 2) Scrape index constituents
<!-- Wikipedia S&P 500/400/600 scraping -->

### 3) Fetch Trading212 instruments & filter tradable universe
<!-- USD + STOCK filtering -->

### 4) Cross‑reference constituents to tradeable instruments
<!-- Matching logic + outputs (matched/unmatched) -->

### 5) Pull market data (yfinance) for momentum window
<!-- Data fetch + retry policy -->

### 6) Compute momentum & apply filters
<!-- ln(Close) regression, ATR20, SMA100, gap>=15% filter -->

### 7) Rank momentum & generate momentum charts
<!-- Top 25/50/75/100 charts -->

### 8) Risk gate: S&P 500 vs 200‑SMA
<!-- Risk‑on vs risk‑off path -->

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
