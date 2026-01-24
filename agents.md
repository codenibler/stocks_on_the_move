This is a DEMO automated implementation of Andreas Clenow's Stocks on the Move Asset Management strategy. It trades the S&P1500, or (S&P500 + S&P400 + S&P600), uses the Trading212 api, for which you have the documentation on ./api_docs/212_api.json. You will log in rich detail, and logs for each run will be stored in logs/{today}, as well as printed to stdout. The strategy logic is as follows.

1) Every time python3 main.py is executed, you will scrape our tickers from wikipedia. Since we are trading the S&P1500, the links are the following:
- https://en.wikipedia.org/wiki/List_of_S%26P_400_companies
- https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
- https://en.wikipedia.org/wiki/List_of_S%26P_600_companies
This will return a list of the current, up to date constituents of the index we are trading. 

2) We will request all tradeable instruments from Trading212 with the "https://demo.trading212.com/api/v0/equity/metadata/instruments" endpoint, and filter them so that *currencyCode* == 'USD' and *type* == 'STOCK'. Then, we cross reference with the ones that we scraped from wikipedia. Reuse the matching algorithm you find in constituents.py.

3) For all stocks, we will request last 3mo of daily candles through yfinance. Add 3 retries in case of errors. If still NaN, empty, or error after, the stock is dropped from our list. For the remainder, we take a natural log (ln) of the price, and run a linear regression on it. Use scipy.stats.linregress ln(Close) prices. The momentum score calculation must follow: $Score = (Slope times 252) \ R^2$. We now have our momentum scores. We will also compute ATR20 of all these stocks. If there is no data, remove from list. If the stock's current price is below its 100SMA, we will drop it from our list. If in the last 90 days, the stock has a gap of >= 15% in a single day, it will be dropped again. Then, we generate a list of stocks, ranked by momentum score descending. Bar charts for the top 25, 25-50, 50-75, and 75-100 momentum ranked stocks should be saved with different colors to logs/{today}.

4) When this has been done, we check whether the S&P500 is ABOVE its 200SMA. If we are NOT above our 200SMA, we execute step 5 and exit the program. Otherwise, we execute everything else. 

5) We will then request all open positions on the account. For every single holding, WE ONLY CONSIDER AS A HOLDING THE **quantityAvailableForTrading**. Once you have the holdings now, you run them through the following guantlet.
- If a holding is not present in the top 100 of our momentum rankings, all of it should be sold. Since the top 100 of the momentum rankings also does not include high momentum stocks which have gapped >= 15% or are below their SMA100, then these are dropped implicitly.

6) Once we've sold the dropout stocks, we request our free cash. This is requested through the account/summary endpoint, where you take ONLY the **availableToTrade** cash attribute from the response. Them you compute total_equity, which is the total cash we requested + the holdings we still have in value. With these, we start from the top of our momentum ranked list and calculate position sizing as (0.02 * (total_equity)) / ATR20. while cash is >0. 

7) Once done, all of the changes should be pushed to the portfolio. 