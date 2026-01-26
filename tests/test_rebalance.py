import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.classes import AnalyzedStock
from execution import portfolio


def _stock(
    ticker: str,
    *,
    atr20: float,
    price: float,
    score: float = 1.0,
) -> AnalyzedStock:
    return AnalyzedStock(
        ticker=ticker,
        base_symbol=ticker.split("_")[0],
        yfinance_symbol=ticker.split("_")[0],
        name=ticker,
        score=score,
        atr20=atr20,
        current_price=price,
        sma100=price,
        max_gap_percent=None,
        slope=0.0,
        r_squared=1.0,
    )


class RebalanceOrdersTest(unittest.TestCase):
    def test_rebalance_buy_and_sell(self) -> None:
        ranked = [
            _stock("A_US_EQ", atr20=2.0, price=100.0, score=3.0),
            _stock("B_US_EQ", atr20=5.0, price=10.0, score=2.0),
            _stock("C_US_EQ", atr20=1.0, price=50.0, score=1.0),
        ]
        positions = {
            "A_US_EQ": 10.05,
            "B_US_EQ": 10.0,
            "C_US_EQ": 30.0,
        }
        new_buy_orders, rebalance_buy_orders, sell_orders, remaining_cash = portfolio.build_rebalance_orders(
            ranked,
            positions_by_ticker=positions,
            cash=500.0,
            total_equity=10000.0,
            risk_fraction=0.01,
            top_n=3,
            gap_lookback_days=90,
            rebalance_threshold=0.01,
            max_position_fraction=0.10,
        )

        self.assertEqual(len(new_buy_orders), 0)
        self.assertEqual([order.ticker for order in rebalance_buy_orders], ["B_US_EQ"])
        self.assertAlmostEqual(rebalance_buy_orders[0].quantity, 10.0, places=3)

        self.assertEqual([order.ticker for order in sell_orders], ["C_US_EQ"])
        self.assertAlmostEqual(sell_orders[0].quantity, -10.0, places=3)

        self.assertAlmostEqual(remaining_cash, 900.0, places=2)

    def test_partial_buy_when_cash_is_limited(self) -> None:
        ranked = [_stock("D_US_EQ", atr20=2.0, price=20.0, score=1.0)]
        new_buy_orders, rebalance_buy_orders, sell_orders, remaining_cash = portfolio.build_rebalance_orders(
            ranked,
            positions_by_ticker={},
            cash=200.0,
            total_equity=10000.0,
            risk_fraction=0.01,
            top_n=1,
            gap_lookback_days=90,
            rebalance_threshold=0.01,
            max_position_fraction=0.10,
        )

        self.assertEqual(len(sell_orders), 0)
        self.assertEqual(len(rebalance_buy_orders), 0)
        self.assertEqual(len(new_buy_orders), 1)
        self.assertAlmostEqual(new_buy_orders[0].quantity, 10.0, places=3)
        self.assertAlmostEqual(remaining_cash, 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
