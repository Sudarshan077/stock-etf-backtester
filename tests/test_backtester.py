"""Test suite for the pure-stdlib backtester.

Three layers:
1. Indicator unit tests   -- hand-checked SMA/EMA/RSI/ATR/highest-high values.
2. Metrics unit tests     -- every formula verified against numbers computed
                             BY HAND on the 10-bar fixture.
3. Engine/strategy tests -- exact entry bar (breakout fires on the bar that
                             exceeds the N-day high), stop-loss fill math
                             correct to the cent, position sizing, and a
                             golden integration test on the bundled SPY data
                             pinning trade count + final equity (regression).

Run:  python -m unittest discover -s tests -v      (from the project root)
"""

import json
import os
import sys
import unittest

# Make the project root importable regardless of the caller's cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtester import engine, indicators, metrics as metrics_mod, report
from backtester.strategy import Strategy

DATA_TINY = os.path.join(ROOT, "data", "sample_tiny.csv")
DATA_SPY = os.path.join(ROOT, "data", "sample_spy.csv")


def tiny_bars():
    """The 10-bar fixture as Bar objects (see data/sample_tiny.csv)."""
    return engine.load_bars(DATA_TINY)


# ===========================================================================
# 1. INDICATORS
# ===========================================================================

class TestIndicators(unittest.TestCase):

    def test_sma_hand_checked(self):
        # closes 1..10 -> SMA(3): first at index 2 = (1+2+3)/3 = 2
        s = indicators.sma([1, 2, 3, 4, 5, 6], 3)
        self.assertEqual(s[:2], [None, None])
        self.assertEqual(s[2], 2.0)
        self.assertEqual(s[3], 3.0)
        self.assertEqual(s[5], 5.0)

    def test_sma_rejects_bad_period(self):
        with self.assertRaises(ValueError):
            indicators.sma([1, 2], 0)

    def test_ema_seed_and_recursion(self):
        # EMA(3) over [1,2,3,4]: seed = mean(1,2,3) = 2 at index 2;
        # alpha = 0.5; ema[3] = 0.5*4 + 0.5*2 = 3
        s = indicators.ema([1, 2, 3, 4], 3)
        self.assertEqual(s[:2], [None, None])
        self.assertAlmostEqual(s[2], 2.0)
        self.assertAlmostEqual(s[3], 3.0)

    def test_rsi_hand_checked(self):
        # 6 constant closes -> no gains, no losses -> RSI undefined until
        # warm; with avg_loss 0 RSI = 100.  Series with only gains:
        closes = [10, 11, 12, 13, 14, 15]  # RSI(5) -> avg_gain>0, loss=0
        s = indicators.rsi(closes, 5)
        self.assertIsNone(s[4])
        self.assertEqual(s[5], 100.0)

    def test_rsi_all_losers_is_zero(self):
        closes = [15, 14, 13, 12, 11, 10]
        s = indicators.rsi(closes, 5)
        self.assertEqual(s[5], 0.0)

    def test_atr_hand_checked(self):
        # 3 bars: closes 100,100,100; highs 102,102,102; lows 98,98,98
        # TR_1 = max(4, |102-100|, |98-100|) = 4; ATR(2) = mean(TR_1..TR_2)=4
        highs = [102.0, 102.0, 102.0, 102.0]
        lows = [98.0, 98.0, 98.0, 98.0]
        closes = [100.0, 100.0, 100.0, 100.0]
        s = indicators.atr(highs, lows, closes, 2)
        self.assertIsNone(s[1])
        self.assertAlmostEqual(s[2], 4.0)
        self.assertAlmostEqual(s[3], 4.0)  # Wilder: (4*1+4)/2 = 4

    def test_highest_high_excludes_current_bar(self):
        # Window = previous N bars only, current bar EXCLUDED; first
        # value at index N.
        highs = [1, 5, 3, 2, 9]
        s = indicators.highest_high(highs, 3)
        # i=3: max(1,5,3) = 5 ; i=4: max(5,3,2) = 5 (the 9 is bar 4
        # itself and must NOT be in its own window)
        self.assertEqual(s[:3], [None, None, None])
        self.assertEqual(s[3], 5)
        self.assertEqual(s[4], 5)

    def test_lowest_low_excludes_current_bar(self):
        lows = [8, 2, 6, 7, 1]
        s = indicators.lowest_low(lows, 3)
        # i=3: min(8,2,6)=2 ; i=4: min(2,6,7)=2 (current bar's 1 excluded)
        self.assertEqual(s[3], 2)
        self.assertEqual(s[4], 2)


# ===========================================================================
# 2. METRICS (hand-computed on the tiny fixture)
# ===========================================================================

class TestMetricsHandChecked(unittest.TestCase):
    """Every value below was computed by hand; see comments."""

    def make_result(self, equities, trades=None, capital=1000.0):
        dates = ["d%02d" % i for i in range(len(equities))]
        r = engine.BacktestResult(
            bars=[], trades=trades or [],
            equity_curve=list(zip(dates, equities)),
            config={"initial_capital": capital})
        return r

    def test_total_return_and_cagr(self):
        # 1000 -> 1100 over 10 bars: return 10%.
        # CAGR = (1.1)^(252/10) - 1  (hand: ln(1.1)=0.09531; x25.2=2.4019;
        # e^2.4019 ~ 11.045 -> 1004.5%).
        r = self.make_result([1000, 1000, 1000, 1000, 1000,
                              1000, 1000, 1000, 1000, 1100])
        m = metrics_mod.compute_metrics(r)
        self.assertAlmostEqual(m["total_return_pct"], 10.0)
        expected_cagr = ((1100.0 / 1000.0) ** (252.0 / 10.0) - 1.0) * 100
        self.assertAlmostEqual(m["cagr_pct"], expected_cagr, places=6)

    def test_max_drawdown_hand_checked(self):
        # Peak 100 at index 2, trough 80 at index 4 -> DD = 20%.
        eq = [90.0, 95.0, 100.0, 85.0, 80.0, 90.0, 105.0]
        r = self.make_result(eq, capital=90.0)
        m = metrics_mod.compute_metrics(r)
        self.assertAlmostEqual(m["max_drawdown_pct"], 20.0)
        self.assertAlmostEqual(m["max_drawdown_abs"], 20.0)
        # Longest time below a running peak: indices 3,4,5 -> 3 bars.
        self.assertEqual(m["longest_drawdown_bars"], 3)

    def test_sharpe_hand_checked(self):
        # Constant daily return +1% on 4 bars -> 3 daily returns all 0.01:
        # stdev = 0 -> Sharpe undefined (None).  Instead use two distinct
        # returns and verify by hand:
        # equities 100,101,99,102 -> returns +0.01, -0.019802, +0.030303
        eq = [100.0, 101.0, 99.0, 102.0]
        r = self.make_result(eq, capital=100.0)
        m = metrics_mod.compute_metrics(r)
        rets = [101 / 100 - 1, 99 / 101 - 1, 102 / 99 - 1]
        mean = sum(rets) / 3
        var = sum((x - mean) ** 2 for x in rets) / 2  # n-1
        import math
        expected = mean / math.sqrt(var) * math.sqrt(252)
        self.assertAlmostEqual(m["sharpe"], expected, places=9)

    def test_sharpe_zero_variance_is_none(self):
        r = self.make_result([100, 101, 102.01])
        m = metrics_mod.compute_metrics(r)
        # returns +1%, +1% -> stdev 0 -> None
        self.assertIsNone(m["sharpe"])

    def test_trade_stats_hand_checked(self):
        # Two trades: +50 and -25.
        t1 = engine.Trade("a", 100.0, 1, 0.0)
        t1.exit_date, t1.exit_price, t1.pnl = "b", 150.0, 50.0
        t2 = engine.Trade("c", 100.0, 1, 0.0)
        t2.exit_date, t2.exit_price, t2.pnl = "d", 75.0, -25.0
        r = self.make_result([1000, 1025], trades=[t1, t2])
        m = metrics_mod.compute_metrics(r)
        self.assertEqual(m["trade_count"], 2)
        self.assertAlmostEqual(m["win_rate_pct"], 50.0)
        self.assertAlmostEqual(m["profit_factor"], 50.0 / 25.0)
        self.assertAlmostEqual(m["avg_win"], 50.0)
        self.assertAlmostEqual(m["avg_loss"], -25.0)

    def test_profit_factor_no_losers(self):
        t = engine.Trade("a", 100.0, 1, 0.0)
        t.exit_date, t.exit_price, t.pnl = "b", 110.0, 10.0
        r = self.make_result([1000, 1010], trades=[t])
        m = metrics_mod.compute_metrics(r)
        self.assertIsNone(m["profit_factor"])
        self.assertAlmostEqual(m["win_rate_pct"], 100.0)


# ===========================================================================
# 3. STRATEGY SIGNALS (exact bars, exact fills)
# ===========================================================================

class TestStrategySignals(unittest.TestCase):

    def BREAKOUT_CFG(self, days=3):
        return {
            "initial_capital": 10000,
            "entry_rules": [{"type": "breakout", "days": days,
                             "basis": "high"}],
            "exit_rules": {},
            "position_sizing": {"mode": "fixed_capital", "value": 10000},
        }

    def test_breakout_fires_exactly_when_close_exceeds_prior_high(self):
        """Fixture: closes 100..105 (step +0.5).  hh_3 at bar i is the max
        high of bars i-3..i-1.  Hand check:
          bar 3: hh_3 = max(101,102,102.5)? highs are 101,102,102.5 -> 102.5;
                 close 102.00 <= 102.50 -> NO entry.
          bar 4: window highs 102,102.5,103 -> close 102.50 <= 103 -> NO.
          bar 5: window highs 102.5,103,103.5 -> close 103.00 <= 103.5 -> NO.
          bar 6: window 103,103.5,104 -> close 103.50 <= 104 -> NO.
          bar 7: window 103.5,104,104.5 -> close 104.00 <= 104.5 -> NO.
          bar 8: window 104,104.5,105 -> close 104.50 <= 105 -> NO.
          bar 9: window 104.5,105,105.5 -> close 105.00 <= 105.5 -> NO.
        With monotonically rising data the close NEVER exceeds the prior
        high -- so this must produce ZERO trades (a key sanity property).
        """
        eng = engine.Engine(self.BREAKOUT_CFG(3))
        result = eng.run(tiny_bars())
        self.assertEqual(len(result.trades), 0)

    def test_breakout_triggers_on_the_exceeding_bar(self):
        """Same fixture but with a spike bar appended: close 110 at bar 10
        exceeds every prior high (max 105.5) -> entry fires EXACTLY at
        bar 10, at that bar's close (110.00), no earlier."""
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        cfg = self.BREAKOUT_CFG(3)
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        self.assertEqual(len(result.trades), 1)  # end_of_data force-close
        t = result.trades[0]
        self.assertEqual(t.entry_date, "2024-01-16")
        self.assertEqual(t.entry_price, 110.0)
        self.assertEqual(t.bars_held, 0)
        self.assertEqual(t.exit_reason, "end_of_data")

    def test_stop_loss_fill_correct_to_the_cent(self):
        """Entry at 110.00 with 8% stop -> stop = 101.20.
        Next bar trades down to low 101.00 (< 101.20): the stop fills at
        101.20 (the trigger, since open 108 >= trigger).  P&L with 100
        shares (fixed 10000 capital / 110 = 90 shares):
        budget 10000 -> int(10000/110) = 90 shares.
        P&L = 90 * (101.20 - 110.00) = 90 * -8.80 = -792.00 exactly."""
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        bars.append(engine.Bar("2024-01-17", 108.0, 108.0, 101.00, 101.50, 1))
        cfg = self.BREAKOUT_CFG(3)
        cfg["exit_rules"] = {"stop_loss_pct": 8}
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        self.assertEqual(len(result.trades), 1)
        t = result.trades[0]
        self.assertEqual(t.exit_reason, "stop_loss")
        self.assertEqual(t.exit_date, "2024-01-17")
        self.assertAlmostEqual(t.exit_price, 101.20, places=2)
        self.assertEqual(t.shares, 90)
        self.assertAlmostEqual(t.pnl, -792.00, places=2)
        # Final equity: 10000 - 792 = 9208.00.
        self.assertAlmostEqual(result.equity_curve[-1][1], 9208.00, places=2)

    def test_stop_gap_down_fills_at_open(self):
        """If the bar OPENS below the stop (gap through), a live stop fills
        at the open: entry 110, stop 101.20, next bar opens at 99 ->
        fill 99.00 (not 101.20)."""
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        bars.append(engine.Bar("2024-01-17", 99.0, 100.0, 98.0, 99.5, 1))
        cfg = self.BREAKOUT_CFG(3)
        cfg["exit_rules"] = {"stop_loss_pct": 8}
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        t = result.trades[0]
        self.assertEqual(t.exit_reason, "stop_loss")
        self.assertEqual(t.exit_price, 99.0)

    def test_take_profit_fill(self):
        """Entry 110, 15% target -> 126.50.  Next bar high 127.00 touches
        it -> fills at 126.50."""
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        bars.append(engine.Bar("2024-01-17", 111.0, 127.0, 110.0, 126.0, 1))
        cfg = self.BREAKOUT_CFG(3)
        cfg["exit_rules"] = {"take_profit_pct": 15}
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        t = result.trades[0]
        self.assertEqual(t.exit_reason, "take_profit")
        self.assertAlmostEqual(t.exit_price, 126.50, places=2)
        # 90 shares * (126.50-110) = +1485.00
        self.assertAlmostEqual(t.pnl, 1485.00, places=2)

    def test_trailing_stop_ratchets(self):
        """Entry 110 (trailing 5% -> initial trail = 110*0.95 = 104.50).
        Bar rises to close 120 -> trail ratchets to 120*0.95 = 114.00.
        Next bar low 113 touches 114 -> trailing_stop exit at 114.00."""
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        bars.append(engine.Bar("2024-01-17", 111.0, 121.0, 110.0, 120.0, 1))
        bars.append(engine.Bar("2024-01-18", 119.0, 119.0, 113.0, 114.5, 1))
        cfg = self.BREAKOUT_CFG(3)
        cfg["exit_rules"] = {"trailing_stop_pct": 5}
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        t = result.trades[0]
        self.assertEqual(t.exit_reason, "trailing_stop")
        self.assertAlmostEqual(t.exit_price, 114.00, places=2)

    def test_max_bars_exit(self):
        """max_bars_in_trade = 2 -> force-exit at the close exactly 2 bars
        after entry (bars held = 2)."""
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        bars.append(engine.Bar("2024-01-17", 111.0, 112.0, 110.0, 111.0, 1))
        bars.append(engine.Bar("2024-01-18", 111.0, 112.0, 110.0, 111.5, 1))
        bars.append(engine.Bar("2024-01-19", 111.0, 112.0, 110.0, 112.0, 1))
        cfg = self.BREAKOUT_CFG(3)
        cfg["exit_rules"] = {"max_bars_in_trade": 2}
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        t = result.trades[0]
        self.assertEqual(t.exit_reason, "max_bars")
        self.assertEqual(t.bars_held, 2)
        self.assertEqual(t.exit_date, "2024-01-18")
        self.assertEqual(t.exit_price, 111.5)

    def test_percent_of_equity_sizing(self):
        """95% of 10000 = 9500 budget; entry 110 -> int(9500/110)=86 shares."""
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        cfg = self.BREAKOUT_CFG(3)
        cfg["position_sizing"] = {"mode": "percent_of_equity", "value": 95}
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        self.assertEqual(result.trades[0].shares, 86)

    def test_rsi_threshold_rule(self):
        """RSI(5) on the fixture: 5 straight gains -> RSI hits 100 at the
        6th bar; rule RSI > 70 must fire there and only there."""
        cfg = {
            "initial_capital": 10000,
            "entry_rules": [{"type": "rsi_threshold", "period": 5,
                             "op": ">", "threshold": 70}],
            "exit_rules": {},
            "position_sizing": {"mode": "fixed_capital", "value": 10000},
        }
        eng = engine.Engine(cfg)
        result = eng.run(tiny_bars())
        # RSI first defined at index 5 (bar 2024-01-09) with value 100.
        self.assertEqual(result.trades[0].entry_date, "2024-01-09")

    def test_multiple_rules_are_anded(self):
        """breakout(3) AND RSI>70: the RSI rule alone fires on 2024-01-09
        (first bar with RSI 100), but the breakout never fires on the
        steadily rising fixture -> combined entry must NEVER happen."""
        cfg = self.BREAKOUT_CFG(3)
        cfg["entry_rules"].append({"type": "rsi_threshold", "period": 5,
                                    "op": ">", "threshold": 70})
        eng = engine.Engine(cfg)
        result = eng.run(tiny_bars())
        self.assertEqual(len(result.trades), 0)

    def test_crossover_rule(self):
        """SMA(2) crossing above SMA(4) on the fixture:
        sma2 defined from idx1, sma4 from idx3.  Cross event needs
        sma2>sma4 now and not before.  Values (hand):
          idx3: sma2=101.75? -> sma2(101.5+102)/2? closes:100,101,101.5,102
          idx3: sma2=(101.5+102)/2=101.75, sma4=(100+101+101.5+102)/4=101.125
               -> holds now; idx2: sma2 undefined -> no cross at 3.
          idx4: sma2=(102+102.5)/2=102.25, sma4=101.75 -> holds now AND
               idx3 held -> not a cross.  Same for all later bars (both
               rise, sma2 stays above, no re-cross).
        So with strict cross semantics on monotonically rising data the
        ONLY cross moment is idx3, where the prior value is undefined --
        hence zero entries.  A state comparison would fire every bar.
        """
        cfg = {
            "initial_capital": 10000,
            "entry_rules": [{"type": "crossover", "fast": "sma_2",
                             "slow": "sma_4", "direction": "above"}],
            "exit_rules": {},
            "position_sizing": {"mode": "fixed_capital", "value": 10000},
        }
        strat = Strategy(cfg)
        bars = tiny_bars()
        fires = [i for i in range(len(bars)) if strat.entry_signal(bars, i)]
        self.assertEqual(fires, [])

    def test_crossover_fires_on_true_cross(self):
        """Sawtooth data where sma_2 really crosses above sma_4:
        closes: 10,10,10,10, 9, 9, 12  -- sma2 dips below sma4 then
        crosses back up at the last bar."""
        bars = [engine.Bar("d%d" % i, c, c + 0.5, c - 0.5, c, 1)
                for i, c in enumerate([10, 10, 10, 10, 9, 9, 12])]
        cfg = {
            "initial_capital": 10000,
            "entry_rules": [{"type": "crossover", "fast": "sma_2",
                             "slow": "sma_4", "direction": "above"}],
            "exit_rules": {},
            "position_sizing": {"mode": "fixed_capital", "value": 10000},
        }
        strat = Strategy(cfg)
        fires = [i for i in range(len(bars)) if strat.entry_signal(bars, i)]
        # Hand: sma4(6)= (10+10+9+9)/4? closes idx3..6 = 10,9,9,12 -> 10.0
        #       sma2(6) = (9+12)/2 = 10.5 > 10.0 holds
        #       sma4(5) = (10+10+9+9)/4 = 9.5 ; sma2(5)=(9+9)/2=9.0 < 9.5
        #       -> prior bar did NOT hold -> TRUE cross at idx 6.
        self.assertEqual(fires, [6])

    def test_exit_on_opposite_signal(self):
        """Entry via RSI>70 (fires 2024-01-09); exit_on_opposite_signal
        closes the trade the next time the entry fires (2024-01-10),
        because entry rules stay true every subsequent bar."""
        cfg = {
            "initial_capital": 10000,
            "entry_rules": [{"type": "rsi_threshold", "period": 5,
                             "op": ">", "threshold": 70}],
            "exit_rules": {"exit_on_opposite_signal": True},
            "position_sizing": {"mode": "fixed_capital", "value": 10000},
        }
        eng = engine.Engine(cfg)
        result = eng.run(tiny_bars())
        t = result.trades[0]
        self.assertEqual(t.entry_date, "2024-01-09")
        self.assertEqual(t.exit_reason, "opposite_signal")
        self.assertEqual(t.exit_date, "2024-01-10")

    def test_commission_and_slippage_math(self):
        """$0.10/share commission each way + 0.5% slippage.
        Hand math (fills rounded to the cent BEFORE P&L, as real
        markets fill in cents):
          entry fill = round(110 * 1.005, 2) = 110.55
          stop       = 110.55 * (1 - 8/100) = 101.706
          exit fill  = round(101.706 * 0.995, 2) = 101.20
          shares     = int(10000 / 110.55) = 90
          fees       = 90*0.10 entry + 90*0.10 exit = 18.00
          pnl = 90*(101.20 - 110.55) - 18 = -859.50
        """
        bars = tiny_bars()
        bars.append(engine.Bar("2024-01-16", 105.5, 110.0, 105.0, 110.0, 1))
        bars.append(engine.Bar("2024-01-17", 108.0, 108.0, 101.00, 101.50, 1))
        cfg = self.BREAKOUT_CFG(3)
        cfg["exit_rules"] = {"stop_loss_pct": 8}
        cfg["commission_per_share"] = 0.10
        cfg["slippage_pct"] = 0.5
        eng = engine.Engine(cfg)
        result = eng.run(bars)
        t = result.trades[0]
        self.assertEqual(t.shares, 90)
        self.assertAlmostEqual(t.entry_price, 110.55, places=2)
        self.assertAlmostEqual(t.exit_price, 101.20, places=2)
        self.assertAlmostEqual(t.pnl, -859.50, places=2)

    def test_invalid_configs_raise(self):
        with self.assertRaises(ValueError):
            Strategy({"entry_rules": []})                      # no rules
        with self.assertRaises(ValueError):
            Strategy({"entry_rules": [{"type": "nope"}]})      # bad type
        with self.assertRaises(ValueError):
            Strategy({"entry_rules": [{"type": "breakout",
                                        "days": 3, "basis": "median"}]})
        with self.assertRaises(ValueError):
            Strategy({"entry_rules": [{"type": "breakout", "days": 3}],
                      "position_sizing": {"mode": "moon"}})


# ===========================================================================
# 4. GOLDEN INTEGRATION (regression pin on the bundled SPY data)
# ===========================================================================

class TestGoldenIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "config.json")) as f:
            cls.cfg = json.load(f)
        eng = engine.Engine(cls.cfg)
        bars = engine.load_bars(DATA_SPY)
        cls.result = eng.run(bars)
        cls.metrics = metrics_mod.compute_metrics(cls.result)

    def test_exact_trade_count_and_final_equity(self):
        """REGRESSION PIN: any change in engine/strategy/indicator math
        that alters the sample run must trip this test deliberately.
        Pinned for the bundled 3-year SPY daily data (752 bars,
        2023-11-02 .. 2026-11-02) with config.json as shipped:
        9 trades, final equity 125,824.68."""
        self.assertEqual(len(self.result.trades), self.metrics["trade_count"])
        self.assertEqual(self.metrics["trade_count"], 9)
        self.assertAlmostEqual(self.metrics["final_equity"],
                               125824.68, places=2)

    def test_every_trade_has_valid_roundtrip(self):
        for t in self.result.trades:
            self.assertIsNotNone(t.entry_date)
            self.assertIsNotNone(t.exit_date)
            self.assertGreater(t.shares, 0)
            self.assertGreater(t.bars_held, 0)
            self.assertIn(t.exit_reason,
                          ("stop_loss", "take_profit", "trailing_stop",
                           "opposite_signal", "max_bars", "signal_rule",
                           "end_of_data"))

    def test_equity_curve_matches_trades(self):
        """Sanity: final equity = initial + sum of trade P&L (long-only,
        one position at a time, fully realised at end of data)."""
        total = self.cfg["initial_capital"] + sum(
            t.pnl for t in self.result.trades)
        self.assertAlmostEqual(self.result.equity_curve[-1][1], total,
                               places=2)

    def test_reports_writable(self):
        """The full report pipeline runs and produces non-empty files."""
        import tempfile
        import shutil
        tmp = tempfile.mkdtemp(dir=os.path.join(ROOT, "tests"))
        try:
            paths = report.write_all_reports(
                self.result, self.metrics, tmp, data_file="sample_spy.csv")
            for p in paths.values():
                self.assertTrue(os.path.isfile(p))
                self.assertGreater(os.path.getsize(p), 100)
            with open(paths["report_html"], encoding="utf-8") as f:
                html = f.read()
            self.assertIn("<svg", html)          # inline chart present
            self.assertIn("Trade Log", html)
            # Self-contained: no external scripts, stylesheets, images
            # or links (the SVG xmlns namespace is fine, it's not a fetch).
            self.assertNotIn("<script", html)
            self.assertNotIn("<link", html)
            self.assertNotIn("<img", html)
            self.assertNotIn("src=", html)
            self.assertNotIn("href=", html)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)