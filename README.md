# Stock/ETF Backtesting Engine (Pure Python Stdlib)

A rule-driven backtesting engine that turns trading rules into software
testable against historical stock and ETF data. Define entries, exits and
position sizing in a single `config.json`, run one command, and get a full
performance report: equity curve, drawdown, Sharpe, win rate and the
complete trade log.

**Zero third-party dependencies.** Everything runs on the Python standard
library (json, csv, math, collections, argparse, unittest) — nothing to
install beyond Python itself, fully offline, no problem on Windows 10.

---

## 1. Requirements & Setup

| Requirement | Version |
|---|---|
| Python | 3.8 or newer (tested on 3.14) |
| Third-party packages | **None** |

```bat
:: from the project root
python run.py --config config.json
```

That is the entire setup. No venv, no pip install, no network access
needed at run time.

## 2. How to Run

```bash
# run the bundled sample strategy (SPY daily data)
python run.py --config config.json

# override the data file (any Date,Open,High,Low,Close[,Volume] CSV)
python run.py --config config.json --data data/sample_spy.csv

# write reports somewhere else / set a non-zero risk-free rate
python run.py --config config.json --output output_alt --risk-free 0.04

# run the test suite (33 tests)
python -m unittest discover -s tests -v
```

Output (into `output_dir`, default `output/`):

| File | Contents |
|---|---|
| `trades.csv` | Full trade log: entry/exit dates and prices, shares, P&L, return %, bars held, exit reason |
| `equity_curve.csv` | Mark-to-market equity after every bar's close |
| `report.html` | Self-contained HTML: metrics table, inline-SVG equity chart, full trade table, the config itself. No external CSS/JS/images — opens offline in any browser |

The console also prints a metrics summary.

## 3. Data Format

CSV with a header row, chronological order (oldest first):

```
Date,Open,High,Low,Close,Volume
2023-11-02,426.58,430.92,426.56,430.76,94938900
```

`Volume` is optional. Malformed rows are skipped with a warning.
Bundled: `data/sample_spy.csv` (3 years of real SPY daily bars) and
`data/sample_tiny.csv` (10-bar hand-checkable fixture used by the tests).

## 4. Execution Model (what the engine actually simulates)

- Long-only, one position at a time, bar-by-bar over daily OHLCV.
- **Entries** are evaluated on the bar CLOSE and filled at that close
  (a close-based signal is only knowable once the bar closes).
- **Protective exits** (stop-loss, take-profit, trailing stop) are checked
  INTRABAR against the bar's low/high, starting the bar AFTER entry.
  Fills use the trigger price; if the bar gaps through the trigger at the
  open, the fill is the open (what a live order would get).
- If one bar touches BOTH stop and target, the stop is assumed first
  (pessimistic, standard for daily data).
- **Close-based exits** (opposite signal, max-bars, exit-signal rules)
  fill at the close.
- An open position at the end of the data is force-closed (`end_of_data`)
  so metrics reflect realised cash.
- All fills are rounded to the cent before P&L is computed, so the trade
  log is arithmetically consistent by hand: `pnl = shares × (exit − entry) − fees`.

## 5. Config Reference (every parameter)

Top level:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `data_file` | string | — | CSV to backtest, relative to the config file |
| `output_dir` | string | `output` | Where reports are written |
| `initial_capital` | number | `100000` | Starting cash |
| `commission_per_share` | number | `0` | Dollars per share per fill (entry and exit each) |
| `slippage_pct` | number | `0` | % added to buy fills / shaved off sell fills |

`position_sizing`:

| Parameter | Example | Meaning |
|---|---|---|
| `mode` | `"percent_of_equity"` | or `"fixed_capital"` |
| `value` | `95` | percent of current equity (95%), or a fixed dollar budget |

`entry_rules`: a list of rule objects — **ALL must be true on the same
bar** to enter. Combine any of:

| Type | Example | Fires when |
|---|---|---|
| `breakout` | `{"type":"breakout","days":20,"basis":"high"}` | close strictly above the highest HIGH of the previous N bars (`basis:"close"` → highest CLOSE) |
| `crossover` | `{"type":"crossover","fast":"ema_9","slow":"ema_21","direction":"above"}` | fast series crosses above (`"below"` → below) slow; a true cross: holds now, did not hold on the prior bar |
| `rsi_threshold` | `{"type":"rsi_threshold","period":14,"op":"<","threshold":30}` | RSI(N) compared to a constant |
| `indicator_compare` | `{"type":"indicator_compare","left":{"kind":"close"},"op":">","right":{"kind":"indicator","name":"sma_50"}}` | generic primitive: any operand vs any operand |

Operands (`left`/`right`): `{"kind":"close"}`, `{"kind":"open"}`,
`{"kind":"high"}`, `{"kind":"low"}`, `{"kind":"value","value":42}`, or
`{"kind":"indicator","name":"<indicator>"}`.
Operators (`op`): `>`, `<`, `>=`, `<=`.

Indicator names: `sma_N`, `ema_N`, `rsi_N`, `atr_N`, `hh_N` (N-day
highest high, prior bars), `ll_N` (lowest low), `hc_N` (highest close).
Indicator series are computed once and cached.

`exit_rules`:

| Parameter | Type | Example | Meaning |
|---|---|---|---|
| `stop_loss_pct` | number or `null` | `8` | sell if price falls N% below entry fill (null = off) |
| `take_profit_pct` | number or `null` | `15` | sell if price rises N% above entry fill (null = off) |
| `trailing_stop_pct` | number or `null` | `5` | trail N% below the highest CLOSE since entry, ratchets up only (null = off) |
| `exit_on_opposite_signal` | bool | `false` | close when the entry rules fire again while in a trade |
| `max_bars_in_trade` | int or `null` | `20` | force-exit at the close after N bars held (null = off) |
| `signal_rules` | list | `[]` | exit-signal rules, same format as `entry_rules` (ANY one firing exits at the close) |

Exit priority within a bar: stop → target → trailing → opposite signal →
max bars → signal rules.

### Example config (as shipped in `config.json`)

```json
{
  "data_file": "data/sample_spy.csv",
  "output_dir": "output",
  "initial_capital": 100000,
  "position_sizing": {"mode": "percent_of_equity", "value": 95},
  "entry_rules": [{"type": "breakout", "days": 20, "basis": "high"}],
  "exit_rules": {
    "stop_loss_pct": 8,
    "take_profit_pct": 15,
    "trailing_stop_pct": 5,
    "exit_on_opposite_signal": false,
    "max_bars_in_trade": null,
    "signal_rules": []
  },
  "commission_per_share": 0,
  "slippage_pct": 0
}
```

## 6. Metric Formulas (exactly as implemented)

| Metric | Formula |
|---|---|
| Total return | `final_equity / initial_capital − 1` |
| CAGR | `(final/initial)^(252/bars) − 1` (252 trading days/year) |
| Max drawdown | `max over t of (peak_t − equity_t) / peak_t`, running peak |
| Longest drawdown | most consecutive bars below the running peak |
| Sharpe (daily) | `mean(daily returns − rf_daily) / stdev(daily returns) × √252`; stdev with n−1; `rf_daily = (1+rf_annual)^(1/252) − 1`; risk-free 0 by default; `None` if stdev is 0 |
| Win rate | winning trades / total trades (a trade with P&L exactly 0 counts as a loss) |
| Profit factor | sum of winning P&L / |sum of losing P&L| (`None` if no losers) |

Every formula is pinned by unit tests against hand-computed values on a
10-bar fixture (`tests/test_backtester.py`).

## 7. Project Layout

```
run.py                    CLI entry point
config.json               example strategy (20-day breakout)
backtester/
  engine.py               bar-by-bar event loop, trades, equity curve
  strategy.py             config validation, rule evaluation, sizing
  indicators.py           SMA, EMA, RSI, ATR, highest-high/lowest-low
  metrics.py              performance metrics + text summary
  report.py               trades.csv / equity_curve.csv / report.html
data/
  sample_spy.csv          3 years of real SPY daily OHLCV
  sample_tiny.csv         10-bar hand-check fixture
tests/
  test_backtester.py      33 tests incl. a golden regression pin
output/                   generated reports (created on first run)
```

## 8. Extending

- **New indicator**: add a function in `indicators.py` (same-length
  list, `None` while warming up), add a name prefix in
  `strategy.py:Strategy.series()` — rules can use it immediately.
- **New rule type**: add a branch in `strategy.py:Strategy._compile()`
  that normalises to the `indicator_compare` primitive.
- **New metric**: add it in `metrics.py:compute_metrics()` and to the
  `nice` table in `report.py:write_html_report()`.
- **Short selling / multiple positions**: the event loop in
  `engine.py:Engine.run()` is the only place that needs changes.

## 9. Verification

- `python -m unittest discover -s tests -v` — 33 tests, all passing:
  indicator math, metrics math (hand-computed), exact entry/exit bars and
  fills (breakout fires exactly on the bar exceeding the N-day high;
  stop-loss fill correct to the cent), and a golden integration test
  pinning the sample run (9 trades, final equity $125,824.68) so any
  future change to the math must consciously update the pin.
- The first trade of the sample run hand-checks exactly:
  entry 2023-12-01 close $459.10 (close > 20-day high $458.32; 20-day
  breakout not yet defined on any earlier bar), 206 shares
  (95% of $100,000), trailing-stop exit 2024-04-19 at $497.01
  (5% below the $523.17 peak close), P&L 206 × ($497.01 − $459.10) =
  **$7,809.46** — matches `output/trades.csv` to the cent.

## 10. Notes & Limitations (honest scope)

- Long-only, single position — matches the client's price-action rules.
- Signals evaluate on closes; no intrabar tick data, so a bar touching
  both stop and target resolves pessimistically (stop first).
- No dividends/splits adjustment — use adjusted OHLC data for full
  accuracy on dividend-heavy instruments.
- Sample data is real SPY daily OHLC obtained from Yahoo Finance's public
  chart API (3 years, 752 bars) because stooq.com's CSV endpoint was
  behind a JavaScript bot-check at download time.