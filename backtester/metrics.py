"""Performance metrics for a BacktestResult.

All formulas are documented in README.md and verified against
hand-computed values on a 10-bar fixture in tests/test_backtester.py.

- Total return   : final_equity / initial_capital - 1
- CAGR           : (final/initial) ** (252 / n_bars) - 1
                   (252 trading days per year; the fixture tests pin the
                   behaviour for runs shorter than a year)
- Max drawdown   : max over t of (peak_t - equity_t) / peak_t
- DD duration    : longest number of consecutive bars below the running
                   peak (recovery to a new high resets the counter)
- Sharpe (daily) : mean(daily returns) / stdev(daily returns) * sqrt(252);
                   stdev uses n-1 degrees of freedom; risk-free = 0 by
                   default (an annualised rate can be passed in)
- Win rate       : winning trades / total closed trades
- Profit factor  : sum of winning P&L / abs(sum of losing P&L)
                   (inf-safe: 0 losers -> None, reported as "n/a")
"""

import math


def compute_metrics(result, risk_free_annual=0.0):
    """Return an ordered dict of metric name -> value (None if n/a)."""
    curve = result.equity_curve
    equities = [e for _, e in curve]
    n = len(equities)
    initial = result.config.get("initial_capital", 100000.0)
    final = equities[-1] if equities else initial

    # ---- return & CAGR --------------------------------------------------
    total_return = (final / initial - 1.0) if initial else 0.0
    if final > 0 and n > 0 and final != initial:
        trading_years = n / 252.0
        cagr = (final / initial) ** (1.0 / trading_years) - 1.0
    else:
        cagr = 0.0

    # ---- drawdown -------------------------------------------------------
    max_dd = 0.0            # as a fraction of peak
    max_dd_abs = 0.0        # in dollars
    dd_duration = 0         # bars
    peak = None
    underwater = 0
    durations = []
    for e in equities:
        if peak is None or e > peak:
            # new high: record the drought length and reset
            durations.append(underwater)
            underwater = 0
            peak = e
        else:
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_abs = peak - e
            underwater += 1
    durations.append(underwater)
    longest_dd_bars = max(durations) if durations else 0

    # ---- daily returns & Sharpe ------------------------------------------
    daily = []
    for i in range(1, n):
        if equities[i - 1] > 0:
            daily.append(equities[i] / equities[i - 1] - 1.0)
    sharpe = None
    if len(daily) >= 2:
        mean = sum(daily) / len(daily)
        var = sum((r - mean) ** 2 for r in daily) / (len(daily) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            rf_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
            sharpe = (mean - rf_daily) / sd * math.sqrt(252.0)

    # ---- trade stats -------------------------------------------------------
    trades = result.trades
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    win_rate = (len(wins) / len(trades) * 100.0) if trades else None
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    avg_win = (gross_profit / len(wins)) if wins else None
    avg_loss = (-gross_loss / len(losses)) if losses else None

    return {
        "initial_capital": initial,
        "final_equity": final,
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "max_drawdown_abs": max_dd_abs,
        "longest_drawdown_bars": longest_dd_bars,
        "sharpe": sharpe,
        "trade_count": len(trades),
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "bars": n,
    }


def format_metrics(m):
    """Render the metrics dict as a plain-text summary block."""
    def pct(v):
        return "n/a" if v is None else "%.2f%%" % v

    def num(v, d=2):
        return "n/a" if v is None else "%.*f" % (d, v)

    lines = [
        "Backtest summary",
        "=================",
        "Bars processed        : %d" % m["bars"],
        "Initial capital       : $%.2f" % m["initial_capital"],
        "Final equity          : $%.2f" % m["final_equity"],
        "Total return          : %s" % pct(m["total_return_pct"]),
        "CAGR                  : %s" % pct(m["cagr_pct"]),
        "Max drawdown          : %s (peak-to-trough $%.2f)"
        % (pct(m["max_drawdown_pct"]), m["max_drawdown_abs"]),
        "Longest drawdown      : %d bars" % m["longest_drawdown_bars"],
        "Sharpe ratio (daily)  : %s" % num(m["sharpe"]),
        "Trades                : %d" % m["trade_count"],
        "Win rate              : %s" % pct(m["win_rate_pct"]),
        "Profit factor         : %s" % num(m["profit_factor"]),
        "Avg win               : $%s" % num(m["avg_win"]),
        "Avg loss              : $%s" % num(m["avg_loss"]),
    ]
    return "\n".join(lines)