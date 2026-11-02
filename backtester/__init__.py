"""Backtester package: a pure-stdlib stock/ETF backtesting engine.

Modules
-------
indicators : hand-written SMA, EMA, RSI, ATR, highest-high, lowest-low
strategy   : config-driven entry/exit rule evaluation + position sizing
engine     : bar-by-bar event loop, trades, equity curve
metrics    : performance statistics (return, CAGR, drawdown, Sharpe, ...)
report     : trades.csv / equity_curve.csv / self-contained report.html
"""

__version__ = "1.0.0"