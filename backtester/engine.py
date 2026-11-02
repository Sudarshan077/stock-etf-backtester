"""Bar-by-bar backtest engine over OHLCV data.

Event model
-----------
For every bar, in this exact order:

1. EXIT an open position if any exit fires:
   - intrabar: stop-loss / take-profit / trailing stop, checked against
     the bar's low/high from the bar AFTER entry (a position filled at
     bar i's close cannot be stopped out by bar i's own range);
   - on close: opposite entry signal, max-bars-in-trade, exit-signal rules.
2. Update the trailing-stop watermark with this bar's close.
3. If flat and ALL entry rules hold at this bar's close, ENTER at the
   close (you can only know a close-based signal once the bar closes).
4. Record equity at this bar's close.

Conservatism (standard for close-only data, no tick data):
- If a bar's range touches BOTH stop and target, the STOP is assumed to
  fill first (worst case).
- Stop/target/trailing fills use the trigger price; if the bar gaps
  through the trigger (open beyond it), the fill is the bar's OPEN,
  which is what a live order would realistically get.

Costs: `commission_per_share` (dollars/share/fill) and `slippage_pct`
(percent added to buys, shaved off sells) both default to 0 so the
hand-checked tests stay exact.
"""

from .strategy import Strategy


class Bar(object):
    """One OHLCV bar; values are floats, volume int, date a string."""

    __slots__ = ("date", "open", "high", "low", "close", "volume")

    def __init__(self, date, open_, high, low, close, volume):
        self.date = date
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class Trade(object):
    """A completed round-trip trade with realised P&L.

    pnl includes commissions and slippage effects (both are embedded in
    the fill prices); return_pct = pnl / total cost of entry.
    """

    __slots__ = ("entry_date", "entry_price", "shares", "exit_date",
                 "exit_price", "pnl", "return_pct", "bars_held",
                 "exit_reason", "entry_fee")

    def __init__(self, entry_date, entry_price, shares, entry_fee=0.0):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.return_pct = 0.0
        self.bars_held = 0
        self.exit_reason = None
        self.entry_fee = entry_fee


class BacktestResult(object):
    """Everything one run produced; feeds metrics and report writers."""

    def __init__(self, bars, trades, equity_curve, config):
        self.bars = bars                  # list[Bar]
        self.trades = trades              # list[Trade]
        self.equity_curve = equity_curve  # list[(date, equity)] per bar
        self.config = config


def load_bars(path):
    """Load a Date,Open,High,Low,Close[,Volume] CSV into list[Bar].

    Malformed rows (bad numbers, high<low) are skipped with a warning
    rather than aborting the run.  The file must be chronological.
    """
    import csv
    bars = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fields = [n.strip() for n in (reader.fieldnames or [])]
        required = {"Date", "Open", "High", "Low", "Close"}
        missing = required - set(fields)
        if missing:
            raise ValueError("CSV %r must have columns "
                             "Date,Open,High,Low,Close[,Volume]; missing: %s"
                             % (path, sorted(missing)))
        for lineno, row in enumerate(reader, start=2):
            row = {k.strip(): v for k, v in row.items() if k is not None}
            try:
                bar = Bar(row["Date"].strip(),
                          float(row["Open"]), float(row["High"]),
                          float(row["Low"]), float(row["Close"]),
                          int(float(row.get("Volume") or 0)))
            except (KeyError, ValueError, TypeError):
                print("WARNING: skipping malformed CSV row %d" % lineno)
                continue
            if bar.high < bar.low:
                print("WARNING: skipping row %d (high < low)" % lineno)
                continue
            bars.append(bar)
    if not bars:
        raise ValueError("No usable rows found in %r" % path)
    return bars


class Engine(object):
    """Drives the bar loop; all strategy decisions come from `Strategy`."""

    def __init__(self, config):
        self.config = config
        self.initial_capital = float(config.get("initial_capital", 100000.0))
        self.commission = float(config.get("commission_per_share", 0.0))
        self.slippage_pct = float(config.get("slippage_pct", 0.0))
        self.strategy = Strategy(config)

        exits = config.get("exit_rules", {})

        def opt(v):
            return None if v is None else float(v)

        self.stop_loss_pct = opt(exits.get("stop_loss_pct"))
        self.take_profit_pct = opt(exits.get("take_profit_pct"))
        self.trailing_stop_pct = opt(exits.get("trailing_stop_pct"))
        self.exit_on_opposite_signal = bool(
            exits.get("exit_on_opposite_signal", False))
        mb = exits.get("max_bars_in_trade")
        self.max_bars_in_trade = None if mb is None else int(mb)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _protective_fill(trigger, bar, is_stop):
        """Fill price for a stop/target touched this bar.

        If the bar GAPS through the trigger at the open, a live order
        fills at the open (worse than trigger for stops, better for
        targets is impossible -- gap direction handles both).
        """
        if is_stop:
            # Sell-side trigger below price: gap down -> fill at open.
            if bar.open < trigger:
                return bar.open
            return trigger
        # Buy-side target above price: gap up -> fill at open.
        if bar.open > trigger:
            return bar.open
        return trigger

    def _close_trade(self, position, exit_index, bar, fill_price,
                     reason, trades):
        """Realise the open position at `fill_price`; append to trades.

        Fills are rounded to the cent BEFORE P&L is computed (real
        markets fill in cents), so the trade log is internally
        consistent: pnl == shares * (exit_price - entry_price) to the
        cent for zero-cost configs.
        """
        fill = round(fill_price * (1.0 - self.slippage_pct / 100.0), 2)
        proceeds = position["shares"] * fill
        fee = position["shares"] * self.commission
        self.cash += proceeds - fee
        cost = position["shares"] * position["entry_price"]
        total_cost = cost + position["entry_fee"]
        trade = Trade(position["entry_date"], position["entry_price"],
                      position["shares"], position["entry_fee"])
        trade.exit_date = bar.date
        trade.exit_price = round(fill, 2)
        trade.pnl = proceeds - fee - total_cost
        trade.return_pct = trade.pnl / total_cost * 100.0
        trade.bars_held = exit_index - position["entry_index"]
        trade.exit_reason = reason
        trades.append(trade)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def run(self, bars):
        """Execute the backtest over chronological `bars`."""
        self.cash = self.initial_capital
        position = None
        equity_curve = []
        trades = []
        watermark = None   # highest close since entry (trailing stop)

        for i, bar in enumerate(bars):

            # -- 1. exits for an open position ---------------------------
            if position is not None:
                fill_price = None
                reason = None
                if i > position["entry_index"]:
                    stop = position["stop_price"]
                    if stop is not None and bar.low <= stop:
                        fill_price = self._protective_fill(stop, bar, True)
                        reason = "stop_loss"
                    if fill_price is None:
                        target = position["target_price"]
                        if target is not None and bar.high >= target:
                            fill_price = self._protective_fill(
                                target, bar, False)
                            reason = "take_profit"
                    if fill_price is None:
                        trail = position["trail_price"]
                        if trail is not None and bar.low <= trail:
                            fill_price = self._protective_fill(
                                trail, bar, True)
                            reason = "trailing_stop"
                # Close-based exits.
                if fill_price is None and self.exit_on_opposite_signal \
                        and self.strategy.entry_signal(bars, i):
                    fill_price, reason = bar.close, "opposite_signal"
                if fill_price is None and self.max_bars_in_trade is not None \
                        and i - position["entry_index"] >= self.max_bars_in_trade:
                    fill_price, reason = bar.close, "max_bars"
                if fill_price is None and self.strategy.exit_signal(bars, i):
                    fill_price, reason = bar.close, "signal_rule"
                if fill_price is not None:
                    self._close_trade(position, i, bar, fill_price, reason,
                                      trades)
                    position = None

            # -- 2. ratchet the trailing-stop watermark ------------------
            if position is not None and self.trailing_stop_pct is not None:
                if watermark is None or bar.close > watermark:
                    watermark = bar.close
                position["trail_price"] = watermark * (
                    1.0 - self.trailing_stop_pct / 100.0)

            # -- 3. entry (only when flat) --------------------------------
            if position is None and self.strategy.entry_signal(bars, i):
                # Entry fills at the close, rounded to the cent, with
                # slippage added on top of the raw close.
                fill = round(bar.close * (1.0 + self.slippage_pct / 100.0), 2)
                shares = self.strategy.shares_for(fill, self.cash)
                if shares > 0:
                    fee = shares * self.commission
                    self.cash -= shares * fill + fee
                    position = {
                        "entry_index": i,
                        "entry_date": bar.date,
                        "entry_price": fill,
                        "shares": shares,
                        "stop_price": (fill * (1 - self.stop_loss_pct / 100)
                                       if self.stop_loss_pct is not None
                                       else None),
                        "target_price": (fill * (1 + self.take_profit_pct / 100)
                                         if self.take_profit_pct is not None
                                         else None),
                        "trail_price": None,
                        "entry_fee": fee,
                    }
                    watermark = bar.close

            # -- 4. equity at the close -----------------------------------
            if position is None:
                equity = self.cash
            else:
                equity = self.cash + position["shares"] * bar.close
            equity_curve.append((bar.date, equity))

        # Force-close anything still open on the final bar so metrics and
        # the trade log reflect fully realised cash.
        if position is not None:
            self._close_trade(position, len(bars) - 1, bars[-1],
                               bars[-1].close, "end_of_data", trades)
            equity_curve[-1] = (bars[-1].date, self.cash)

        return BacktestResult(bars, trades, equity_curve, self.config)