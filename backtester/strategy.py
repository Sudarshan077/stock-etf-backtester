"""Config-driven strategy: entry rules, exit rules, position sizing.

The strategy layer turns the JSON config into plain predicates the
engine can call on each bar.  Three rule "shortcuts" are expanded here
into a generic comparison form so the engine only understands ONE
primitive:

    breakout       close > N-day highest high (strictly above)
    crossover      fast series crosses above/below slow series
    rsi_threshold  RSI_N compared against a constant

Any number of rules can be combined: entries require ALL to be true
(logical AND).  The generic primitive is also available directly:

    {"type": "indicator_compare",
     "left":  {"kind": "close" | "indicator" | "value", ...},
     "op":    ">" | "<" | ">=" | "<=",
     "right": {"kind": "close" | "indicator" | "value", ...}}

Indicator names: "sma_N", "ema_N", "rsi_N", "atr_N", "hh_N" (N-day
highest high), "ll_N" (N-day lowest low).  Series are computed once and
cached, so two rules referencing sma_50 cost one computation.
"""

from . import indicators as _ind


_OPS = {">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b}


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


class Strategy(object):
    """Validates the strategy config and evaluates it bar by bar.

    Public API used by the engine:
      entry_signal(bars, i) -> bool   all entry rules true at bar i
      exit_signal(bars, i)  -> bool   any exit-signal rule true at bar i
      shares_for(price, cash) -> int  position sizing
    """

    def __init__(self, config):
        self.config = config

        sizing = config.get("position_sizing", {})
        self.sizing_mode = sizing.get("mode", "percent_of_equity")
        _require(self.sizing_mode in ("percent_of_equity", "fixed_capital"),
                 "position_sizing.mode must be 'percent_of_equity' "
                 "or 'fixed_capital'")
        self.sizing_value = float(sizing.get("value", 95.0))

        # Compiled entry rules (all must hold) and exit-signal rules.
        self.entry_rules = [self._compile(r, "entry_rules")
                            for r in config.get("entry_rules", [])]
        _require(self.entry_rules,
                 "entry_rules must contain at least one rule")

        exits = config.get("exit_rules", {})
        self.exit_signal_rules = [self._compile(r, "exit_rules.signal_rules")
                                  for r in exits.get("signal_rules", [])]

        # Indicator cache: name -> list (computed lazily, once).
        self._cache = {}

    # ------------------------------------------------------------------
    # Rule compilation (config -> generic comparison)
    # ------------------------------------------------------------------

    def _compile(self, rule, where):
        """Validate one rule and normalize it to 'indicator_compare'."""
        _require(isinstance(rule, dict), "%s: each rule must be a JSON "
                 "object, got %r" % (where, rule))
        _require("type" in rule, "%s: rule missing 'type': %r" % (where, rule))
        t = rule["type"]

        if t == "breakout":
            # close strictly above the N-day highest high.  Two flavors:
            #   basis "high" (default): N-day highest HIGH  (Donchian)
            #   basis "close":           N-day highest CLOSE
            basis = rule.get("basis", "high")
            _require(basis in ("high", "close"),
                     "breakout.basis must be 'high' or 'close'")
            # 'high'  -> N-day highest HIGH  (Donchian channel)
            # 'close' -> N-day highest CLOSE (uses the same rolling-max
            #            code, run over closes instead of highs)
            name = ("hh_%d" if basis == "high" else "hc_%d") % int(rule["days"])
            return {"type": "indicator_compare",
                    "left": {"kind": "close"}, "op": ">",
                    "right": {"kind": "indicator", "name": name}}

        if t == "crossover":
            # fast crosses above ("above") or below ("below") slow: the
            # relation must hold now and NOT have held on the prior bar.
            direction = rule.get("direction", "above")
            _require(direction in ("above", "below"),
                     "crossover.direction must be 'above' or 'below'")
            fast = {"kind": "indicator", "name": rule["fast"]}
            slow = {"kind": "indicator", "name": rule["slow"]}
            if direction == "above":
                left, op, right = fast, ">", slow
            else:
                left, op, right = fast, "<", slow
            return {"type": "indicator_compare",
                    "left": left, "op": op, "right": right,
                    "cross": True}

        if t == "rsi_threshold":
            # {"rsi": 30} alone means compare RSI period 14.
            name = rule.get("indicator", "rsi_%d" % int(rule.get("period", 14)))
            return {"type": "indicator_compare",
                    "left": {"kind": "indicator", "name": name},
                    "op": rule.get("op", "<"),
                    "right": {"kind": "value", "value": rule["threshold"]}}

        if t == "indicator_compare":
            out = {"type": "indicator_compare",
                   "left": self._operand(rule["left"], where),
                   "op": rule["op"],
                   "right": self._operand(rule["right"], where)}
            _require(out["op"] in _OPS,
                     "%s: op must be one of > < >= <=" % where)
            return out

        raise ValueError("%s: unknown rule type %r" % (where, t))

    @staticmethod
    def _operand(spec, where):
        """Validate one side of a comparison."""
        _require(isinstance(spec, dict) and "kind" in spec,
                 "%s: operand must be an object with a 'kind': %r"
                 % (where, spec))
        kind = spec["kind"]
        if kind in ("close", "open", "high", "low"):
            return {"kind": kind}
        if kind == "value":
            _require("value" in spec,
                     "%s: kind 'value' needs a 'value'" % where)
            return {"kind": "value", "value": float(spec["value"])}
        if kind == "indicator":
            _require("name" in spec,
                     "%s: kind 'indicator' needs a 'name'" % where)
            name = spec["name"]
            valid = name.startswith(("sma_", "ema_", "rsi_", "atr_",
                                     "hh_", "ll_", "hc_"))
            _require(valid, "%s: unknown indicator %r (use sma_N, ema_N, "
                     "rsi_N, atr_N, hh_N, ll_N, hc_N)" % (where, name))
            int(name.split("_", 1)[1])  # raises if N is not an integer
            return {"kind": "indicator", "name": name}
        raise ValueError("%s: unknown operand kind %r" % (where, kind))

    # ------------------------------------------------------------------
    # Indicator series (cached)
    # ------------------------------------------------------------------

    def series(self, name, bars):
        """Compute (once) and return the named indicator over `bars`."""
        if name in self._cache:
            return self._cache[name]
        closes = [b.close for b in bars]
        kind, _, nstr = name.partition("_")
        n = int(nstr)
        if kind == "sma":
            s = _ind.sma(closes, n)
        elif kind == "ema":
            s = _ind.ema(closes, n)
        elif kind == "rsi":
            s = _ind.rsi(closes, n)
        elif kind == "atr":
            s = _ind.atr([b.high for b in bars], [b.low for b in bars],
                         closes, n)
        elif kind == "hh":
            s = _ind.highest_high([b.high for b in bars], n)
        elif kind == "ll":
            s = _ind.lowest_low([b.low for b in bars], n)
        elif kind == "hc":
            s = _ind.highest_high(closes, n)
        else:
            raise ValueError("Unknown indicator %r" % name)
        self._cache[name] = s
        return s

    def _operand_value(self, spec, bars, i):
        kind = spec["kind"]
        if kind == "value":
            return spec["value"]
        if kind in ("close", "open", "high", "low"):
            return getattr(bars[i], kind)
        return self.series(spec["name"], bars)[i]

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def _holds(self, rule, bars, i):
        left = self._operand_value(rule["left"], bars, i)
        right = self._operand_value(rule["right"], bars, i)
        if left is None or right is None:
            return False  # indicator still warming up at bar i
        op = _OPS[rule["op"]]
        if rule.get("cross"):
            # True CROSS event: condition holds now but did NOT hold on
            # the previous bar (warm-up or opposite side both count as
            # "did not hold").
            if i == 0:
                return False
            prev_l = self._operand_value(rule["left"], bars, i - 1)
            prev_r = self._operand_value(rule["right"], bars, i - 1)
            if prev_l is None or prev_r is None:
                return False
            return op(left, right) and not op(prev_l, prev_r)
        return op(left, right)

    def entry_signal(self, bars, i):
        """True when every entry rule holds at bar i."""
        return all(self._holds(r, bars, i) for r in self.entry_rules)

    def exit_signal(self, bars, i):
        """True when any exit-signal rule holds at bar i."""
        return any(self._holds(r, bars, i) for r in self.exit_signal_rules)

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def shares_for(self, price, cash):
        """Whole shares to buy at `price` given available `cash`.

        percent_of_equity : value% of current cash (default 95%)
        fixed_capital     : fixed dollar budget, capped by cash
        """
        if self.sizing_mode == "percent_of_equity":
            budget = cash * (self.sizing_value / 100.0)
        else:
            budget = min(self.sizing_value, cash)
        if price <= 0 or budget <= 0:
            return 0
        return int(budget / price)