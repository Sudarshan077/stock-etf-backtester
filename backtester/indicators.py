"""Technical indicators, hand-written with pure Python stdlib.

Every function takes a list of floats (the full series up to the current
bar) and returns a list of the same length, where index i holds the
indicator value at bar i or None while the indicator is still "warming
up" (fewer bars than its period).  This "same-length list with None
warm-up" convention keeps the engine dead simple: it just reads
ind[i] and treats None as "no value yet, rule cannot fire".

No third-party libraries are used anywhere.
"""


# ---------------------------------------------------------------------------
# Simple Moving Average
# ---------------------------------------------------------------------------

def sma(values, period):
    """Simple Moving Average.

    sma[i] = mean(values[i-period+1 .. i]) once at least `period` values
    exist, else None.
    """
    if period < 1:
        raise ValueError("SMA period must be >= 1")
    out = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


# ---------------------------------------------------------------------------
# Exponential Moving Average
# ---------------------------------------------------------------------------

def ema(values, period):
    """Exponential Moving Average, seeded with the SMA of the first window.

    alpha = 2 / (period + 1)
    ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]

    The first value is emitted at index period-1 (the SMA of the first
    `period` values), which is the standard seeding convention.
    """
    if period < 1:
        raise ValueError("EMA period must be >= 1")
    out = [None] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# Relative Strength Index
# ---------------------------------------------------------------------------

def rsi(closes, period=14):
    """RSI using the classic Wilder smoothing (modified moving average).

    - First value at index `period`: 100 - 100/(1+RS) where RS is the
      simple average gain / simple average loss over the first window.
    - Subsequent values use Wilder's smoothing: avgGain and avgLoss are
      updated as avg = (prev_avg*(period-1) + current_change) / period.
    """
    if period < 2:
        raise ValueError("RSI period must be >= 2")
    out = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain, avg_loss):
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# Average True Range
# ---------------------------------------------------------------------------

def atr(highs, lows, closes, period=14):
    """Average True Range with Wilder smoothing.

    True Range for bar i:  max(high-low, |high-prev_close|, |low-prev_close|)
    First ATR at index `period` is the simple average of TRs 1..period;
    afterwards ATR = (prev_atr*(period-1) + TR_i) / period.
    """
    n = len(closes)
    if period < 1:
        raise ValueError("ATR period must be >= 1")
    out = [None] * n
    if n <= period:
        return out
    trs = _true_ranges(highs, lows, closes)
    # TR is defined from index 1 onwards (needs previous close).
    first = sum(trs[1:period + 1]) / period
    out[period] = first
    prev = first
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + trs[i]) / period
        out[i] = prev
    return out


def _true_ranges(highs, lows, closes):
    """True Range per bar; index 0 is None (no previous close)."""
    n = len(closes)
    trs = [None] * n
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        trs[i] = max(hl, hc, lc)
    return trs


# ---------------------------------------------------------------------------
# Rolling extremes (Donchian-style channels)
# ---------------------------------------------------------------------------

def highest_high(highs, period):
    """highest_high[i] = max(values[i-period .. i-1]) -- the previous
    `period` bars EXCLUDING the current bar.

    Excluding the current bar is what makes breakout entries possible:
    "close above the N-day high" can never trigger if the current bar's
    own high is inside the window (close <= high always).  None is
    emitted until at least `period` PRIOR bars exist, i.e. first value
    at index `period`.  Works on any numeric series (highs, closes...).
    Implemented with a monotonic deque: O(n) total.
    """
    return _rolling_extreme(highs, period, max)


def lowest_low(lows, period):
    """lowest_low[i] = min(values[i-period .. i-1]) -- previous `period`
    bars excluding the current bar (see highest_high for why)."""
    return _rolling_extreme(lows, period, min)


def _rolling_extreme(values, period, better):
    """Shared monotonic-deque rolling window (previous `period` bars,
    current bar excluded).  `better` is max or min."""
    if period < 1:
        raise ValueError("period must be >= 1")
    from collections import deque
    out = [None] * len(values)
    dq = deque()  # indices within the window, values monotone
    for i, v in enumerate(values):
        # Evict indices that fell out of the window [i-period, i-1].
        if dq and dq[0] < i - period:
            dq.popleft()
        # Emit BEFORE pushing the current bar: the window excludes i.
        if i >= period and dq:
            out[i] = values[dq[0]]
        # Push v, discarding dominated candidates.
        while dq and better(values[dq[-1]], v) == v:
            dq.pop()
        dq.append(i)
    return out