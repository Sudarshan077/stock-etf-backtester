#!/usr/bin/env python
"""CLI entry point: run a backtest from a JSON config file.

Usage:
    python run.py --config config.json
    python run.py --config config.json --data data/sample_spy.csv
    python run.py --config config.json --output output_alt
    python run.py --config config.json --risk-free 0.04

Everything is pure Python stdlib; see README.md for the full config
reference.
"""

import argparse
import json
import os
import sys

# Allow running either as `python run.py` (from the project root) or
# `python tests/...` -- add this file's directory to sys.path so the
# `backtester` package imports cleanly either way.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import engine, metrics as metrics_mod, report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pure-Python stock/ETF backtesting engine.")
    parser.add_argument("--config", required=True,
                        help="Path to the strategy config JSON file.")
    parser.add_argument("--data", default=None,
                        help="Override the data CSV from the config.")
    parser.add_argument("--output", default=None,
                        help="Override the output directory from the config.")
    parser.add_argument("--risk-free", type=float, default=0.0,
                        help="Annual risk-free rate for the Sharpe ratio "
                             "(default 0.0).")
    args = parser.parse_args(argv)

    # ---- load config ----------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ---- locate data ----------------------------------------------------
    data_file = args.data or config.get("data_file")
    if not data_file:
        parser.error("No data file: use --data or set 'data_file' in the "
                     "config.")
    # Config paths are resolved relative to the config file so the whole
    # project is portable; --data on the CLI is taken as-is.
    if args.data is None and not os.path.isabs(data_file):
        data_file = os.path.join(os.path.dirname(
            os.path.abspath(args.config)), data_file)
    if not os.path.isfile(data_file):
        print("ERROR: data file not found: %s" % data_file, file=sys.stderr)
        return 2

    # ---- run the backtest --------------------------------------------------
    print("Loading data    : %s" % data_file)
    bars = engine.load_bars(data_file)
    print("Bars loaded     : %d (%s .. %s)" % (len(bars), bars[0].date,
                                                bars[-1].date))

    eng = engine.Engine(config)
    result = eng.run(bars)

    # ---- metrics + report ---------------------------------------------------
    m = metrics_mod.compute_metrics(result, risk_free_annual=args.risk_free)
    out_dir = args.output or config.get("output_dir", "output")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(
            os.path.abspath(args.config)), out_dir)
    paths = report.write_all_reports(result, m, out_dir,
                                     data_file=os.path.basename(data_file))

    print()
    print(metrics_mod.format_metrics(m))
    print()
    print("Reports written:")
    for label, p in paths.items():
        print("  %-12s %s" % (label, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())