#!/usr/bin/env python3
"""Live online-test probe for the dependence estimators.

Connects to the running datastreamer server over its WebSocket (TCP), reads the
estimates and the true Spearman rho that the server broadcasts in each tick, and
prints rolling accuracy metrics per estimator in real time:

  err       instantaneous (estimate - true)
  rmse(W)   root-mean-square error over the last W ticks
  bias(W)   mean signed error over the last W ticks
  lag       how many ticks the estimator trails the moving truth
            (argmax cross-correlation; only meaningful when truth varies)

This tests the whole online path -- the server-side monitor, serialization, and
real-time timing -- not just the estimator math.

Prereq: a WebSocket client library, and the server running. See the header of
`what this needs to build` in the repo notes, or:

    .venv/bin/pip install websocket-client
    make server && ./server          # in another terminal
    .venv/bin/python experiments/live_probe.py

Usage:
    python experiments/live_probe.py [--url ws://localhost:9002]
                                         [--window 200] [--interval 20]
                                         [--maxlag 60] [--csv out.csv]
"""
import argparse
import json
import math
import signal
import sys
from collections import deque

try:
    import websocket  # from the 'websocket-client' package
except ImportError:
    sys.exit("missing dependency: pip install websocket-client")


def corr(a, b):
    """Pearson correlation of two equal-length sequences (0 if degenerate)."""
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b))
    da = math.sqrt(sum((ai - ma) ** 2 for ai in a))
    db = math.sqrt(sum((bi - mb) ** 2 for bi in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def estimate_lag(truth, est, maxlag):
    """Ticks the estimator trails the truth: argmax_k corr(truth[t], est[t+k]).

    Returns (lag, correlation), or (None, None) if the truth barely moves in the
    window (lag is undefined for a constant target).
    """
    n = len(truth)
    if n < 20:
        return None, None
    mt = sum(truth) / n
    if math.sqrt(sum((t - mt) ** 2 for t in truth) / n) < 1e-3:
        return None, None  # truth ~ constant
    best_k, best_c = 0, -2.0
    for k in range(0, maxlag + 1):
        if n - k < 10:
            break
        c = corr(truth[: n - k], list(est)[k:])
        if c > best_c:
            best_c, best_k = c, k
    return best_k, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:9002")
    ap.add_argument("--window", type=int, default=200, help="rolling metric window (ticks)")
    ap.add_argument("--interval", type=int, default=20, help="reprint every N ticks")
    ap.add_argument("--maxlag", type=int, default=60, help="max lag searched (ticks)")
    ap.add_argument("--csv", default=None, help="optional path to log raw stream")
    args = ap.parse_args()

    truth = deque(maxlen=args.window)
    ests = {}          # name -> deque of estimates (aligned with truth)
    n_ticks = 0

    csv_file = open(args.csv, "w") if args.csv else None
    header_written = False

    def summary(_sig=None, _frame=None):
        print("\n--- final (window={}, ticks={}) ---".format(args.window, n_ticks))
        if truth:
            _print_table()
        if csv_file:
            csv_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, summary)

    def _print_table():
        cur_true = truth[-1]
        print("\ntick={}  true_rho_s={:+.3f}  (window={})".format(n_ticks, cur_true, len(truth)))
        print("  {:<16}{:>8}{:>8}{:>9}{:>9}{:>6}".format(
            "estimator", "est", "err", "rmse", "bias", "lag"))
        for name, dq in ests.items():
            if not dq:
                continue
            cur = dq[-1]
            errs = [e - t for e, t in zip(dq, truth)]
            rmse = math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else 0.0
            bias = sum(errs) / len(errs) if errs else 0.0
            lag, _ = estimate_lag(list(truth), dq, args.maxlag)
            lag_s = "-" if lag is None else str(lag)
            print("  {:<16}{:>+8.3f}{:>+8.3f}{:>9.4f}{:>+9.4f}{:>6}".format(
                name, cur, cur - cur_true, rmse, bias, lag_s))

    print("connecting to {} ...".format(args.url))
    try:
        ws = websocket.create_connection(args.url)
    except Exception as exc:  # noqa: BLE001
        sys.exit("could not connect ({}). Is the server running? `make server && ./server`".format(exc))
    print("connected; waiting for ticks (Ctrl-C for final summary)\n")

    while True:
        try:
            raw = ws.recv()
        except Exception as exc:  # noqa: BLE001
            print("connection closed ({})".format(exc))
            summary()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "true_rho_s" not in msg or "dependence" not in msg:
            continue  # a tick without ground truth / estimates

        n_ticks += 1
        truth.append(float(msg["true_rho_s"]))
        for name, val in msg["dependence"].items():
            ests.setdefault(name, deque(maxlen=args.window)).append(float(val))

        if csv_file:
            names = sorted(ests)
            if not header_written:
                csv_file.write("tick,true_rho_s," + ",".join(names) + "\n")
                header_written = True
            row = [str(n_ticks), "{:.6f}".format(truth[-1])]
            row += ["{:.6f}".format(ests[nm][-1]) for nm in names]
            csv_file.write(",".join(row) + "\n")

        if n_ticks % args.interval == 0:
            _print_table()


if __name__ == "__main__":
    main()
