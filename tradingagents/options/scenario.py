"""Rank long calls or puts for a view the USER states: a direction, a target price and a date.

The model is not asked which way the stock goes (it has no measured edge at that; see
the stock-view backtest). The user supplies the view and code answers the narrower
question: given "MU to 1200 by 31 Dec", which contract turns that into the most money
per dollar risked, after time decay, a change in implied volatility and the spread?

How a contract is valued at the exit date (the target date):

* Black-Scholes repricing with the time then left to expiry, at the contract's current
  implied volatility plus ``iv_shift`` (a volatility crush is a negative shift). This is
  a model estimate, not a quote, and the output says so.
* Entry at the ask. Exit at the model value less half of today's spread, so a wide
  market costs twice: going in and coming out.
* Two distributions for the stock at the exit date, both lognormal with the contract's OWN
  implied volatility over that horizon: "market" (median at the forward price, so the
  option is worth about what it costs) and "your view" (median at your target). The market
  column is the cost of the trade before any opinion and should sit near minus the spread;
  a value far from that means the repricing disagrees with the quote.
* Contracts that expire within ``MIN_DAYS_AFTER_EXIT`` of the exit date are refused: the
  last days of an option's life are where liquidity and gamma are worst.

Usage::

    python -m tradingagents.options.scenario MU --target 1200 --by 2026-12-31
    python -m tradingagents.options.scenario IREN --put --target 30 --by 2026-11-20
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date

from tradingagents.dataflows.vendors.options import Quote, usable

RISK_FREE = 0.04            # flat; a move of a point changes a 3-month call by well under 1%
MIN_DAYS_AFTER_EXIT = 14
MAX_DAYS_AFTER_EXIT = 400   # past a year beyond the exit is a stock-replacement trade, not this question
MAX_SPREAD_PCT = 15.0
MIN_OPEN_INTEREST = 50
STRIKE_BAND = 0.35          # only strikes within +-35% of spot
STAKE = 0.10                # share of an account one contract is sized at for the growth score: large enough that a
                            # total loss counts, so a cheap far-out strike cannot win on its upside tail alone
QUAD_POINTS = 161           # standard-normal quadrature grid over [-4, 4]


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(right: str, spot: float, strike: float, years: float, vol: float, rate: float = RISK_FREE) -> float:
    """Black-Scholes value of a European option (no dividends). Intrinsic value at or past expiry."""
    intrinsic = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
    if years <= 0 or vol <= 0:
        return intrinsic
    d1 = (math.log(spot / strike) + (rate + vol * vol / 2) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    disc = math.exp(-rate * years)
    if right == "C":
        return spot * _cdf(d1) - strike * disc * _cdf(d2)
    return strike * disc * _cdf(-d2) - spot * _cdf(-d1)


@dataclass
class Row:
    symbol: str
    strike: float
    expiry: date
    ask: float
    spread_pct: float
    iv: float
    delta: float | None
    open_interest: float | None
    breakeven: float              # stock price at the exit date where the trade returns its cost
    roi_target: float             # return if the stock is exactly at the target on the exit date
    roi_half: float               # ... half way from spot to target
    roi_flat: float               # ... unchanged
    roi_down: float               # ... 10% below spot
    ev_market: float              # expected return with no opinion: the cost of the trade
    ev_view: float                # expected return if your target is the median outcome
    p_profit_market: float
    p_profit_view: float
    growth: float                 # expected log growth of the account at STAKE, under your view


def _exit_value(q: Quote, stock: float, exit_years: float, expiry_years: float, iv_shift: float) -> float:
    """What the contract is worth on the exit date, sold at today's half-spread below the model value."""
    left = max(expiry_years - exit_years, 0.0)
    value = bs_price(q.right, stock, q.strike, left, max(q.iv + iv_shift, 0.05))
    return value * (1.0 - q.spread_pct / 200.0)


def _atm_iv(quotes: list[Quote], spot: float, expiry: date) -> float | None:
    side = [q for q in quotes if q.expiry == expiry and usable(q)]
    if not side:
        return None
    near = sorted(side, key=lambda q: abs(q.strike - spot))[:4]
    return sum(q.iv for q in near) / len(near)


def market_vol(quotes: list[Quote], spot: float, exit_date: date) -> float | None:
    """The market's volatility for the horizon: at-the-money implied volatility of the first expiry after the exit."""
    later = sorted({q.expiry for q in quotes if q.expiry > exit_date})
    for expiry in later:
        iv = _atm_iv(quotes, spot, expiry)
        if iv:
            return iv
    return None


def rank_contracts(quotes: list[Quote], spot: float, right: str, target: float, exit_date: date,
                   today: date, iv_shift: float = 0.0, rate: float = RISK_FREE) -> tuple[list[Row], dict]:
    """Candidates that survive the liquidity and expiry filters, best expected growth first.

    Returns ``(rows, context)``; ``context`` carries the volatility and horizon the
    distributions used and how many contracts each filter removed.
    """
    horizon = (exit_date - today).days / 365.0
    sigma = market_vol(quotes, spot, exit_date)
    if sigma is None or horizon <= 0:
        return [], {"error": "no usable quotes expiring after the exit date, or the exit date is not in the future"}
    s_mkt = sigma * math.sqrt(horizon)
    zs = [-4 + 8 * i / (QUAD_POINTS - 1) for i in range(QUAD_POINTS)]
    w = [math.exp(-z * z / 2) for z in zs]
    total = sum(w)
    w = [x / total for x in w]

    dropped = {"wrong side": 0, "expires too soon": 0, "expires too late": 0, "not usable": 0, "wide spread": 0, "thin open interest": 0,
               "far from spot": 0}
    rows: list[Row] = []
    for q in quotes:
        if q.right != right:
            dropped["wrong side"] += 1
            continue
        if (q.expiry - exit_date).days < MIN_DAYS_AFTER_EXIT:
            dropped["expires too soon"] += 1
            continue
        if (q.expiry - exit_date).days > MAX_DAYS_AFTER_EXIT:
            dropped["expires too late"] += 1
            continue
        if not usable(q):
            dropped["not usable"] += 1
            continue
        if q.spread_pct > MAX_SPREAD_PCT:
            dropped["wide spread"] += 1
            continue
        if (q.open_interest or 0) < MIN_OPEN_INTEREST:
            dropped["thin open interest"] += 1
            continue
        if abs(q.strike / spot - 1) > STRIKE_BAND:
            dropped["far from spot"] += 1
            continue
        expiry_years = (q.expiry - today).days / 365.0
        # The contract's own volatility drives the spread of outcomes, so that with no opinion its
        # expected return is about minus the cost of getting in and out.
        s_h = q.iv * math.sqrt(horizon)
        grids = {"market": [spot * math.exp((rate - q.iv * q.iv / 2) * horizon + s_h * z) for z in zs],
                 "view": [target * math.exp(s_h * z) for z in zs]}

        def roi(stock: float) -> float:
            return _exit_value(q, stock, horizon, expiry_years, iv_shift) / q.ask - 1.0

        stats = {}
        for name, grid in grids.items():
            rois = [roi(s) for s in grid]
            stats[name] = (sum(p * r for p, r in zip(w, rois)),
                           sum(p for p, r in zip(w, rois) if r > 0),
                           sum(p * math.log(max(1.0 + STAKE * r, 1e-9)) for p, r in zip(w, rois)))
        # Breakeven on the exit date: the lowest (call) / highest (put) stock price whose ROI is >= 0.
        lo, hi = (spot * 0.2, spot * 3.0)
        for _ in range(60):
            mid = (lo + hi) / 2
            profitable = roi(mid) >= 0
            if q.right == "C":
                lo, hi = (lo, mid) if profitable else (mid, hi)
            else:
                lo, hi = (mid, hi) if profitable else (lo, mid)
        rows.append(Row(q.symbol, q.strike, q.expiry, q.ask, q.spread_pct, q.iv, q.delta, q.open_interest,
                        (lo + hi) / 2, roi(target), roi((spot + target) / 2), roi(spot), roi(spot * 0.9),
                        stats["market"][0], stats["view"][0], stats["market"][1], stats["view"][1],
                        stats["view"][2]))
    rows.sort(key=lambda r: r.growth, reverse=True)
    return rows, {"sigma": sigma, "horizon_days": (exit_date - today).days, "s_h": s_mkt,
                  "dropped": dropped, "iv_shift": iv_shift}


def render(rows: list[Row], ctx: dict, symbol: str, right: str, spot: float, target: float,
           exit_date: date, per_expiry: int = 3) -> str:
    if not rows:
        return f"{symbol}: no contract passes the filters ({ctx.get('error') or ctx.get('dropped')})."
    kind = "call" if right == "C" else "put"
    move = target / spot - 1
    out = [
        f"{symbol} long {kind}s for your view: {target:,.0f} by {exit_date} (spot {spot:,.2f}, {move:+.1%}, "
        f"{ctx['horizon_days']} days).",
        f"Market volatility for the horizon {ctx['sigma']:.0%} -> a normal move to the exit date is about "
        f"+-{ctx['s_h']:.0%}, so your target is {abs(move) / ctx['s_h']:.1f} of one standard move away.",
        f"Model values at the exit date, entry at the ask, exit at value less half today's spread, "
        f"IV shift {ctx['iv_shift']:+.0%}. Estimates, not quotes. Best {per_expiry} per expiry by expected growth:",
        "",
        f"{'contract':22}{'ask':>8}{'sprd':>6}{'IV':>5}{'delta':>6}{'break-even':>11}"
        f"{'at target':>10}{'half way':>9}{'flat':>7}{'-10%':>7}{'EV view':>9}{'EV mkt':>8}{'P(win)':>8}",
    ]
    for expiry in sorted({r.expiry for r in rows}):
        for r in [r for r in rows if r.expiry == expiry][:per_expiry]:
            out.append(f"{r.symbol:22}{r.ask:8.2f}{r.spread_pct:5.1f}%{r.iv:5.0%}{(r.delta or 0):6.2f}"
                       f"{r.breakeven:11,.0f}{r.roi_target:+10.0%}{r.roi_half:+9.0%}{r.roi_flat:+7.0%}"
                       f"{r.roi_down:+7.0%}{r.ev_view:+9.0%}{r.ev_market:+8.0%}{r.p_profit_view:8.0%}")
    out += ["", "at target / half way / flat / -10% = return if the stock is exactly there on the exit date.",
            "EV view = expected return if your target is the median outcome; EV mkt = expected return with no "
            "opinion (the cost of the trade).", "P(win) = chance of a positive return under your view.",
            "Filtered out: " + ", ".join(f"{k} {v}" for k, v in ctx["dropped"].items() if v)]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from tradingagents.dataflows.vendors.options import fetch_chain, ny_today

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("symbol")
    ap.add_argument("--target", type=float, required=True, help="the price you expect")
    ap.add_argument("--by", required=True, help="the date you expect it by, YYYY-MM-DD")
    ap.add_argument("--put", action="store_true", help="bearish: rank puts (default is calls)")
    ap.add_argument("--iv-shift", type=float, default=0.0, help="change in implied volatility by the exit, e.g. -0.15")
    ap.add_argument("--per-expiry", type=int, default=3, help="contracts shown per expiry")
    args = ap.parse_args(argv)

    meta, quotes = fetch_chain(args.symbol.upper())
    spot = meta["spot"]
    if not spot:
        print(f"{args.symbol}: the source gave no underlying price")
        return 1
    right = "P" if args.put else "C"
    if (right == "C") != (args.target > spot):
        print(f"note: a {'put' if right == 'P' else 'call'} profits from a fall/rise, but the target "
              f"{args.target:,.2f} is {'above' if args.target > spot else 'below'} spot {spot:,.2f}")
    exit_date = date.fromisoformat(args.by)
    rows, ctx = rank_contracts(quotes, spot, right, args.target, exit_date, date.fromisoformat(ny_today()),
                               iv_shift=args.iv_shift)
    print(render(rows, ctx, args.symbol.upper(), right, spot, args.target, exit_date, args.per_expiry))
    try:
        from tradingagents.dataflows.vendors.options import earnings_events

        upcoming = [(d.date(), when) for d, when in earnings_events(args.symbol.upper())
                    if date.fromisoformat(ny_today()) <= d.date() <= exit_date]
        if upcoming:
            print("\nEarnings before your exit date: " + ", ".join(f"{d} ({when})" for d, when in upcoming)
                  + ". Implied volatility usually falls after the report; try --iv-shift -0.10.")
    except Exception as exc:  # noqa: BLE001 — a calendar failure must not hide the ranking
        print(f"\n(earnings dates unavailable: {exc})")
    print(f"\nData: {meta['source']} as of {meta['as_of']} (delayed, not executable prices).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
