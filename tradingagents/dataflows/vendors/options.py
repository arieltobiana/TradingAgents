"""Option-chain facts for the fact sheet: prices, greeks, dates and events.

The question this answers is "which call (or put) should I buy, if any?", so
everything a buyer weighs is computed here in code and handed to the agents as
facts: what each candidate costs, how wide its market is, what its greeks are,
where it breaks even, how that compares with the move the market is pricing,
whether its life spans an earnings report, and how the stock actually moved
on past reports.

Sources, in order:

* **Cboe delayed quotes** (``cdn-api.cboe.com``): the whole chain in one
  request, with bid/ask, open interest, IV and greeks, about 15 minutes late.
  It is a public but undocumented endpoint and can change without notice.
* **Alpaca options snapshots** (``feed=indicative``): used to cross-check the
  candidates' IV and delta, and as the chain when Cboe fails. Needs
  ``ALPACA_API_KEY_ID`` / ``ALPACA_API_SECRET_KEY``.

Greeks are received from these sources, never computed here. A quote with no
bid, no ask, a crossed market or an IV the source could not solve is dropped,
not repaired. Option data is live-only, so a run dated in the past gets a gap,
not today's chain.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv, yf_retry

logger = logging.getLogger(__name__)

CBOE_URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
ALPACA_URL = "https://data.alpaca.markets/v1beta1/options/snapshots"
HTTP_TIMEOUT = 20

# A candidate must have a two-sided market no wider than this share of its mid.
MAX_SPREAD_PCT = 15.0
# Expiries sampled: the nearest one at or beyond each of these days-to-expiry.
EXPIRY_TARGETS_DAYS = (7, 21, 45, 75, 120)
# Deltas sampled per expiry (absolute value for puts), and how far a contract's
# delta may sit from its target and still stand for it.
DELTA_TARGETS = (0.70, 0.50, 0.35, 0.20)
DELTA_TOLERANCE = 0.08
# Cross-check thresholds between sources for the same contract.
MAX_IV_DISAGREEMENT = 0.05       # 5 vol points
MAX_DELTA_DISAGREEMENT = 0.05
# Past earnings reactions averaged for the comparison with the priced move.
EARNINGS_HISTORY = 8
# IV rank needs this much saved history before it is reported at all.
IV_RANK_MIN_DAYS = 60

_OCC = re.compile(r"^(?P<root>[A-Z.]{1,6})(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})(?P<right>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class Quote:
    symbol: str
    right: str  # "C" or "P"
    strike: float
    expiry: date
    bid: float
    ask: float
    iv: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    open_interest: float | None
    volume: float | None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid * 100


def parse_occ(symbol: str) -> tuple[str, date, str, float] | None:
    m = _OCC.match(symbol.replace(" ", ""))
    if not m:
        return None
    expiry = date(2000 + int(m["y"]), int(m["m"]), int(m["d"]))
    return m["root"], expiry, m["right"], int(m["strike"]) / 1000


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def usable(q: Quote) -> bool:
    """A quote a buyer could act on: two-sided, not crossed, with a solved IV and delta."""
    return (q.bid > 0 and q.ask > 0 and q.ask >= q.bid and q.iv is not None and q.iv > 0.01
            and q.delta is not None)


# ------------------------------------------------------------------ sources

def fetch_cboe(symbol: str) -> tuple[dict, list[Quote]]:
    r = requests.get(CBOE_URL.format(symbol=symbol.upper()), headers={"User-Agent": "Mozilla/5.0"},
                     timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    data = payload["data"]
    quotes = []
    for o in data.get("options", []):
        occ = parse_occ(o.get("option", ""))
        if not occ:
            continue
        _, expiry, right, strike = occ
        quotes.append(Quote(o["option"], right, strike, expiry, _num(o.get("bid")) or 0.0,
                            _num(o.get("ask")) or 0.0, _num(o.get("iv")), _num(o.get("delta")),
                            _num(o.get("gamma")), _num(o.get("theta")), _num(o.get("vega")),
                            _num(o.get("open_interest")), _num(o.get("volume"))))
    meta = {"source": "Cboe delayed quotes", "as_of": payload.get("timestamp"),
            "spot": _num(data.get("current_price")), "iv30": _num(data.get("iv30"))}
    return meta, quotes


def _alpaca_headers() -> dict | None:
    key, secret = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _alpaca_quote(sym: str, snap: dict) -> Quote | None:
    occ = parse_occ(sym)
    if not occ:
        return None
    _, expiry, right, strike = occ
    q, g = snap.get("latestQuote") or {}, snap.get("greeks") or {}
    return Quote(sym, right, strike, expiry, _num(q.get("bp")) or 0.0, _num(q.get("ap")) or 0.0,
                 _num(snap.get("impliedVolatility")), _num(g.get("delta")), _num(g.get("gamma")),
                 _num(g.get("theta")), _num(g.get("vega")), None, None)


def fetch_alpaca(symbol: str, symbols: list[str] | None = None) -> list[Quote]:
    """Alpaca snapshots: named contracts, or the whole chain for ``symbol``."""
    headers = _alpaca_headers()
    if headers is None:
        raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY are not set")
    quotes, token = [], None
    if symbols:
        r = requests.get(ALPACA_URL, headers=headers, timeout=HTTP_TIMEOUT,
                         params={"symbols": ",".join(symbols), "feed": "indicative"})
        r.raise_for_status()
        snaps = r.json().get("snapshots") or {}
        return [q for s, v in snaps.items() if (q := _alpaca_quote(s, v))]
    for _ in range(20):
        params = {"feed": "indicative", "limit": 1000}
        if token:
            params["page_token"] = token
        r = requests.get(f"{ALPACA_URL}/{symbol.upper()}", headers=headers, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        quotes += [q for s, v in (body.get("snapshots") or {}).items() if (q := _alpaca_quote(s, v))]
        token = body.get("next_page_token")
        if not token:
            break
    return quotes


# ----------------------------------------------------------- earnings dates

def earnings_events(symbol: str) -> list[tuple[pd.Timestamp, str]]:
    """(report time, 'before open' | 'after close' | 'during session') newest first."""
    frame = yf_retry(lambda: yf.Ticker(normalize_symbol(symbol)).get_earnings_dates(limit=EARNINGS_HISTORY + 4))
    out = []
    for ts in (frame.index if frame is not None else []):
        ts = pd.Timestamp(ts)
        hour = ts.tz_convert("America/New_York").hour if ts.tzinfo else ts.hour
        # The vendor writes a placeholder mid-session time when the report's
        # timing is not confirmed; say so rather than invent a session.
        when = "before open" if hour < 9 else "after close" if hour >= 16 else "time not confirmed"
        out.append((ts, when))
    return out


def earnings_reactions(symbol: str, trade_date: str, events) -> list[tuple[str, float]]:
    """Close-to-close move on each past report's reaction session, newest first."""
    bars = load_ohlcv(symbol, trade_date, fill_gaps=False).copy()
    bars["Date"] = pd.to_datetime(bars["Date"]).dt.normalize()
    closes = bars.set_index("Date")["Close"].dropna()
    moves = []
    for ts, when in events:
        day = pd.Timestamp(ts.date())
        if day >= pd.Timestamp(trade_date):
            continue
        before = closes[closes.index < day] if when != "after close" else closes[closes.index <= day]
        after = closes[closes.index >= day] if when != "after close" else closes[closes.index > day]
        if before.empty or after.empty:
            continue
        moves.append((f"{day:%Y-%m-%d}", (after.iloc[0] / before.iloc[-1] - 1) * 100))
        if len(moves) == EARNINGS_HISTORY:
            break
    return moves


# ------------------------------------------------------------- IV history

def record_iv(cache_dir: str, symbol: str, day: str, iv30: float | None, spot: float | None) -> list[tuple[str, float]]:
    """Append today's IV30 (once per day) and return the saved history."""
    path = Path(cache_dir) / "options_history" / f"{normalize_symbol(symbol).upper()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open() as f:
            rows = [(r["date"], float(r["iv30"])) for r in csv.DictReader(f) if r.get("iv30")]
    if iv30 is not None and all(d != day for d, _ in rows):
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "iv30", "spot"])
            w.writerow([day, iv30, spot if spot is not None else ""])
        rows.append((day, iv30))
    return rows


# ------------------------------------------------------------ fact building

def _pick_expiries(expiries: list[date], today: date) -> list[date]:
    picked = []
    for target in EXPIRY_TARGETS_DAYS:
        e = next((e for e in sorted(expiries) if (e - today).days >= target), None)
        if e and e not in picked:
            picked.append(e)
    return picked


def _nearest_delta(quotes: list[Quote], target: float) -> Quote | None:
    best = min(quotes, key=lambda q: abs(abs(q.delta) - target), default=None)
    if best is None or abs(abs(best.delta) - target) > DELTA_TOLERANCE:
        return None
    return best


def _atm(quotes: list[Quote], spot: float, right: str) -> Quote | None:
    side = [q for q in quotes if q.right == right]
    return min(side, key=lambda q: abs(q.strike - spot), default=None)


def option_facts(sheet, symbol: str, trade_date: str, right: str, cache_dir: str | None) -> None:
    """Add the options section to ``sheet`` (a ``yahoo.facts._Sheet``)."""
    if trade_date < get_current_date():
        sheet.gaps.append("options (quotes and greeks are live-only; not available for a past date)")
        return
    today = date.fromisoformat(trade_date)
    try:
        meta, chain = fetch_cboe(symbol)
    except Exception as exc:  # noqa: BLE001 — fall back to the second source
        logger.warning("options: Cboe unavailable for %s (%s); trying Alpaca", symbol, exc)
        chain = fetch_alpaca(symbol)
        meta = {"source": "Alpaca indicative feed", "as_of": datetime.now().isoformat(timespec="minutes"),
                "spot": None, "iv30": None}
    src = f"{meta['source']}, as of {meta['as_of']}"

    close = next((f.value for f in sheet.facts if f.label == "Latest close"), None)
    spot = meta["spot"] or close
    if spot is None:
        sheet.gaps.append("options (no underlying price)")
        return
    sheet.add("Underlying price used for the options section", spot, "price", src)
    if close and meta["spot"] and abs(meta["spot"] / close - 1) > 0.02:
        sheet.add("Options source price vs latest daily close",
                  f"differ by {(meta['spot'] / close - 1) * 100:+.1f}%: the chain may be stale or the "
                  "session moved", "text", src)

    total = len(chain)
    live = [q for q in chain if q.expiry > today and usable(q)]
    sheet.add("Option contracts quoted / usable (two-sided, solved IV and delta)",
              f"{total} quoted, {len(live)} usable", "text", src)
    if not live:
        sheet.gaps.append("options (no usable quotes)")
        return

    # Volatility context.
    hv = next((f.value for f in sheet.facts if f.label.startswith("Annualized volatility")), None)
    if meta["iv30"]:
        sheet.add("30-day implied volatility (IV30)", meta["iv30"], "pct", src)
        if hv:
            sheet.add("IV30 vs 20-day realized volatility", meta["iv30"] / hv, "x", src)
    history = record_iv(os.fspath(cache_dir), symbol, trade_date, meta["iv30"], spot) if cache_dir else None
    if history is None:
        sheet.gaps.append("IV rank (no cache directory to save daily IV snapshots in)")
    elif len(history) >= IV_RANK_MIN_DAYS:
        ivs = [v for _, v in history[-252:]]  # at most a year of trading days
        rank = (ivs[-1] - min(ivs)) / (max(ivs) - min(ivs)) * 100 if max(ivs) > min(ivs) else 50.0
        sheet.add(f"IV rank over {len(ivs)} saved days", rank, "pct", "saved daily IV30 snapshots")
    else:
        sheet.gaps.append(f"IV rank ({len(history)} of {IV_RANK_MIN_DAYS} daily IV snapshots saved so far)")

    oi_c = sum(q.open_interest or 0 for q in chain if q.right == "C")
    oi_p = sum(q.open_interest or 0 for q in chain if q.right == "P")
    if oi_c:
        sheet.add("Put/call open-interest ratio (whole chain)", oi_p / oi_c, "x", src)

    # Earnings: the next report, and how the stock reacted to past ones.
    next_report = None
    try:
        events = earnings_events(symbol)
        upcoming = [e for e in events if e[0].date() >= today]
        if upcoming:
            next_report = min(upcoming, key=lambda e: e[0])
            sheet.add("Next earnings report", f"{next_report[0]:%Y-%m-%d} ({next_report[1]})", "date",
                      "vendor earnings calendar")
        reactions = earnings_reactions(symbol, trade_date, events)
        if reactions:
            moves = [abs(m) for _, m in reactions]
            sheet.add(f"Average absolute earnings-day move, last {len(moves)} reports",
                      sum(moves) / len(moves), "pct", "daily closes around past report dates")
            sheet.add(f"Largest absolute earnings-day move, last {len(moves)} reports", max(moves), "pct",
                      "daily closes around past report dates")
            sheet.add("Past earnings-day moves (newest first)",
                      ", ".join(f"{d}: {m:+.1f}%" for d, m in reactions), "text",
                      "daily closes around past report dates")
    except Exception as exc:  # noqa: BLE001
        sheet.gaps.append(f"earnings history ({type(exc).__name__})")

    def spans_earnings(expiry: date) -> str:
        if next_report is None:
            return "unknown"
        return "yes" if next_report[0].date() <= expiry else "no"

    # Per-expiry market-implied move and skew.
    by_expiry: dict[date, list[Quote]] = {}
    for q in live:
        by_expiry.setdefault(q.expiry, []).append(q)
    expiries = _pick_expiries(list(by_expiry), today)
    for e in expiries:
        qs = by_expiry[e]
        call, put = _atm(qs, spot, "C"), _atm(qs, spot, "P")
        dte = (e - today).days
        if call and put and call.strike == put.strike:
            move = (call.mid + put.mid) / spot * 100
            sheet.add(f"Expiry {e} ({dte} days, spans earnings: {spans_earnings(e)}): at-the-money straddle "
                      "implied move", move, "pct", src)
            sheet.add(f"Expiry {e}: at-the-money IV", (call.iv + put.iv) / 2 * 100, "pct", src)
        if 20 <= dte <= 60:
            c25, p25 = _nearest_delta([q for q in qs if q.right == "C"], 0.25), \
                _nearest_delta([q for q in qs if q.right == "P"], 0.25)
            if c25 and p25:
                sheet.add(f"Expiry {e}: skew, 25-delta put IV minus 25-delta call IV (vol points)",
                          f"{(p25.iv - c25.iv) * 100:+.1f} vol points", "text", src)

    # The candidates: one per (expiry, delta target), liquid ones only.
    candidates, thin = [], 0
    for e in expiries:
        side = [q for q in by_expiry[e] if q.right == right]
        liquid = [q for q in side if q.spread_pct <= MAX_SPREAD_PCT]
        thin += len(side) - len(liquid)
        for target in DELTA_TARGETS:
            q = _nearest_delta(liquid, target)
            if q and q not in candidates:
                candidates.append(q)
    if not candidates:
        sheet.gaps.append(f"option candidates (no {right}-side contract with a spread under {MAX_SPREAD_PCT:.0f}%)")
        return
    if thin:
        sheet.add(f"Contracts excluded for a bid-ask spread over {MAX_SPREAD_PCT:.0f}% of mid (sampled expiries)",
                  thin, "level", src)

    cross = _cross_check(candidates) if meta["source"].startswith("Cboe") else {}
    if cross is None:
        sheet.gaps.append("second-source cross-check of the candidates (Alpaca unavailable)")
        cross = {}

    for q in candidates:
        dte = (q.expiry - today).days
        be = q.strike + q.mid if q.right == "C" else q.strike - q.mid
        be_move = (be / spot - 1) * 100
        sd_move = q.iv * math.sqrt(max(dte, 1) / 365) * 100
        theta_pct = abs(q.theta) / q.mid * 100 if q.theta is not None and q.mid else None
        parts = [
            f"{dte} days to expiry",
            f"spans earnings: {spans_earnings(q.expiry)}",
            f"mid ${q.mid:.2f} (bid {q.bid:.2f} / ask {q.ask:.2f}), ${q.mid * 100:,.0f} per contract",
            f"spread {q.spread_pct:.1f}% of mid",
            f"IV {q.iv * 100:.1f}%",
            f"delta {q.delta:.2f}",
        ]
        if q.theta is not None:
            parts.append(f"theta -${abs(q.theta):.3f}/day ({theta_pct:.1f}% of premium per day)")
        if q.vega is not None:
            parts.append(f"vega ${q.vega:.3f} per vol point")
        parts += [
            f"breakeven at expiry ${be:.2f} ({be_move:+.1f}% from spot)",
            f"1-standard-deviation move to expiry at this IV {sd_move:.1f}%",
            f"breakeven move / 1-sd move {abs(be_move) / sd_move:.2f}x" if sd_move else "",
            f"open interest {q.open_interest:,.0f}" if q.open_interest is not None else "open interest unknown",
            f"volume {q.volume:,.0f}" if q.volume is not None else "volume unknown",
        ]
        if q.symbol in cross:
            parts.append(cross[q.symbol])
        sheet.add(f"Candidate {q.symbol} ({q.expiry} {q.strike:g}{q.right})",
                  "; ".join(p for p in parts if p), "text", src)


def _cross_check(candidates: list[Quote]) -> dict[str, str] | None:
    """Per candidate: a note when the second source disagrees. None when it cannot be asked."""
    try:
        other = {q.symbol: q for q in fetch_alpaca("", [q.symbol for q in candidates])}
    except Exception as exc:  # noqa: BLE001
        logger.info("options: Alpaca cross-check unavailable: %s", exc)
        return None
    notes = {}
    for q in candidates:
        o = other.get(q.symbol)
        if o is None or o.iv is None or o.delta is None:
            notes[q.symbol] = "second source: no greeks for this contract"
            continue
        if abs(o.iv - q.iv) > MAX_IV_DISAGREEMENT or abs(o.delta - q.delta) > MAX_DELTA_DISAGREEMENT:
            notes[q.symbol] = (f"SOURCES DISAGREE: second source IV {o.iv * 100:.1f}%, delta {o.delta:.2f}")
        else:
            notes[q.symbol] = "second source agrees on IV and delta"
    return notes
