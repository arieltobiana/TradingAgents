"""Deterministic fact sheet: the numbers every agent is allowed to reason from.

The agents pass prose to each other, so an arithmetic slip in one report
becomes a premise for every agent downstream: a fundamentals analyst once read
receivables +341% against revenue "4.7x" and called receivables the faster of
the two, and three later agents built a thesis on it. This module computes the
growth rates, ratios and comparisons in code, once, so no agent has to derive
one. Each fact carries an id (``F1``..) that agents cite and the fact checker
matches against.

Point in time: prices come from ``load_ohlcv`` (rows after the analysis date
dropped). This vendor dates statements by period end, not filing date, and
serves restated values, so a run dated in the past treats a quarter as unknown
until a conservative filing lag has passed (longer for a fiscal Q4, which is
filed with the 10-K, and for non-US listings). Everything is best-effort: a
section that cannot be computed is reported as a gap, never as zero.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv, yf_retry

logger = logging.getLogger(__name__)

# 10-Q deadlines are 40-45 days after quarter end; a 10-K (fiscal Q4) is 60-90
# days; foreign filers are later still.
QUARTER_LAG_DAYS = 45
FISCAL_YEAR_END_LAG_DAYS = 90
NON_US_LAG_DAYS = 120
INSIDER_WINDOW_DAYS = 90
MIN_RECEIVABLES_BASE_DAYS = 5
_YEAR_AGO_TOLERANCE_DAYS = 20
_QUARTER_GAP_DAYS = (70, 110)


@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    value: float | str
    unit: str  # "money", "price", "pct", "x", "days", "level", "date", "text"
    source: str
    currency: str = "USD"

    def display(self) -> str:
        v = self.value
        if isinstance(v, str):
            return v
        if self.unit == "money":
            return _money(v, self.currency)
        if self.unit == "price":
            return f"{v:,.2f}"
        if self.unit == "pct":
            signed = any(w in self.label.lower() for w in ("change", "growth", " vs "))
            return f"{v:+.1f}%" if signed else f"{v:.1f}%"
        if self.unit == "x":
            return f"{v:.2f}x"
        if self.unit == "days":
            return f"{v:.0f} days"
        if self.unit == "level":
            return f"{v:.1f}"
        return f"{v:,.2f}"


def _money(v: float, currency: str) -> str:
    a, sign = abs(v), "-" if v < 0 else ""
    prefix = "$" if currency == "USD" else f"{currency} "
    if a >= 1e9:
        return f"{sign}{prefix}{a / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{sign}{prefix}{a / 1e6:,.1f}M"
    return f"{sign}{prefix}{a:,.0f}"


class _Sheet:
    def __init__(self):
        self.facts: list[Fact] = []
        self.gaps: list[str] = []
        self.currency = "USD"

    def add(self, label, value, unit, source) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            value = float(value)
            if not math.isfinite(value):
                return
        currency = self.currency if unit == "money" else "USD"
        self.facts.append(Fact(f"F{len(self.facts) + 1}", label, value, unit, source, currency))


# ---------------------------------------------------------------- price facts

def _session_open(trade_date: str) -> bool:
    """Whether ``trade_date`` is today and the US cash session has not closed."""
    now = datetime.now(ZoneInfo("America/New_York"))
    return trade_date == get_current_date() and now.weekday() < 5 and now.hour < 16


def _price_facts(sheet: _Sheet, symbol: str, trade_date: str, asset_type: str) -> None:
    df = load_ohlcv(symbol, trade_date, fill_gaps=False).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[(df["Date"] <= pd.to_datetime(trade_date)) & (df["Close"] > 0)]
    df = df.sort_values("Date").reset_index(drop=True)
    if len(df) < 2:
        sheet.gaps.append("price history (fewer than 2 bars)")
        return

    close, high, low = df["Close"], df["High"], df["Low"]
    last = float(close.iloc[-1])
    last_date = f"{df['Date'].iloc[-1]:%Y-%m-%d}"
    src = f"daily bars through {last_date}"
    if asset_type == "stock" and last_date == trade_date and _session_open(trade_date):
        src += " (the last bar is an unfinished session: its close is a live price)"
    sheet.add("Latest close", last, "price", src)
    sheet.add("Latest bar date", last_date, "date", src)

    for n in (1, 5, 20, 60):
        if len(close) > n:
            sheet.add(f"Price change over last {n} sessions",
                      (last / float(close.iloc[-1 - n]) - 1) * 100, "pct", src)

    peak = float(close.tail(10).max())
    sheet.add("Highest close in last 10 sessions", peak, "price", src)
    sheet.add("Current close vs highest close of last 10 sessions", (last / peak - 1) * 100, "pct", src)

    yr = df[df["Date"] > df["Date"].iloc[-1] - pd.Timedelta(days=365)]
    sheet.add("52-week closing high", float(yr["Close"].max()), "price", src)
    sheet.add("52-week closing low", float(yr["Close"].min()), "price", src)
    sheet.add("Current close vs 52-week closing high", (last / float(yr["Close"].max()) - 1) * 100, "pct", src)

    for n in (50, 200):
        if len(close) >= n:
            sma = float(close.tail(n).mean())
            sheet.add(f"{n}-day simple moving average", sma, "price", src)
            sheet.add(f"Close vs {n}-day SMA", (last / sma - 1) * 100, "pct", src)
        else:
            sheet.gaps.append(f"{n}-day SMA (only {len(close)} bars)")

    sheet.add("10-day EMA", float(close.ewm(span=10, adjust=False).mean().iloc[-1]), "price", src)

    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    sheet.add("RSI(14)", float(rsi.iloc[-1]), "level", src)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    sheet.add("ATR(14), average daily range", atr, "price", src)
    sheet.add("ATR(14) as % of close", atr / last * 100, "pct", src)
    for mult in (1.0, 1.5, 2.0):
        sheet.add(f"Price {mult:g} ATR below close (stop reference)", last - mult * atr, "price", src)

    daily = close.pct_change().dropna().tail(20)
    if len(daily) >= 10:
        periods = 365 if asset_type == "crypto" else 252
        sheet.add("Annualized volatility, last 20 sessions", float(daily.std()) * periods ** 0.5 * 100, "pct", src)


# ---------------------------------------------------------- fundamental facts

def _row(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in frame.index:
            s = frame.loc[name]
            if isinstance(s, pd.DataFrame):  # a duplicated row label
                s = s.iloc[0]
            s = pd.to_numeric(s, errors="coerce").dropna()
            if not s.empty:
                return s
    return None


def _published(frame, trade_date: str, fiscal_year_ends: set, non_us: bool) -> pd.DataFrame:
    """Quarterly columns known on ``trade_date``, newest first."""
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    live = trade_date >= get_current_date()
    run = pd.Timestamp(trade_date)
    keep = []
    for col in frame.columns:
        end = pd.to_datetime(col, errors="coerce")
        if pd.isna(end) or end > run:
            continue
        if not live:
            lag = (NON_US_LAG_DAYS if non_us else
                   FISCAL_YEAR_END_LAG_DAYS if end.normalize() in fiscal_year_ends else QUARTER_LAG_DAYS)
            if end + pd.Timedelta(days=lag) > run:
                continue
        keep.append(col)
    out = frame[keep]
    return out[sorted(out.columns, key=pd.Timestamp, reverse=True)]


def _dated(s: pd.Series, anchor, days: int, tolerance: int):
    """The label of the value ``days`` before ``anchor``, or None."""
    anchor = pd.Timestamp(anchor)
    for d in s.index:
        if abs((anchor - pd.Timestamp(d)).days - days) <= tolerance:
            return d
    return None


def _growth(sheet: _Sheet, name: str, s: pd.Series, src: str, yoy_rate: bool = True) -> dict:
    """Latest, quarter-on-quarter and year-on-year for one quarterly series."""
    out: dict = {}
    latest_d, latest = s.index[0], float(s.iloc[0])
    out["latest"] = latest
    sheet.add(f"{name}, latest quarter ({pd.Timestamp(latest_d):%Y-%m-%d})", latest, "money", src)
    lo, hi = _QUARTER_GAP_DAYS
    prior = next((d for d in s.index[1:] if lo <= (pd.Timestamp(latest_d) - pd.Timestamp(d)).days <= hi), None)
    if prior is not None and s[prior] > 0 and latest >= 0:
        sheet.add(f"{name} growth vs prior quarter", (latest / float(s[prior]) - 1) * 100, "pct", src)
    yago = _dated(s, latest_d, 365, _YEAR_AGO_TOLERANCE_DAYS)
    if yago is not None:
        base = float(s[yago])
        sheet.add(f"{name}, same quarter a year earlier ({pd.Timestamp(yago):%Y-%m-%d})", base, "money", src)
        if yoy_rate and base > 0 and latest >= 0:
            g = (latest / base - 1) * 100
            out["yoy_pct"] = g
            sheet.add(f"{name} growth year over year (percent)", g, "pct", src)
            sheet.add(f"{name} year over year as a multiple (latest / year-ago; {g:+.1f}% growth)",
                      latest / base, "x", src)
    return out


def _fundamental_facts(sheet: _Sheet, symbol: str, trade_date: str) -> None:
    canonical = normalize_symbol(symbol)
    t = yf.Ticker(canonical)
    try:
        sheet.currency = (yf_retry(lambda: t.info) or {}).get("financialCurrency") or "USD"
    except Exception:  # noqa: BLE001 — unknown currency is stated, not assumed away
        sheet.currency = "unknown-currency"
    try:
        annual = yf_retry(lambda: t.income_stmt)
        fy_ends = {pd.Timestamp(c).normalize() for c in annual.columns} if annual is not None else set()
    except Exception:  # noqa: BLE001 — without it every quarter takes the longer lag
        fy_ends = None
    non_us = "." in canonical
    lag_args = (fy_ends if fy_ends is not None else _AllQuarters(), non_us)

    inc = _published(yf_retry(lambda: t.quarterly_income_stmt), trade_date, *lag_args)
    bal = _published(yf_retry(lambda: t.quarterly_balance_sheet), trade_date, *lag_args)
    cf = _published(yf_retry(lambda: t.quarterly_cashflow), trade_date, *lag_args)
    if inc.empty:
        sheet.gaps.append("quarterly income statement")
        return
    src = "quarterly statements, dated by period end, as currently reported (may include restatements)"

    rev = _row(inc, "Total Revenue", "Operating Revenue")
    if rev is not None:
        _growth(sheet, "Revenue", rev, src)
    ni = _row(inc, "Net Income", "Net Income Common Stockholders")
    if ni is not None:
        _growth(sheet, "Net income", ni, src)
    gp = _row(inc, "Gross Profit")
    if rev is not None and gp is not None:
        margin = (gp / rev.reindex(gp.index) * 100).dropna()
        if not margin.empty:
            sheet.add("Gross margin, latest quarter", float(margin.iloc[0]), "pct", src)
            yago = _dated(margin, margin.index[0], 365, _YEAR_AGO_TOLERANCE_DAYS)
            if yago is not None:
                sheet.add("Gross margin, same quarter a year earlier", float(margin[yago]), "pct", src)

    if not bal.empty:
        ar = _row(bal, "Accounts Receivable", "Receivables")
        if ar is not None:
            # Receivables are judged against revenue through days sales
            # outstanding (91-day quarter), not by comparing two growth rates:
            # a growth rate off a tiny base (IREN: $1.6M to $21.1M, +1,247%)
            # reads as alarming while collections actually sped up.
            dso = (ar / rev.reindex(ar.index) * 91).dropna() if rev is not None else pd.Series(dtype=float)
            yago_dso = None
            if not dso.empty:
                top = dso.index[0]
                d = _dated(dso, top, 365, _YEAR_AGO_TOLERANCE_DAYS)
                yago_dso = float(dso[d]) if d is not None else None
            tiny_base = yago_dso is not None and yago_dso < MIN_RECEIVABLES_BASE_DAYS
            _growth(sheet, "Accounts receivable", ar, src, yoy_rate=not tiny_base)
            if tiny_base:
                sheet.gaps.append(
                    f"accounts receivable growth rate (year-ago base was {yago_dso:.1f} days of sales, "
                    "too small for a rate to mean anything; use days sales outstanding)")
            if not dso.empty:
                sheet.add("Days sales outstanding, latest quarter", float(dso.iloc[0]), "days", src)
                prior = _dated(dso, top, 91, _YEAR_AGO_TOLERANCE_DAYS)
                if prior is not None:
                    sheet.add("Days sales outstanding, prior quarter", float(dso[prior]), "days", src)
                if yago_dso is not None:
                    sheet.add("Days sales outstanding, year-ago quarter", yago_dso, "days", src)
                if yago_dso is not None and not tiny_base:
                    now = float(dso.iloc[0])
                    faster, moved = ("FASTER", "rose") if now > yago_dso else ("SLOWER", "fell")
                    sheet.add("Receivables vs revenue, year over year",
                              f"receivables grew {faster} than revenue: days sales outstanding {moved} "
                              f"from {yago_dso:.0f} to {now:.0f} days", "text", src)
        for label, names in (("Cash and equivalents", ("Cash And Cash Equivalents",)),
                             ("Total debt", ("Total Debt",)),
                             ("Inventory", ("Inventory",))):
            s = _row(bal, *names)
            if s is not None:
                sheet.add(f"{label}, latest quarter", float(s.iloc[0]), "money", src)

    if not cf.empty:
        fcf = _row(cf, "Free Cash Flow")
        if fcf is not None:
            # Operating cash flow minus what this vendor classes as capital
            # spending, which can be broader than the filing's PP&E line
            # (IREN FY2026: $4.45B here, $3.0B in the 10-K).
            sheet.add("Free cash flow, latest quarter (vendor's definition of capital spending; "
                      "may differ from the filing)", float(fcf.iloc[0]), "money", src)


class _AllQuarters(set):
    """Stands in for unknown fiscal year ends: every quarter takes the Q4 lag."""

    def __contains__(self, item) -> bool:
        return True


def _insider_facts(sheet: _Sheet, symbol: str, trade_date: str) -> None:
    data = yf_retry(lambda: yf.Ticker(normalize_symbol(symbol)).insider_transactions)
    if data is None or data.empty or "Start Date" not in data:
        sheet.gaps.append("insider transactions")
        return
    end = pd.Timestamp(trade_date)
    start = end - pd.Timedelta(days=INSIDER_WINDOW_DAYS)
    dates = pd.to_datetime(data["Start Date"], errors="coerce")
    # The vendor serves only recent filings; a window it does not reach back to
    # is unobserved, not a window with no insider activity.
    if dates.min() > start:
        sheet.gaps.append(f"insider transactions before {dates.min():%Y-%m-%d} (vendor serves recent filings only)")
        return
    d = data[(dates <= end) & (dates > start)]
    text = d.get("Text", pd.Series("", index=d.index)).fillna("").astype(str)
    value = pd.to_numeric(d.get("Value", pd.Series(0.0, index=d.index)), errors="coerce").fillna(0)
    is_sale = text.str.contains("Sale", case=False)
    is_buy = text.str.contains("Purchase", case=False)
    unstated = (text.str.strip() == "") & (value > 0)
    src = f"insider filings, {INSIDER_WINDOW_DAYS} days to {trade_date} (dated by transaction; filed up to 2 business days later)"
    currency, sheet.currency = sheet.currency, "USD"
    sheet.add(f"Insider open-market sales, last {INSIDER_WINDOW_DAYS} days (value)", float(value[is_sale].sum()), "money", src)
    sheet.add(f"Insider open-market purchases, last {INSIDER_WINDOW_DAYS} days (value)", float(value[is_buy].sum()), "money", src)
    if unstated.any():
        sheet.add(f"Insider transactions with no stated type, last {INSIDER_WINDOW_DAYS} days (value)",
                  float(value[unstated].sum()), "money", src)
    sales = d[is_sale]
    if not sales.empty:
        who = sales["Insider"].fillna("?") + " (" + sales.get("Position", pd.Series("", index=sales.index)).fillna("") + ")"
        for name, v in value[is_sale].groupby(who).sum().sort_values(ascending=False).items():
            sheet.add(f"Insider sales by {name}", float(v), "money", src)
    sheet.currency = currency


def _calendar_facts(sheet: _Sheet, symbol: str, trade_date: str) -> None:
    # The vendor's calendar is live-only: a past run would see a later date.
    if trade_date < get_current_date():
        return
    cal = yf_retry(lambda: yf.Ticker(normalize_symbol(symbol)).calendar) or {}
    dates = cal.get("Earnings Date") or []
    if dates:
        sheet.add("Next scheduled earnings date", str(dates[0]), "date", "vendor calendar (live)")


# ------------------------------------------------------------------ assembly

def build_fact_sheet(symbol: str, trade_date: str, asset_type: str = "stock") -> dict:
    """Compute the fact sheet. Returns ``{"facts": [Fact dicts], "gaps": [str]}``."""
    sheet = _Sheet()
    sections = [("price", lambda s, sym, d: _price_facts(s, sym, d, asset_type))]
    if asset_type == "stock":
        sections += [("fundamentals", _fundamental_facts), ("insiders", _insider_facts),
                     ("calendar", _calendar_facts)]
    for name, fn in sections:
        try:
            fn(sheet, symbol, trade_date)
        except Exception as exc:  # noqa: BLE001 — a missing section is a gap, not a failed run
            logger.warning("fact sheet: %s unavailable for %s: %s", name, symbol, exc)
            sheet.gaps.append(f"{name} ({type(exc).__name__})")
    return {"facts": [asdict(f) for f in sheet.facts], "gaps": sheet.gaps}
