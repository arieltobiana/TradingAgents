"""Deterministic fact sheet: the numbers every agent is allowed to reason from.

The agents pass prose to each other, so an arithmetic slip in one report
becomes a premise for every agent downstream: a fundamentals analyst once read
receivables +341% against revenue "4.7x" and called receivables the faster of
the two, and three later agents built a thesis on it. This module computes the
growth rates, ratios and comparisons in code, once, so no agent has to derive
one. Each fact carries an id (``F1``..) that agents cite and the fact checker
matches against.

Point in time: prices come from ``load_ohlcv`` (rows after the analysis date
dropped); a quarter counts only once it is plausibly published — for a run
dated in the past, ``REPORTING_LAG_DAYS`` after its period end — because this
vendor dates statements by period end, not filing date. Everything is
best-effort: a section that cannot be computed is reported as unavailable,
never as zero.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv, yf_retry

logger = logging.getLogger(__name__)

# SEC deadlines are 40-45 days after quarter end for most filers; a historical
# run treats a quarter as unknown until then.
REPORTING_LAG_DAYS = 45
INSIDER_WINDOW_DAYS = 90


@dataclass(frozen=True)
class Fact:
    id: str
    label: str
    value: float | str
    unit: str  # "usd", "price", "pct", "x", "days", "date", "text", "count"
    source: str

    def display(self) -> str:
        v = self.value
        if isinstance(v, str):
            return v
        if self.unit == "usd":
            return _usd(v)
        if self.unit == "price":
            return f"{v:,.2f}"
        if self.unit == "pct":
            signed = any(w in self.label.lower() for w in ("change", "growth", " vs "))
            return f"{v:+.1f}%" if signed else f"{v:.1f}%"
        if self.unit == "x":
            return f"{v:.2f}x"
        if self.unit == "days":
            return f"{v:.0f} days"
        if self.unit == "count":
            return f"{v:.0f}"
        return f"{v:,.2f}"


def _usd(v: float) -> str:
    a = abs(v)
    sign = "-" if v < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.1f}M"
    return f"{sign}${a:,.0f}"


class _Sheet:
    def __init__(self):
        self.facts: list[Fact] = []
        self.gaps: list[str] = []

    def add(self, label, value, unit, source) -> None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return
        if not isinstance(value, str):
            value = float(value)
        self.facts.append(Fact(f"F{len(self.facts) + 1}", label, value, unit, source))


# ---------------------------------------------------------------- price facts

def _price_facts(sheet: _Sheet, symbol: str, trade_date: str) -> None:
    df = load_ohlcv(symbol, trade_date, fill_gaps=False)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Date"] <= pd.to_datetime(trade_date)].sort_values("Date").reset_index(drop=True)
    if len(df) < 2:
        sheet.gaps.append("price history (fewer than 2 bars)")
        return

    close, high, low = df["Close"], df["High"], df["Low"]
    last = float(close.iloc[-1])
    src = f"daily bars through {df['Date'].iloc[-1]:%Y-%m-%d}"
    sheet.add("Latest close", last, "price", src)
    sheet.add("Latest bar date", f"{df['Date'].iloc[-1]:%Y-%m-%d}", "date", src)

    for n in (1, 5, 20, 60):
        if len(close) > n:
            sheet.add(f"Price change over last {n} sessions",
                      (last / float(close.iloc[-1 - n]) - 1) * 100, "pct", src)

    recent = close.tail(10)
    peak = float(recent.max())
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

    ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
    sheet.add("10-day EMA", ema10, "price", src)

    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    sheet.add("RSI(14)", float(rsi.iloc[-1]), "count", src)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    sheet.add("ATR(14), average daily range", atr, "price", src)
    sheet.add("ATR(14) as % of close", atr / last * 100, "pct", src)
    for mult in (1.0, 1.5, 2.0):
        sheet.add(f"Price {mult:g} ATR below close (stop reference)", last - mult * atr, "price", src)

    daily = close.pct_change().dropna().tail(20)
    if len(daily) >= 10:
        sheet.add("Annualized volatility, last 20 sessions", float(daily.std()) * (252 ** 0.5) * 100, "pct", src)


# ---------------------------------------------------------- fundamental facts

def _row(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in frame.index:
            s = pd.to_numeric(frame.loc[name], errors="coerce")
            if s.notna().any():
                return s
    return None


def _published(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    """Quarterly columns known on ``trade_date``, newest first."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    cols = pd.to_datetime(frame.columns, errors="coerce")
    cutoff = pd.Timestamp(trade_date)
    if trade_date < get_current_date():
        cutoff -= pd.Timedelta(days=REPORTING_LAG_DAYS)
    keep = [c for c, d in zip(frame.columns, cols, strict=True) if pd.notna(d) and d <= cutoff]
    out = frame[keep]
    return out[sorted(out.columns, key=pd.Timestamp, reverse=True)]


def _growth(sheet: _Sheet, name: str, s: pd.Series, src: str) -> dict:
    """Latest, QoQ and YoY for one quarterly series; returns the numbers used."""
    out: dict = {}
    s = s.dropna()
    if s.empty:
        return out
    latest_d = s.index[0]
    latest = float(s.iloc[0])
    out["latest"] = latest
    sheet.add(f"{name}, latest quarter ({pd.Timestamp(latest_d):%Y-%m-%d})", latest, "usd", src)
    if len(s) > 1 and s.iloc[1] > 0 and latest >= 0:
        sheet.add(f"{name} growth vs prior quarter", (latest / float(s.iloc[1]) - 1) * 100, "pct", src)
    yago = [d for d in s.index if abs((pd.Timestamp(latest_d) - pd.Timestamp(d)).days - 365) <= 20]
    if yago:
        base = float(s[yago[0]])
        out["year_ago"] = base
        sheet.add(f"{name}, same quarter a year earlier ({pd.Timestamp(yago[0]):%Y-%m-%d})", base, "usd", src)
        if base > 0 and latest >= 0:
            g = (latest / base - 1) * 100
            out["yoy_pct"] = g
            sheet.add(f"{name} growth year over year (percent)", g, "pct", src)
            sheet.add(f"{name} year over year as a multiple (latest / year-ago; {g:+.1f}% growth)",
                      latest / base, "x", src)
    return out


def _fundamental_facts(sheet: _Sheet, symbol: str, trade_date: str) -> None:
    t = yf.Ticker(normalize_symbol(symbol))
    inc = _published(yf_retry(lambda: t.quarterly_income_stmt), trade_date)
    bal = _published(yf_retry(lambda: t.quarterly_balance_sheet), trade_date)
    cf = _published(yf_retry(lambda: t.quarterly_cashflow), trade_date)
    if inc.empty:
        sheet.gaps.append("quarterly income statement")
        return
    src = "quarterly statements (dated by period end)"

    rev = _row(inc, "Total Revenue", "Operating Revenue")
    rev_g = _growth(sheet, "Revenue", rev, src) if rev is not None else {}
    ni = _row(inc, "Net Income", "Net Income Common Stockholders")
    if ni is not None:
        _growth(sheet, "Net income", ni, src)
    gp = _row(inc, "Gross Profit")
    if rev is not None and gp is not None:
        margin = (gp / rev * 100).dropna()
        if not margin.empty:
            sheet.add("Gross margin, latest quarter", float(margin.iloc[0]), "pct", src)
            if len(margin) > 4:
                sheet.add("Gross margin, same quarter a year earlier", float(margin.iloc[4]), "pct", src)

    if not bal.empty:
        ar = _row(bal, "Accounts Receivable", "Receivables")
        if ar is not None:
            ar_g = _growth(sheet, "Accounts receivable", ar, src)
            if "yoy_pct" in ar_g and "yoy_pct" in rev_g:
                faster = "FASTER" if ar_g["yoy_pct"] > rev_g["yoy_pct"] else "SLOWER"
                sheet.add(
                    "Receivables vs revenue, year over year",
                    f"receivables grew {faster} than revenue "
                    f"({ar_g['yoy_pct']:+.1f}% vs {rev_g['yoy_pct']:+.1f}%)",
                    "text", src,
                )
            if rev is not None:
                # Days sales outstanding on a 91-day quarter, per quarter where both exist.
                dso = (ar / rev.reindex(ar.index) * 91).dropna()
                labels = ["latest quarter", "prior quarter", None, None, "year-ago quarter"]
                for i, lab in enumerate(labels):
                    if lab and i < len(dso):
                        sheet.add(f"Days sales outstanding, {lab}", float(dso.iloc[i]), "days", src)
        for label, names in (("Cash and equivalents", ("Cash And Cash Equivalents",)),
                             ("Total debt", ("Total Debt",)),
                             ("Inventory", ("Inventory",))):
            s = _row(bal, *names)
            if s is not None and s.notna().any():
                sheet.add(f"{label}, latest quarter", float(s.dropna().iloc[0]), "usd", src)

    if not cf.empty:
        fcf = _row(cf, "Free Cash Flow")
        if fcf is not None and fcf.notna().any():
            sheet.add("Free cash flow, latest quarter", float(fcf.dropna().iloc[0]), "usd", src)


def _insider_facts(sheet: _Sheet, symbol: str, trade_date: str) -> None:
    t = yf.Ticker(normalize_symbol(symbol))
    data = yf_retry(lambda: t.insider_transactions)
    if data is None or data.empty or "Start Date" not in data:
        sheet.gaps.append("insider transactions")
        return
    end = pd.Timestamp(trade_date)
    d = data[(pd.to_datetime(data["Start Date"]) <= end)
             & (pd.to_datetime(data["Start Date"]) > end - pd.Timedelta(days=INSIDER_WINDOW_DAYS))]
    text = d.get("Text", pd.Series("", index=d.index)).fillna("").astype(str)
    value = pd.to_numeric(d.get("Value"), errors="coerce").fillna(0)
    sales, buys = d[text.str.contains("Sale", case=False)], d[text.str.contains("Purchase", case=False)]
    src = f"insider filings, {INSIDER_WINDOW_DAYS} days to {trade_date} (dated by transaction)"
    sheet.add(f"Insider open-market sales, last {INSIDER_WINDOW_DAYS} days (value)", float(value[sales.index].sum()), "usd", src)
    sheet.add(f"Insider open-market purchases, last {INSIDER_WINDOW_DAYS} days (value)", float(value[buys.index].sum()), "usd", src)
    if not sales.empty:
        by = value[sales.index].groupby(sales["Insider"] + " (" + sales["Position"].fillna("") + ")").sum()
        for who, v in by.sort_values(ascending=False).items():
            sheet.add(f"Insider sales by {who}", float(v), "usd", src)


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
    sections = [("price", _price_facts)]
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
