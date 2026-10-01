"""A local record of every pick the scenario ranking makes, and a review of how each one turned out.

The ranking is only worth keeping if its picks make money, and the only way to find out is to write each
one down when it is made and look at it again after its exit date. Each run of
``python -m tradingagents.options.scenario`` appends one line here; ``review`` later prices every pick
against the option chains the nightly snapshot job archived (``tradingagents.options.archive``).

Layout and rotation. One file per calendar month, ``<cache>/option_picks/picks-YYYY-MM.jsonl``, appended
under an flock. A month's file is gzipped once that month ended more than ``GZIP_AFTER_DAYS`` ago and is
never deleted: the log is the evidence, and deleting old picks would quietly make the record look better
than it was. Readers take plain and gzipped files alike. (The archive of chains is gzip JSON already and
the cron log is rotated by ``cron-dashboard/rotate_logs.sh``.)

How a pick is scored. Entry is the ask at the time of the pick. Exit is the bid in the last archived
chain at or before the exit date (a chain older than ``MAX_EXIT_STALENESS_DAYS`` is refused: a missing
exit quote is reported as not scorable, never filled in). Shares are compared from the underlying
price recorded at the pick to the one in that chain. Before the exit date a pick is "open" and is marked
to the newest archived chain, labelled as a mark and not a result.

    python -m tradingagents.options.picklog review [--symbol MU]
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import logging
import os
import re
import statistics
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = 1
GZIP_AFTER_DAYS = 90
MAX_EXIT_STALENESS_DAYS = 3
_NAME = re.compile(r"^picks-(?P<month>\d{4}-\d{2})\.jsonl(?P<gz>\.gz)?$")


def pick_dir(cache_dir: str) -> Path:
    return Path(cache_dir) / "option_picks"


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def rotate(cache_dir: str, now: datetime | None = None) -> list[Path]:
    """Gzip every monthly file that ended more than GZIP_AFTER_DAYS ago. Returns the files written."""
    now = now or datetime.now(timezone.utc)
    root, done = pick_dir(cache_dir), []
    for path in sorted(root.glob("picks-*.jsonl")):
        m = _NAME.match(path.name)
        if not m:
            continue
        first = date.fromisoformat(m["month"] + "-01")
        month_end = (first.replace(day=28) + timedelta(days=4)).replace(day=1)   # first day of the next month
        if (now.date() - month_end).days <= GZIP_AFTER_DAYS:
            continue
        target = path.with_name(path.name + ".gz")
        _atomic_write(target, gzip.compress(path.read_bytes()))
        path.unlink()
        done.append(target)
    return done


def record_pick(cache_dir: str, entry: dict, now: datetime | None = None) -> Path:
    """Append ``entry`` (one run) to this month's file, stamping when and under which schema."""
    now = now or datetime.now(timezone.utc)
    root = pick_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"schema": SCHEMA, "logged_at": now.isoformat(), **entry}, sort_keys=True, default=str)
    path = root / f"picks-{now:%Y-%m}.jsonl"
    with open(root / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(path, "a") as f:
            f.write(line + "\n")
        rotate(cache_dir, now)
    return path


def load_picks(cache_dir: str) -> list[dict]:
    """Every logged run, oldest first. A torn or unreadable line is skipped, never fatal."""
    root = pick_dir(cache_dir)
    if not root.is_dir():
        return []
    with open(root / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)    # rotation holds the exclusive lock; never read half of it
        return _read_all(root)


def _read_all(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.glob("picks-*.jsonl*")):
        if not _NAME.match(path.name):
            continue
        try:
            text = (gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()).decode()
        except (OSError, EOFError, ValueError) as exc:
            logger.warning("option picks: cannot read %s (%s)", path, exc)
            continue
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("logged_at"):
                out.append(row)
    return sorted(out, key=lambda r: r["logged_at"])


def open_pick_symbols(cache_dir: str, today: date) -> list[str]:
    """Tickers with a pick whose exit date has not passed: the nightly job archives them up to and including that
    day, because scoring accepts only a chain fetched by the end of the exit date, so a later one cannot help."""
    cutoff = today
    seen = []
    for row in load_picks(cache_dir):
        try:
            if date.fromisoformat(row["exit_date"]) >= cutoff and row["symbol"] not in seen:
                seen.append(row["symbol"])
        except (KeyError, ValueError):
            continue
    return seen


def _row(r) -> dict:
    return {"symbol": r.symbol, "strike": r.strike, "expiry": r.expiry.isoformat(), "ask": r.ask,
            "spread_pct": r.spread_pct, "iv": r.iv, "delta": r.delta, "breakeven": r.breakeven,
            "roi_target": r.roi_target, "roi_half": r.roi_half, "roi_flat": r.roi_flat, "roi_down": r.roi_down}


def make_entry(symbol: str, right: str, spot: float, target: float, exit_date: date, budget: float | None,
               iv_shift: float, meta: dict, picks, ctx: dict, snapshot: Path | None) -> dict:
    """The record of one run: the view, what the market looked like, and what was picked (or why nothing was)."""
    entry = {"symbol": symbol, "right": right, "spot": spot, "target": target, "exit_date": exit_date.isoformat(),
             "budget": budget, "iv_shift": iv_shift, "source": meta.get("source"), "as_of": meta.get("as_of"),
             "feed": meta.get("feed"), "snapshot": str(snapshot) if snapshot else None,
             "market_vol": ctx.get("sigma"), "pick": None, "leverage": None, "qualifying": 0}
    if picks is not None:
        entry.update(pick=_row(picks.best), leverage=_row(picks.leverage) if picks.leverage else None,
                     qualifying=picks.qualifying,
                     pick_after_drop_roi_target=picks.best_after_drop.roi_target)
    return entry


@dataclass
class Outcome:
    logged: str
    symbol: str
    contract: str | None          # None for a request that named no contract
    right: str
    status: str                   # "settled" | "open" | "not scorable" | "no contract"
    entry_ask: float | None
    target: float
    exit_date: str
    exit_bid: float | None = None
    ret: float | None = None
    shares_ret: float | None = None
    stock_at_exit: float | None = None        # the stock in the chain used: at the exit date once settled, else now
    predicted_at_target: float | None = None
    reason: str = ""
    strike: float | None = None
    expiry: str | None = None
    breakeven: float | None = None
    spot_at_pick: float | None = None
    as_of: str | None = None            # when the quotes behind the pick were made, per the source
    # The request, then and now. Everything below is computed here, from what the log and the archive hold.
    logged_at: str | None = None        # when the request was made (ISO, UTC)
    mark_at: str | None = None          # when the chain behind "now" was fetched (ISO, UTC)
    days_left: int | None = None        # days from today to the exit date; negative once it has passed
    pnl_per_contract: float | None = None     # (bid - ask) x 100: one contract, bought at the ask, sold at the bid
    shares_pnl_per_100: float | None = None   # (stock now - stock then) x 100 shares; calls only
    target_progress: float | None = None      # how far the stock has gone from then toward the target (1.0 = there)
    target_touched: bool | None = None        # did any saved price reach the target since the request
    target_touched_on: str | None = None
    view: str | None = None             # "reached" | "missed" | "toward" | "flat" | "away"; None when unknown


VIEW_FLAT_PROGRESS = 0.02       # within 2% of the way either side of where it started counts as "flat"


def _quote_bid(quotes, symbol: str) -> float | None:
    q = next((q for q in quotes if q.symbol.replace(" ", "") == symbol), None)
    return q.bid if q is not None and q.bid > 0 else None


def _compare_view(out: Outcome, row: dict, cache_dir: str, settled: bool) -> None:
    """Fill in how the stock has done against the view: progress to the target, whether it was ever reached."""
    from tradingagents.options.archive import spot_history

    then, target, now = row.get("spot"), row["target"], out.stock_at_exit
    if then and now and target != then:
        out.target_progress = (now - then) / (target - then)
    try:
        since = datetime.fromisoformat(row["logged_at"])
    except (KeyError, ValueError):
        since = None
    if since is not None and then:
        hit = lambda spot: spot >= target if row["right"] == "C" else spot <= target   # noqa: E731
        history = spot_history(cache_dir, row["symbol"], since)
        first = next(((day, spot) for _, day, spot in history if hit(spot)), None)
        out.target_touched = first is not None or hit(then)
        out.target_touched_on = first[0] if first else (out.logged if hit(then) else None)
    if out.target_touched:
        out.view = "reached"
    elif settled and out.stock_at_exit is not None:
        out.view = "missed"
    elif out.target_progress is not None:
        out.view = ("toward" if out.target_progress > VIEW_FLAT_PROGRESS
                    else "away" if out.target_progress < -VIEW_FLAT_PROGRESS else "flat")


def score(row: dict, cache_dir: str, today: date) -> Outcome | None:
    """One logged request against the archived chains: how the stock and the contract have done since."""
    from tradingagents.options.archive import load_chain_snapshot

    p = row.get("pick")
    exit_d = date.fromisoformat(row["exit_date"])
    out = Outcome(row["logged_at"][:10], row["symbol"], p["symbol"] if p else None, row["right"],
                  "not scorable" if p else "no contract", p["ask"] if p else None, row["target"], row["exit_date"],
                  predicted_at_target=p["roi_target"] if p else None, strike=p.get("strike") if p else None,
                  expiry=p.get("expiry") if p else None, breakeven=p.get("breakeven") if p else None,
                  spot_at_pick=row.get("spot"), as_of=row.get("as_of"), logged_at=row["logged_at"],
                  days_left=(exit_d - today).days,
                  reason="" if p else "no contract paid even if the view was right")
    settled = today > exit_d
    # A bare date means the start of that day, so the day after is "through the end of that day": the exit date once
    # it has passed, otherwise today (which is what makes a review reproducible for a given ``today``).
    as_of = (exit_d if settled else today) + timedelta(days=1)
    snap = load_chain_snapshot(cache_dir, row["symbol"], as_of, max_age=timedelta(days=MAX_EXIT_STALENESS_DAYS))
    if snap is None:
        out.reason = (f"no chain archived within {MAX_EXIT_STALENESS_DAYS} days of "
                      f"{'the exit date' if settled else 'now'}; is {row['symbol']} on the nightly snapshot list?")
        return out
    meta, quotes = snap
    out.mark_at = meta.get("fetched_at")
    out.stock_at_exit = meta.get("spot")
    if row["right"] == "C" and out.stock_at_exit and row.get("spot"):
        out.shares_ret = out.stock_at_exit / row["spot"] - 1
        out.shares_pnl_per_100 = (out.stock_at_exit - row["spot"]) * 100
    _compare_view(out, row, cache_dir, settled)
    if not p:
        out.reason = ("no contract paid even if the view was right"
                      + ("" if settled else f"; the stock is marked to the chain of {meta.get('session') or meta.get('as_of')}"))
        return out
    bid = _quote_bid(quotes, p["symbol"])
    if bid is None:
        out.reason = "the contract has no bid in the archived chain (a worthless or missing quote is not filled in)"
        return out
    out.exit_bid, out.ret = bid, bid / p["ask"] - 1
    out.pnl_per_contract = (bid - p["ask"]) * 100
    out.status = "settled" if settled else "open"
    out.reason = "" if settled else f"marked to the chain of {meta.get('session') or meta.get('as_of')}, not a result"
    return out


def review(cache_dir: str, today: date, symbol: str | None = None) -> list[Outcome]:
    rows = [r for r in load_picks(cache_dir) if symbol is None or r.get("symbol") == symbol.upper()]
    return [score(r, cache_dir, today) for r in rows]


def summary(outcomes: list[Outcome]) -> dict:
    """Counts over the settled picks only (an open pick is a mark, not a result), and over the views."""
    picks = [o for o in outcomes if o.status != "no contract"]
    done = [o for o in picks if o.status == "settled" and o.ret is not None]
    paired = [o for o in done if o.shares_ret is not None]
    viewed = [o for o in outcomes if o.view is not None]
    return {"picks": len(picks), "settled": len(done), "open": sum(o.status == "open" for o in picks),
            "not_scorable": sum(o.status == "not scorable" for o in picks),
            "mean_return": sum(o.ret for o in done) / len(done) if done else None,
            "median_return": statistics.median(o.ret for o in done) if done else None,
            "made_money": sum(o.ret > 0 for o in done), "paired": len(paired),
            "beat_shares": sum(o.ret > o.shares_ret for o in paired),
            # The views themselves, whether or not a contract was named: did the stock get to the target?
            "views": len(outcomes), "views_tracked": len(viewed),
            "views_reached": sum(o.view == "reached" for o in viewed),
            "views_missed": sum(o.view == "missed" for o in viewed)}


def review_payload(cache_dir: str, today: date, now: datetime | None = None) -> dict:
    """The review as data, for another program to display. Everything in it was computed here."""
    rows = load_picks(cache_dir)
    outcomes = [score(r, cache_dir, today) for r in rows]
    return {"schema": SCHEMA, "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
            "as_of_date": today.isoformat(), "runs": len(rows),
            "runs_without_pick": sum(o.status == "no contract" for o in outcomes),
            "summary": summary(outcomes), "picks": [asdict(o) for o in outcomes]}


def write_review(cache_dir: str, today: date, now: datetime | None = None) -> Path:
    """Write ``review.json`` beside the log, atomically, so a reader never sees half of it."""
    root = pick_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "review.json"
    _atomic_write(path, json.dumps(review_payload(cache_dir, today, now), indent=1, sort_keys=True).encode())
    return path


_VIEW_WORDS = {"reached": "reached your target", "missed": "did not reach your target", "toward": "moving toward your target",
               "flat": "not moved yet", "away": "moving away from your target"}


def _money(x: float | None, signed: bool = False) -> str:
    return "-" if x is None else (f"{x:+,.2f}" if signed else f"{x:,.2f}")


def _pct(x: float | None) -> str:
    """A signed percent; a figure that rounds to zero carries no sign (\"-0.0%\" reads as a loss that is not there)."""
    if x is None:
        return "-"
    return "0.0%" if abs(x) < 0.0005 else f"{x:+.1%}"


def render_review(outcomes: list[Outcome], total_runs: int) -> str:
    """Each request as then and now: what was true when you asked, what is true at the latest saved chain."""
    if not outcomes:
        return f"{total_runs} runs logged, none to review yet."
    lines = []
    for o in outcomes:
        kind = "call" if o.right == "C" else "put"
        what = f"{o.contract} ({o.status})" if o.contract else "no contract named"
        lines += [f"{o.symbol} {kind}  {what}",
                  f"  asked   {o.logged_at[:16].replace('T', ' ') if o.logged_at else o.logged} UTC   stock {_money(o.spot_at_pick)}"
                  f"   contract ask {_money(o.entry_ask)}   view: {_money(o.target)} by {o.exit_date}"]
        now = (o.mark_at or "")[:16].replace("T", " ")
        stock_chg = (o.stock_at_exit / o.spot_at_pick - 1) if o.stock_at_exit and o.spot_at_pick else None
        lines.append(f"  {'result' if o.status == 'settled' else 'latest'}  {now or '-'} UTC   stock {_money(o.stock_at_exit)} "
                     f"({_pct(stock_chg)})   contract bid {_money(o.exit_bid)} ({_pct(o.ret)})")
        if o.pnl_per_contract is not None or o.shares_pnl_per_100 is not None:
            lines.append(f"  P&L     contract {_money(o.pnl_per_contract, True)} $ per contract (ask in, bid out)"
                         f"   |   100 shares {_money(o.shares_pnl_per_100, True)} $")
        if o.view:
            prog = f" ({0 if abs(o.target_progress) < 0.005 else o.target_progress:.0%} of the way)" if o.target_progress is not None else ""
            touched = f"; first reached {o.target_touched_on}" if o.target_touched_on else ""
            days = ""
            if o.days_left is not None:
                days = f", {o.days_left} days left" if o.days_left >= 0 else f", {-o.days_left} days past the exit date"
            lines.append(f"  view    {_VIEW_WORDS[o.view]}{prog}{touched}{days}")
        if o.reason:
            lines.append(f"  note    {o.reason}")
        lines.append("")
    done = [o for o in outcomes if o.status == "settled" and o.ret is not None]
    if not done:
        lines.append("Nothing has reached its exit date yet. Open figures are marks, not results.")
    else:
        paired = [o for o in done if o.shares_ret is not None]
        beat = sum(o.ret > o.shares_ret for o in paired)
        lines.append(f"Settled: {len(done)} of {len(outcomes)} picks. Mean return {sum(o.ret for o in done) / len(done):+.0%}, "
                     f"median {statistics.median(o.ret for o in done):+.0%}, made money {sum(o.ret > 0 for o in done)} times."
                     + (f" The call beat the shares in {beat} of {len(paired)}." if paired else ""))
        if len(done) < 30:
            lines.append(f"{len(done)} settled picks is a handful: the picks cluster in a few names and dates, so "
                         "read this as a first look, not evidence.")
    lines.append("Returns use the ask in and the bid out, from delayed chains: a result, not a fill.")
    return "\n".join(lines)


def refresh_open(cache_dir: str, today: date, symbol: str | None = None) -> list[str]:
    """Fetch and archive a fresh chain for each ticker with a request still open; returns the tickers refreshed.

    A failed fetch is reported and skipped: the review then uses the newest chain it already has.
    """
    from tradingagents.dataflows.vendors.options import fetch_chain
    from tradingagents.options.archive import record_chain_snapshot

    done = []
    for sym in open_pick_symbols(cache_dir, today):
        if symbol and sym != symbol.upper():
            continue
        try:
            meta, quotes = fetch_chain(sym)
            record_chain_snapshot(cache_dir, sym, meta, quotes)
            done.append(sym)
        except Exception as exc:  # noqa: BLE001 — one ticker must not stop the review
            print(f"{sym}: could not refresh ({type(exc).__name__}: {exc})")
    return done


def main(argv: list[str] | None = None) -> int:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.dataflows.vendors.options import ny_today

    ap = argparse.ArgumentParser(description="Review the logged option picks.")
    ap.add_argument("command", choices=["review"])
    ap.add_argument("--symbol")
    ap.add_argument("--cache-dir", default=DEFAULT_CONFIG.get("data_cache_dir"))
    ap.add_argument("--write", action="store_true", help="also write option_picks/review.json for other programs to read")
    ap.add_argument("--live", action="store_true",
                    help="first fetch and save a fresh chain for every ticker whose exit date has not passed, so 'now' is today")
    args = ap.parse_args(argv)
    today = date.fromisoformat(ny_today())
    if args.live:
        refresh_open(args.cache_dir, today, args.symbol)
    runs = load_picks(args.cache_dir)
    print(render_review(review(args.cache_dir, today, args.symbol), len(runs)))
    if args.write:
        print(f"wrote {write_review(args.cache_dir, today)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
