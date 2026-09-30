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
            if row.get("pick") and date.fromisoformat(row["exit_date"]) >= cutoff and row["symbol"] not in seen:
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
    contract: str
    right: str
    status: str                   # "settled" | "open" | "not scorable"
    entry_ask: float
    target: float
    exit_date: str
    exit_bid: float | None = None
    ret: float | None = None
    shares_ret: float | None = None
    stock_at_exit: float | None = None
    predicted_at_target: float | None = None
    reason: str = ""
    strike: float | None = None
    expiry: str | None = None
    breakeven: float | None = None
    spot_at_pick: float | None = None
    as_of: str | None = None            # when the quotes behind the pick were made, per the source


def _quote_bid(quotes, symbol: str) -> float | None:
    q = next((q for q in quotes if q.symbol.replace(" ", "") == symbol), None)
    return q.bid if q is not None and q.bid > 0 else None


def score(row: dict, cache_dir: str, today: date) -> Outcome | None:
    """One logged run against the archived chains; None for a run that picked nothing."""
    from tradingagents.options.archive import load_chain_snapshot

    p = row.get("pick")
    if not p:
        return None
    exit_d = date.fromisoformat(row["exit_date"])
    out = Outcome(row["logged_at"][:10], row["symbol"], p["symbol"], row["right"], "not scorable", p["ask"],
                  row["target"], row["exit_date"], predicted_at_target=p["roi_target"], strike=p.get("strike"),
                  expiry=p.get("expiry"), breakeven=p.get("breakeven"), spot_at_pick=row.get("spot"),
                  as_of=row.get("as_of"))
    settled = today > exit_d
    as_of = exit_d + timedelta(days=1) if settled else datetime.now(timezone.utc)   # a bare date = start of that day
    snap = load_chain_snapshot(cache_dir, row["symbol"], as_of, max_age=timedelta(days=MAX_EXIT_STALENESS_DAYS))
    if snap is None:
        out.reason = (f"no chain archived within {MAX_EXIT_STALENESS_DAYS} days of "
                      f"{'the exit date' if settled else 'now'}; is {row['symbol']} on the nightly snapshot list?")
        return out
    meta, quotes = snap
    bid = _quote_bid(quotes, p["symbol"])
    if bid is None:
        out.reason = "the contract has no bid in the archived chain (a worthless or missing quote is not filled in)"
        return out
    out.exit_bid, out.ret = bid, bid / p["ask"] - 1
    out.stock_at_exit = meta.get("spot")
    if row["right"] == "C" and out.stock_at_exit and row.get("spot"):
        out.shares_ret = out.stock_at_exit / row["spot"] - 1
    out.status = "settled" if settled else "open"
    out.reason = "" if settled else f"marked to the chain of {meta.get('session') or meta.get('as_of')}, not a result"
    return out


def review(cache_dir: str, today: date, symbol: str | None = None) -> list[Outcome]:
    rows = [r for r in load_picks(cache_dir) if symbol is None or r.get("symbol") == symbol.upper()]
    return [o for o in (score(r, cache_dir, today) for r in rows) if o is not None]


def summary(outcomes: list[Outcome]) -> dict:
    """Counts over the settled picks only: an open pick is a mark, not a result."""
    done = [o for o in outcomes if o.status == "settled" and o.ret is not None]
    paired = [o for o in done if o.shares_ret is not None]
    return {"picks": len(outcomes), "settled": len(done), "open": sum(o.status == "open" for o in outcomes),
            "not_scorable": sum(o.status == "not scorable" for o in outcomes),
            "mean_return": sum(o.ret for o in done) / len(done) if done else None,
            "median_return": statistics.median(o.ret for o in done) if done else None,
            "made_money": sum(o.ret > 0 for o in done), "paired": len(paired),
            "beat_shares": sum(o.ret > o.shares_ret for o in paired)}


def review_payload(cache_dir: str, today: date, now: datetime | None = None) -> dict:
    """The review as data, for another program to display. Everything in it was computed here."""
    rows = load_picks(cache_dir)
    outcomes = [o for o in (score(r, cache_dir, today) for r in rows) if o is not None]
    return {"schema": SCHEMA, "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
            "as_of_date": today.isoformat(), "runs": len(rows), "runs_without_pick": len(rows) - len(outcomes),
            "summary": summary(outcomes), "picks": [asdict(o) for o in outcomes]}


def write_review(cache_dir: str, today: date, now: datetime | None = None) -> Path:
    """Write ``review.json`` beside the log, atomically, so a reader never sees half of it."""
    root = pick_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "review.json"
    _atomic_write(path, json.dumps(review_payload(cache_dir, today, now), indent=1, sort_keys=True).encode())
    return path


def render_review(outcomes: list[Outcome], total_runs: int) -> str:
    if not outcomes:
        return f"{total_runs} runs logged, none with a pick to review yet."
    lines = [f"{'logged':11}{'contract':22}{'status':13}{'paid':>7}{'exit bid':>9}{'return':>8}{'shares':>8}"
             f"{'model@target':>13}  note"]
    for o in outcomes:
        lines.append(f"{o.logged:11}{o.contract:22}{o.status:13}{o.entry_ask:7.2f}"
                     f"{(f'{o.exit_bid:.2f}' if o.exit_bid is not None else '-'):>9}"
                     f"{(f'{o.ret:+.0%}' if o.ret is not None else '-'):>8}"
                     f"{(f'{o.shares_ret:+.0%}' if o.shares_ret is not None else '-'):>8}"
                     f"{(f'{o.predicted_at_target:+.0%}' if o.predicted_at_target is not None else '-'):>13}  {o.reason}")
    done = [o for o in outcomes if o.status == "settled" and o.ret is not None]
    lines.append("")
    if not done:
        lines.append("Nothing has reached its exit date yet. Open picks are marks, not results.")
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


def main(argv: list[str] | None = None) -> int:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.dataflows.vendors.options import ny_today

    ap = argparse.ArgumentParser(description="Review the logged option picks.")
    ap.add_argument("command", choices=["review"])
    ap.add_argument("--symbol")
    ap.add_argument("--cache-dir", default=DEFAULT_CONFIG.get("data_cache_dir"))
    ap.add_argument("--write", action="store_true", help="also write option_picks/review.json for other programs to read")
    args = ap.parse_args(argv)
    today = date.fromisoformat(ny_today())
    runs = load_picks(args.cache_dir)
    print(render_review(review(args.cache_dir, today, args.symbol), len(runs)))
    if args.write:
        print(f"wrote {write_review(args.cache_dir, today)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
