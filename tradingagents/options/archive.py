"""Point-in-time archive of option chains, recorded daily and replayed without lookahead.

Free historical option chains do not exist, so the chains this repo fetches
are kept here to backtest a contract selector against later.

Layout: ``<cache_dir>/options_chains/<SYMBOL>/<NY fetch date>/<fetched_at>-<feed>-<digest>.json.gz``,
one gzip JSON file per snapshot holding ``{"meta": ..., "quotes": [...]}`` (JSON
rather than parquet because no parquet engine is installed). Every intraday
fetch is its own snapshot, so replay can pick the one that existed at the
decision time rather than the session's last.

* **Availability is our clock, not the source's.** A snapshot becomes visible
  to replay at ``fetched_at``, the UTC moment this process received it. The
  source's own ``as_of`` is kept verbatim but never used for selection: Cboe
  stamps it without a timezone, and a guessed zone could make data look older
  than it is, which is lookahead.
* **Duplicates.** A snapshot whose source, ``as_of`` and quotes all equal one
  already archived is not written again; the existing file is returned. A
  re-fetch of an unchanged payload therefore keeps its FIRST ``fetched_at``,
  which is the earliest moment it was known.
* **Stale.** Replay treats a snapshot as absent when the data is older than
  ``max_age`` at the decision time, measured from the earlier of ``fetched_at``
  and the close of the ``session`` the source says the quotes describe. It
  returns ``None`` rather than a flagged answer, so no caller can price a
  decision from a week-old chain by forgetting to read a flag.
* **Not fills.** Quotes are delayed (Cboe) or indicative (Alpaca) bid/ask.
  ``meta["executable"]`` is always ``False``: a replay may use them to choose a
  contract, never as the price a fill happened at.

Writes take an flock on ``<SYMBOL>/.lock`` for the duplicate check and write
through a unique temp file plus ``os.replace``, as ``record_iv`` does.

Run daily after the close (does NOT install itself anywhere)::

    # crontab: 30 16 * * 1-5  cd ~/Code/Finance/TradingAgents && .venv/bin/python -m tradingagents.options.archive --file ~/watchlist.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from tradingagents.dataflows.symbols import normalize_symbol
from tradingagents.dataflows.vendors.options import Quote

logger = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
SCHEMA_VERSION = 1
# Long enough to span a Friday-close snapshot through a Monday holiday.
DEFAULT_MAX_AGE = timedelta(days=4)
NOT_EXECUTABLE_NOTE = ("delayed/indicative bid-ask quotes, not executable prices: use them to choose a "
                       "contract, never as the price a fill happened at")
_STAMP = "%Y%m%dT%H%M%S%fZ"
_NAME = re.compile(r"^(?P<stamp>\d{8}T\d{12}Z)-(?P<feed>[a-z]+)-(?P<digest>[0-9a-f]{16})\.json\.gz$")


def _symbol_dir(cache_dir: str, symbol: str) -> Path:
    return Path(cache_dir) / "options_chains" / normalize_symbol(symbol).upper()


def _quote_row(q: Quote) -> dict:
    row = dataclasses.asdict(q)
    row["expiry"] = q.expiry.isoformat()
    return row


def _feed(meta: dict) -> str:
    if meta.get("feed"):
        return str(meta["feed"])
    return "indicative" if "alpaca" in str(meta.get("source", "")).lower() else "delayed"


def _utc(ts: datetime) -> datetime:
    """A tz-aware UTC time; a naive one is read as New York, the market's clock."""
    return (ts if ts.tzinfo else ts.replace(tzinfo=NY)).astimezone(timezone.utc)


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def record_chain_snapshot(cache_dir: str, symbol: str, meta: dict, quotes: list[Quote],
                          fetched_at: datetime | None = None) -> Path:
    """Archive one fetched chain; returns its file (the existing one for an exact duplicate).

    ``meta`` is what ``fetch_cboe``/``fetch_chain`` return: source, as_of,
    spot, iv30, session and optionally feed. ``fetched_at`` defaults to now.
    """
    fetched = _utc(fetched_at or datetime.now(timezone.utc))
    feed = _feed(meta)
    rows = sorted((_quote_row(q) for q in quotes), key=lambda r: r["symbol"])
    # Every stored field is in the digest, so a snapshot differing in any of
    # them (a moved spot on an unchanged chain) is kept rather than deduplicated.
    identity = [meta.get(k) for k in ("source", "as_of", "session", "spot", "iv30")] + [feed, rows]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]
    root = _symbol_dir(cache_dir, symbol)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = sorted(root.glob(f"*/*-{digest}.json.gz"))
        if existing:
            return existing[0]
        record = {
            "meta": {
                "schema": SCHEMA_VERSION,
                "symbol": normalize_symbol(symbol).upper(),
                "source": meta.get("source"),
                "feed": feed,
                "as_of": meta.get("as_of"),
                "session": meta.get("session"),
                "spot": meta.get("spot"),
                "iv30": meta.get("iv30"),
                "fetched_at": fetched.isoformat(),
                "available_at": fetched.isoformat(),
                "quote_count": len(rows),
                "executable": False,
                "note": NOT_EXECUTABLE_NOTE,
            },
            "quotes": rows,
        }
        day = root / fetched.astimezone(NY).strftime("%Y-%m-%d")
        day.mkdir(exist_ok=True)
        path = day / f"{fetched.strftime(_STAMP)}-{feed}-{digest}.json.gz"
        _write_atomic(path, gzip.compress(json.dumps(record, sort_keys=True).encode()))
    return path


def record_candidates(snapshot: Path, right: str, trade_date: str, contracts: list[str],
                      recorded_at: datetime | None = None) -> Path:
    """Note which contracts the fact sheet offered from ``snapshot`` for ``right`` on ``trade_date``.

    The first record stands: a later run on the same snapshot would move
    ``recorded_at`` forward and hide candidates a replay could already see.
    """
    path = snapshot.with_name(snapshot.name.replace(".json.gz", f".candidates-{right}.json"))
    body = {"right": right, "trade_date": trade_date, "contracts": list(contracts),
            "recorded_at": _utc(recorded_at or datetime.now(timezone.utc)).isoformat()}
    with open(snapshot.parent.parent / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            _write_atomic(path, json.dumps(body, sort_keys=True).encode())
    return path


def _read(path: Path, decision: datetime) -> tuple[dict, list[Quote]]:
    record = json.loads(gzip.decompress(path.read_bytes()))
    quotes = [Quote(**{**r, "expiry": date.fromisoformat(r["expiry"])}) for r in record["quotes"]]
    meta = record["meta"]
    meta["candidates"] = {}
    for side in path.parent.glob(path.name.replace(".json.gz", ".candidates-*.json")):
        try:
            c = json.loads(side.read_text())
            # Candidates are chosen after the fetch: ones picked after the
            # decision time were not known at it.
            if datetime.fromisoformat(c["recorded_at"]) <= decision:
                meta["candidates"][c["right"]] = c
        except (OSError, ValueError, KeyError):
            continue
    return meta, quotes


def _decision_time(as_of: datetime | date | str) -> datetime:
    """A decision time in UTC. A bare date means the START of that New York day: nothing from it is known yet."""
    if isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of) if len(as_of) > 10 else date.fromisoformat(as_of)
    if not isinstance(as_of, datetime):
        as_of = datetime(as_of.year, as_of.month, as_of.day, tzinfo=NY)
    return _utc(as_of)


def _data_time(meta: dict, fetched: datetime) -> datetime:
    """How old the quotes are: the session's close when the source named one, else when we fetched them."""
    try:
        close = datetime.fromisoformat(f"{meta['session']}T16:00").replace(tzinfo=NY).astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError):
        return fetched
    return min(fetched, close)


def list_snapshots(cache_dir: str, symbol: str) -> list[tuple[datetime, Path]]:
    """Every archived snapshot for ``symbol`` as (fetched_at UTC, file), oldest first."""
    out = []
    for path in _symbol_dir(cache_dir, symbol).glob("*/*.json.gz"):
        m = _NAME.match(path.name)
        if m:
            out.append((datetime.strptime(m["stamp"], _STAMP).replace(tzinfo=timezone.utc), path))
    return sorted(out)


def load_chain_snapshot(cache_dir: str, symbol: str, as_of: datetime | date | str,
                        max_age: timedelta = DEFAULT_MAX_AGE,
                        feed: str | None = None) -> tuple[dict, list[Quote]] | None:
    """The latest non-stale snapshot fetched at or before ``as_of``; None when there is none.

    Never returns a snapshot fetched after ``as_of``, whatever its source
    timestamp says. ``feed`` ("delayed" / "indicative") restricts the pick.
    ``meta["age_seconds"]`` reports how old the data was at the decision time.
    """
    decision = _decision_time(as_of)
    for fetched, path in reversed(list_snapshots(cache_dir, symbol)):
        if fetched > decision or (feed and _NAME.match(path.name)["feed"] != feed):
            continue
        meta, quotes = _read(path, decision)
        age = decision - _data_time(meta, fetched)
        if age > max_age:
            # A later fetch can carry an older session than an earlier one,
            # so a stale snapshot does not end the search.
            logger.info("options archive: %s snapshot fetched %s is %s old at %s; skipping",
                        symbol, fetched.isoformat(), age, decision.isoformat())
            continue
        meta["age_seconds"] = age.total_seconds()
        return meta, quotes
    return None


# ----------------------------------------------------------------- CLI

def _watchlist(symbols: list[str], files: list[str]) -> list[str]:
    out = list(symbols)
    for name in files:
        for line in Path(name).expanduser().read_text().splitlines():
            out += [s for s in re.split(r"[\s,]+", line.split("#", 1)[0]) if s]
    seen, ordered = set(), []
    for s in (normalize_symbol(s).upper() for s in out):
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def main(argv: list[str] | None = None) -> int:
    from tradingagents.dataflows.vendors.options import fetch_chain
    from tradingagents.default_config import DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description="Archive today's option chain for each symbol in a watchlist.")
    parser.add_argument("symbols", nargs="*", help="tickers to snapshot")
    parser.add_argument("--file", action="append", default=[], help="watchlist file: tickers separated by "
                        "whitespace/commas, '#' comments (repeatable)")
    parser.add_argument("--cache-dir", default=DEFAULT_CONFIG["data_cache_dir"])
    args = parser.parse_args(argv)
    symbols = _watchlist(args.symbols, args.file)
    if not symbols:
        parser.error("no symbols given")
    ok = 0
    for symbol in symbols:
        try:
            meta, quotes = fetch_chain(symbol)
            path = record_chain_snapshot(args.cache_dir, symbol, meta, quotes)
        except Exception as exc:  # noqa: BLE001 — one symbol's failure must not stop the rest
            print(f"{symbol}: FAILED {type(exc).__name__}: {exc}")
            continue
        ok += 1
        print(f"{symbol}: {len(quotes)} quotes, {meta.get('source')} as of {meta.get('as_of')} -> {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
