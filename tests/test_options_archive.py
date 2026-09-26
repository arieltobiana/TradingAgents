"""The point-in-time option-chain archive: what is kept, and what replay may see."""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from tradingagents.dataflows.vendors import options
from tradingagents.options import archive
from tradingagents.options.archive import list_snapshots, load_chain_snapshot, record_chain_snapshot

UTC = timezone.utc
META = {"source": "Cboe delayed quotes", "as_of": "2026-09-24 16:15:02", "spot": 46.0, "iv30": 76.0,
        "session": "2026-09-24", "feed": "delayed"}


def _quotes(bid=1.10):
    e = date(2026, 10, 16)
    return [
        options.Quote("IREN261016C00046000", "C", 46.0, e, bid, 1.20, 0.81, 0.52, 0.04, -0.07, 0.05, 1200.0, 340.0),
        options.Quote("IREN261016P00046000", "P", 46.0, e, 1.00, 1.08, 0.79, -0.48, 0.04, -0.06, 0.05, None, None),
    ]


def _at(day, hh, mm=0):
    return datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}").replace(tzinfo=archive.NY)


@pytest.mark.unit
def test_every_field_round_trips_and_the_quotes_are_labelled_not_executable(tmp_path):
    fetched = _at("2026-09-24", 16, 30)
    record_chain_snapshot(str(tmp_path), "iren", META, _quotes(), fetched_at=fetched)
    meta, quotes = load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 17))
    assert sorted(quotes, key=lambda q: q.symbol) == sorted(_quotes(), key=lambda q: q.symbol)
    for key in ("source", "as_of", "spot", "iv30", "session", "feed"):
        assert meta[key] == META[key]
    assert meta["symbol"] == "IREN" and meta["quote_count"] == 2
    assert datetime.fromisoformat(meta["fetched_at"]) == fetched
    assert meta["executable"] is False and "not executable" in meta["note"]
    assert meta["age_seconds"] == 60 * 60  # measured from the 16:00 session close, not the 16:30 fetch


@pytest.mark.unit
def test_an_exact_duplicate_is_kept_once_with_its_first_fetch_time(tmp_path):
    # All during the session: after the close, a later fetch is skipped for another reason.
    first = record_chain_snapshot(str(tmp_path), "IREN", META, _quotes(), fetched_at=_at("2026-09-24", 14, 30))
    again = record_chain_snapshot(str(tmp_path), "IREN", META, list(reversed(_quotes())),
                                  fetched_at=_at("2026-09-24", 15, 0))
    assert again == first and len(list_snapshots(str(tmp_path), "IREN")) == 1
    # A changed quote is a new snapshot, even under the same source timestamp.
    record_chain_snapshot(str(tmp_path), "IREN", META, _quotes(bid=1.12), fetched_at=_at("2026-09-24", 15, 0))
    assert len(list_snapshots(str(tmp_path), "IREN")) == 2
    # So is a moved underlying on an unchanged chain: dedup must not keep the old spot.
    record_chain_snapshot(str(tmp_path), "IREN", {**META, "spot": 46.5}, _quotes(bid=1.12),
                          fetched_at=_at("2026-09-24", 15, 30))
    assert len(list_snapshots(str(tmp_path), "IREN")) == 3
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 15, 45))[0]["spot"] == 46.5


@pytest.mark.unit
def test_a_closed_session_is_saved_once_however_often_it_is_fetched(tmp_path):
    # Intraday fetches of a live session are all kept.
    record_chain_snapshot(str(tmp_path), "IREN", {**META, "as_of": "a"}, _quotes(bid=1.00),
                          fetched_at=_at("2026-09-24", 11))
    record_chain_snapshot(str(tmp_path), "IREN", {**META, "as_of": "b"}, _quotes(bid=1.05),
                          fetched_at=_at("2026-09-24", 15, 45))
    final = record_chain_snapshot(str(tmp_path), "IREN", {**META, "as_of": "c"}, _quotes(bid=1.10),
                                  fetched_at=_at("2026-09-24", 16, 30))
    assert len(list_snapshots(str(tmp_path), "IREN")) == 3
    # Later fetches of the same, closed session (a re-stamped payload overnight or
    # on Saturday) return the final snapshot and write nothing.
    for when in (_at("2026-09-24", 23, 30), _at("2026-09-26", 7)):
        again = record_chain_snapshot(str(tmp_path), "IREN", {**META, "as_of": "d"}, _quotes(bid=1.10),
                                      fetched_at=when)
        assert again == final
    assert len(list_snapshots(str(tmp_path), "IREN")) == 3
    # The next session, and a feed that names no session, are still saved.
    record_chain_snapshot(str(tmp_path), "IREN", {**META, "session": "2026-09-25"}, _quotes(),
                          fetched_at=_at("2026-09-25", 16, 30))
    record_chain_snapshot(str(tmp_path), "IREN", {"source": "Alpaca indicative feed", "as_of": "x",
                                                   "session": None}, _quotes(), fetched_at=_at("2026-09-26", 7))
    record_chain_snapshot(str(tmp_path), "IREN", {"source": "Alpaca indicative feed", "as_of": "y",
                                                   "session": None}, _quotes(bid=1.2), fetched_at=_at("2026-09-26", 8))
    assert len(list_snapshots(str(tmp_path), "IREN")) == 6


@pytest.mark.unit
def test_concurrent_records_lose_nothing(tmp_path):
    def write(i):
        record_chain_snapshot(str(tmp_path), "IREN", {**META, "as_of": f"2026-09-24 15:{i:02d}:00"},
                              _quotes(bid=1.0 + i / 100), fetched_at=_at("2026-09-24", 15, i))

    threads = [threading.Thread(target=write, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snaps = list_snapshots(str(tmp_path), "IREN")
    assert len(snaps) == 12
    assert not list((tmp_path / "options_chains").rglob("*.tmp"))
    for _, path in snaps:
        archive._read(path, datetime.now(UTC))  # every file is whole


@pytest.mark.unit
def test_replay_never_returns_a_snapshot_fetched_after_the_decision(tmp_path):
    record_chain_snapshot(str(tmp_path), "IREN", META, _quotes(bid=1.10), fetched_at=_at("2026-09-24", 11))
    # Its source stamp claims an earlier time, but we only had it at 15:00.
    record_chain_snapshot(str(tmp_path), "IREN", {**META, "as_of": "2026-09-24 09:31:00"}, _quotes(bid=1.15),
                          fetched_at=_at("2026-09-24", 15))
    meta, quotes = load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 14, 59))
    assert quotes[0].bid == 1.10 and meta["as_of"] == META["as_of"]
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 15))[1][0].bid == 1.15
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 10, 59)) is None
    # A bare date is the start of that New York day; a naive time is New York time.
    assert load_chain_snapshot(str(tmp_path), "IREN", "2026-09-24") is None
    assert load_chain_snapshot(str(tmp_path), "IREN", "2026-09-25")[1][0].bid == 1.15
    assert load_chain_snapshot(str(tmp_path), "IREN", "2026-09-24T14:00")[1][0].bid == 1.10
    # A UTC decision time is compared on the same clock (15:00 NY is 19:00 UTC).
    assert load_chain_snapshot(str(tmp_path), "IREN", datetime(2026, 9, 24, 18, 59, tzinfo=UTC))[1][0].bid == 1.10


@pytest.mark.unit
def test_a_stale_snapshot_is_absent(tmp_path):
    record_chain_snapshot(str(tmp_path), "IREN", META, _quotes(), fetched_at=_at("2026-09-24", 16, 30))
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-28", 9, 45)) is not None  # over a weekend
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-29", 9, 45)) is None
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-29", 9, 45), max_age=timedelta(days=7))


@pytest.mark.unit
def test_an_old_session_is_stale_even_when_fetched_recently(tmp_path):
    # The source served a week-old payload today: its age is the session's, not the fetch's.
    record_chain_snapshot(str(tmp_path), "IREN", {**META, "session": "2026-09-17"}, _quotes(),
                          fetched_at=_at("2026-09-24", 16, 30))
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 17)) is None


@pytest.mark.unit
def test_a_stale_later_fetch_does_not_hide_a_fresh_earlier_one(tmp_path):
    record_chain_snapshot(str(tmp_path), "IREN", META, _quotes(bid=1.10), fetched_at=_at("2026-09-24", 16, 30))
    record_chain_snapshot(str(tmp_path), "IREN", {**META, "session": "2026-09-17"}, _quotes(bid=1.20),
                          fetched_at=_at("2026-09-24", 16, 45))
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 17))[1][0].bid == 1.10


@pytest.mark.unit
def test_candidates_chosen_after_the_decision_are_not_replayed(tmp_path):
    snap = record_chain_snapshot(str(tmp_path), "IREN", META, _quotes(), fetched_at=_at("2026-09-24", 16, 30))
    archive.record_candidates(snap, "C", "2026-09-24", ["IREN261016C00046000"],
                              recorded_at=_at("2026-09-24", 16, 40))
    assert load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 16, 35))[0]["candidates"] == {}
    seen = load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 16, 40))[0]["candidates"]
    assert seen["C"]["contracts"] == ["IREN261016C00046000"]
    # A later run on the same snapshot does not rewrite what was already known.
    archive.record_candidates(snap, "C", "2026-09-24", ["IREN261016C00047000"],
                              recorded_at=_at("2026-09-24", 17))
    seen = load_chain_snapshot(str(tmp_path), "IREN", _at("2026-09-24", 16, 45))[0]["candidates"]
    assert seen["C"]["contracts"] == ["IREN261016C00046000"]


@pytest.mark.unit
def test_feed_filter_keeps_indicative_quotes_apart(tmp_path):
    record_chain_snapshot(str(tmp_path), "IREN", META, _quotes(), fetched_at=_at("2026-09-24", 16))
    record_chain_snapshot(str(tmp_path), "IREN", {"source": "Alpaca indicative feed", "as_of": "x", "spot": None,
                                                   "iv30": None, "session": None}, _quotes(bid=1.13),
                          fetched_at=_at("2026-09-24", 16, 30))
    when = _at("2026-09-24", 17)
    assert load_chain_snapshot(str(tmp_path), "IREN", when)[0]["feed"] == "indicative"
    assert load_chain_snapshot(str(tmp_path), "IREN", when, feed="delayed")[0]["feed"] == "delayed"


@pytest.fixture
def market(monkeypatch):
    monkeypatch.setattr(options, "ny_today", lambda: "2026-09-25")
    monkeypatch.setattr(options, "earnings_events", lambda s: [])
    monkeypatch.setattr(options, "earnings_reactions", lambda s, d, ev: [])
    monkeypatch.setattr(options, "fetch_alpaca", lambda s, syms=None: [])


def _sheet(cache_dir):
    from tradingagents.dataflows.vendors.yahoo.facts import _Sheet

    sheet = _Sheet()
    sheet.add("Latest close", 46.0, "price", "bars")
    options.option_facts(sheet, "IREN", "2026-09-25", "C", cache_dir)
    return sheet


@pytest.mark.unit
def test_option_facts_archives_the_chain_and_its_candidates(market, monkeypatch, tmp_path):
    from tests.test_options_facts import _chain

    monkeypatch.setattr(options, "fetch_cboe", lambda s: (dict(META), _chain()))
    sheet = _sheet(str(tmp_path))
    offered = [f.label.split()[1] for f in sheet.facts if f.label.startswith("Candidate ")]
    assert offered
    meta, quotes = load_chain_snapshot(str(tmp_path), "IREN", datetime.now(UTC), max_age=timedelta.max)
    assert len(quotes) == len(_chain()) and meta["feed"] == "delayed"
    assert meta["candidates"]["C"]["contracts"] == offered
    assert meta["candidates"]["C"]["trade_date"] == "2026-09-25"


@pytest.mark.unit
def test_an_archive_failure_never_costs_the_fact_sheet(market, monkeypatch, tmp_path):
    from tests.test_options_facts import _chain

    monkeypatch.setattr(options, "fetch_cboe", lambda s: (dict(META), _chain()))

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(archive, "record_chain_snapshot", broken)
    sheet = _sheet(str(tmp_path))
    assert any(f.label.startswith("Candidate ") for f in sheet.facts)
    assert not (tmp_path / "options_chains").exists()


@pytest.mark.unit
def test_cli_snapshots_a_watchlist_and_fails_only_when_everything_failed(monkeypatch, tmp_path, capsys):
    def fetch(symbol):
        if symbol == "BAD":
            raise RuntimeError("no chain")
        return dict(META), _quotes()

    monkeypatch.setattr(options, "fetch_chain", fetch)
    watch = tmp_path / "watch.txt"
    watch.write_text("iren, bad  # comment\nIREN\n")
    assert archive.main(["--cache-dir", str(tmp_path), "--file", str(watch)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[0].startswith("IREN: 2 quotes") and out[1].startswith("BAD: FAILED")
    assert len(list_snapshots(str(tmp_path), "IREN")) == 1
    assert archive.main(["--cache-dir", str(tmp_path), "BAD"]) == 1
