"""The log of option picks: what is kept, how it rotates, and how a pick is scored later."""

from __future__ import annotations

import gzip
import json
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from tradingagents.dataflows.vendors.options import Quote
from tradingagents.options import picklog
from tradingagents.options.archive import record_chain_snapshot

UTC = timezone.utc
CONTRACT = "IREN270115C00030000"
PICK = {"symbol": CONTRACT, "ask": 10.0, "roi_target": 0.4}


def _entry(symbol="IREN", exit_date="2026-12-31", pick=PICK, right="C", spot=40.0, target=50.0):
    return {"symbol": symbol, "right": right, "spot": spot, "target": target, "exit_date": exit_date, "pick": pick}


def _quote(bid, symbol=CONTRACT):
    return Quote(symbol, "C", 30.0, date(2027, 1, 15), bid, bid * 1.03, 0.8, 0.8, None, None, None, 500.0, 10.0)


def _snap(cache, when, spot, bid, session=None, symbol="IREN"):
    meta = {"source": "Cboe delayed quotes", "as_of": when.isoformat(), "spot": spot, "iv30": 70.0,
            "session": session or when.date().isoformat(), "feed": "delayed"}
    record_chain_snapshot(str(cache), symbol, meta, [_quote(bid)], fetched_at=when)


@pytest.mark.unit
def test_every_run_is_appended_to_the_months_file_and_read_back_in_order(tmp_path):
    a = datetime(2026, 9, 30, 12, tzinfo=UTC)
    picklog.record_pick(str(tmp_path), _entry("MU"), now=a)
    picklog.record_pick(str(tmp_path), _entry("IREN"), now=a + timedelta(hours=1))
    assert [p.name for p in (tmp_path / "option_picks").glob("picks-*")] == ["picks-2026-09.jsonl"]
    rows = picklog.load_picks(str(tmp_path))
    assert [r["symbol"] for r in rows] == ["MU", "IREN"] and rows[0]["schema"] == picklog.SCHEMA
    assert rows[0]["logged_at"] == a.isoformat()


@pytest.mark.unit
def test_concurrent_writers_lose_no_line(tmp_path):
    now = datetime(2026, 9, 30, tzinfo=UTC)
    threads = [threading.Thread(target=picklog.record_pick, args=(str(tmp_path), _entry(f"S{i}")),
                                kwargs={"now": now + timedelta(seconds=i)}) for i in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(r["symbol"] for r in picklog.load_picks(str(tmp_path))) == sorted(f"S{i}" for i in range(16))


@pytest.mark.unit
def test_old_months_are_gzipped_but_never_deleted_and_still_read(tmp_path):
    picklog.record_pick(str(tmp_path), _entry("OLD"), now=datetime(2026, 1, 15, tzinfo=UTC))
    picklog.record_pick(str(tmp_path), _entry("MID"), now=datetime(2026, 6, 15, tzinfo=UTC))   # rotates January
    names = sorted(p.name for p in (tmp_path / "option_picks").glob("picks-*"))
    assert names == ["picks-2026-01.jsonl.gz", "picks-2026-06.jsonl"]
    picklog.record_pick(str(tmp_path), _entry("NEW"), now=datetime(2026, 9, 30, tzinfo=UTC))  # rotates June
    assert sorted(p.name for p in (tmp_path / "option_picks").glob("picks-*")) == \
        ["picks-2026-01.jsonl.gz", "picks-2026-06.jsonl.gz", "picks-2026-09.jsonl"]
    assert [r["symbol"] for r in picklog.load_picks(str(tmp_path))] == ["OLD", "MID", "NEW"]
    assert not list((tmp_path / "option_picks").glob("*.tmp"))


@pytest.mark.unit
def test_a_recent_month_is_not_rotated_yet(tmp_path):
    picklog.record_pick(str(tmp_path), _entry(), now=datetime(2026, 8, 20, tzinfo=UTC))
    picklog.record_pick(str(tmp_path), _entry(), now=datetime(2026, 9, 30, tzinfo=UTC))   # August ended 30 days ago
    assert (tmp_path / "option_picks" / "picks-2026-08.jsonl").exists()


@pytest.mark.unit
def test_a_torn_line_or_corrupt_file_never_hides_the_rest(tmp_path):
    picklog.record_pick(str(tmp_path), _entry("GOOD"), now=datetime(2026, 9, 30, tzinfo=UTC))
    d = tmp_path / "option_picks"
    with open(d / "picks-2026-09.jsonl", "a") as f:
        f.write('{"logged_at": "2026-09-30T13:00:00+00:00", "symb\n')
    (d / "picks-2026-01.jsonl.gz").write_bytes(b"not gzip")
    assert [r["symbol"] for r in picklog.load_picks(str(tmp_path))] == ["GOOD"]


@pytest.mark.unit
def test_a_pick_is_scored_against_the_chain_archived_at_its_exit(tmp_path):
    _snap(tmp_path, datetime(2026, 12, 31, 21, tzinfo=UTC), spot=52.0, bid=14.0)            # exit-day chain
    _snap(tmp_path, datetime(2027, 1, 4, 21, tzinfo=UTC), spot=60.0, bid=30.0)              # after: must not be used
    row = {"logged_at": "2026-09-30T12:00:00+00:00", **_entry()}
    out = picklog.score(row, str(tmp_path), today=date(2027, 1, 5))
    assert out.status == "settled" and out.exit_bid == 14.0
    assert out.ret == pytest.approx(0.4) and out.shares_ret == pytest.approx(0.30) and out.stock_at_exit == 52.0


@pytest.mark.unit
def test_no_chain_near_the_exit_is_not_scorable_not_guessed(tmp_path):
    _snap(tmp_path, datetime(2026, 12, 10, 21, tzinfo=UTC), spot=45.0, bid=12.0)            # three weeks before
    out = picklog.score({"logged_at": "2026-09-30T12:00:00+00:00", **_entry()}, str(tmp_path), today=date(2027, 1, 5))
    assert out.status == "not scorable" and out.ret is None and "nightly snapshot" in out.reason


@pytest.mark.unit
def test_a_contract_with_no_bid_is_not_scorable_not_a_zero(tmp_path):
    _snap(tmp_path, datetime(2026, 12, 31, 21, tzinfo=UTC), spot=30.0, bid=0.0)
    out = picklog.score({"logged_at": "2026-09-30T12:00:00+00:00", **_entry()}, str(tmp_path), today=date(2027, 1, 5))
    assert out.status == "not scorable" and "no bid" in out.reason


@pytest.mark.unit
def test_before_the_exit_date_a_pick_is_a_labelled_mark_and_a_run_with_no_pick_is_skipped(tmp_path):
    _snap(tmp_path, datetime.now(UTC) - timedelta(hours=2), spot=41.0, bid=9.5)
    row = {"logged_at": datetime.now(UTC).isoformat(), **_entry(exit_date=(date.today() + timedelta(days=60)).isoformat())}
    out = picklog.score(row, str(tmp_path), today=date.today())
    assert out.status == "open" and out.ret == pytest.approx(-0.05) and "not a result" in out.reason
    nothing = picklog.score({**row, "pick": None}, str(tmp_path), today=date.today())
    assert nothing.status == "no contract" and nothing.contract is None and nothing.stock_at_exit == 41.0


@pytest.mark.unit
def test_the_review_summary_counts_only_settled_picks_and_warns_on_a_small_sample(tmp_path):
    _snap(tmp_path, datetime(2026, 12, 31, 21, tzinfo=UTC), spot=52.0, bid=14.0)
    picklog.record_pick(str(tmp_path), _entry(), now=datetime(2026, 9, 30, tzinfo=UTC))
    text = picklog.render_review(picklog.review(str(tmp_path), date(2027, 1, 5)), 1)
    assert "Settled: 1 of 1" in text and "beat the shares in 1 of 1" in text and "first look" in text
    assert "0 runs" not in text
    assert "none to review yet" in picklog.render_review([], 3)


@pytest.mark.unit
def test_tickers_with_a_pick_still_to_score_are_named_for_the_nightly_snapshot(tmp_path):
    now = datetime(2026, 9, 30, tzinfo=UTC)
    picklog.record_pick(str(tmp_path), _entry("IREN", exit_date="2026-12-31"), now=now)
    picklog.record_pick(str(tmp_path), _entry("OLD", exit_date="2026-08-01"), now=now)
    picklog.record_pick(str(tmp_path), _entry("NONE", pick=None), now=now)
    # A request that named no contract is still a view to compare with the stock, so its ticker is archived too.
    assert picklog.open_pick_symbols(str(tmp_path), date(2026, 10, 1)) == ["IREN", "NONE"]
    assert picklog.open_pick_symbols(str(tmp_path), date(2026, 12, 31)) == ["IREN", "NONE"]   # the exit day's chain is still wanted
    assert picklog.open_pick_symbols(str(tmp_path), date(2027, 1, 1)) == []          # a later chain cannot be used to score it


@pytest.mark.unit
def test_a_reader_waits_for_a_rotation_in_progress_instead_of_seeing_a_month_twice(tmp_path):
    import fcntl

    picklog.record_pick(str(tmp_path), _entry("ONE"), now=datetime(2026, 1, 15, tzinfo=UTC))
    root = tmp_path / "option_picks"
    got, started = [], threading.Event()

    def read():
        started.set()
        got.append(picklog.load_picks(str(tmp_path)))

    with open(root / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)           # a rotation holds this while it swaps plain for gzip
        t = threading.Thread(target=read)
        t.start()
        started.wait(2)
        t.join(0.3)
        assert t.is_alive() and not got            # blocked, not reading a half-rotated directory
    t.join(3)
    assert [r["symbol"] for r in got[0]] == ["ONE"]


@pytest.mark.unit
def test_the_review_is_written_as_json_another_program_can_read(tmp_path):
    _snap(tmp_path, datetime(2026, 12, 31, 21, tzinfo=UTC), spot=52.0, bid=14.0)
    picklog.record_pick(str(tmp_path), {**_entry(), "spot": 40.0, "as_of": "2026-09-30 20:42:40",
                                        "pick": {**PICK, "strike": 30.0, "expiry": "2027-01-15", "breakeven": 44.0}},
                        now=datetime(2026, 9, 30, tzinfo=UTC))
    picklog.record_pick(str(tmp_path), _entry("NOPICK", pick=None), now=datetime(2026, 9, 30, 1, tzinfo=UTC))
    path = picklog.write_review(str(tmp_path), date(2027, 1, 5), now=datetime(2027, 1, 5, tzinfo=UTC))
    data = json.loads(path.read_text())
    assert path.name == "review.json" and data["schema"] == picklog.SCHEMA and data["runs"] == 2
    assert data["runs_without_pick"] == 1 and data["as_of_date"] == "2027-01-05"
    pick, nothing = data["picks"]
    assert nothing["status"] == "no contract" and nothing["contract"] is None and nothing["symbol"] == "NOPICK"
    assert pick["status"] == "settled" and pick["ret"] == pytest.approx(0.4) and pick["breakeven"] == 44.0
    assert pick["strike"] == 30.0 and pick["spot_at_pick"] == 40.0 and pick["as_of"] == "2026-09-30 20:42:40"
    assert data["summary"]["settled"] == 1 and data["summary"]["beat_shares"] == 1
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.unit
def test_an_empty_log_still_writes_a_review_a_reader_can_tell_from_no_review(tmp_path):
    data = json.loads(picklog.write_review(str(tmp_path), date(2027, 1, 5)).read_text())
    assert data["runs"] == 0 and data["picks"] == [] and data["summary"]["settled"] == 0


def _asked(spot=40.0, target=50.0, right="C", pick=PICK, when="2026-09-30T12:00:00+00:00", exit_date="2026-12-31"):
    return {"logged_at": when, **_entry(spot=spot, target=target, right=right, pick=pick, exit_date=exit_date)}


@pytest.mark.unit
def test_a_request_is_compared_then_and_now_in_dollars_and_in_progress_to_the_target(tmp_path):
    _snap(tmp_path, datetime(2026, 10, 15, 21, tzinfo=UTC), spot=45.0, bid=12.0)
    out = picklog.score(_asked(), str(tmp_path), today=date(2026, 10, 16))
    assert out.status == "open" and out.logged_at == "2026-09-30T12:00:00+00:00" and out.mark_at.startswith("2026-10-15T21")
    assert out.pnl_per_contract == pytest.approx(200.0)              # (12.00 - 10.00) x 100: one contract, ask in, bid out
    assert out.shares_pnl_per_100 == pytest.approx(500.0)             # (45 - 40) x 100 shares
    assert out.target_progress == pytest.approx(0.5)                  # half way from 40 to 50
    assert out.view == "toward" and out.target_touched is False and out.days_left == 76


@pytest.mark.unit
def test_the_target_counts_as_reached_when_any_saved_price_got_there_even_if_it_fell_back(tmp_path):
    _snap(tmp_path, datetime(2026, 10, 10, 21, tzinfo=UTC), spot=51.0, bid=20.0)       # touched the target
    _snap(tmp_path, datetime(2026, 10, 20, 21, tzinfo=UTC), spot=44.0, bid=11.0)       # then fell back
    out = picklog.score(_asked(), str(tmp_path), today=date(2026, 10, 21))
    assert out.target_touched is True and out.target_touched_on == "2026-10-10" and out.view == "reached"
    assert out.target_progress == pytest.approx(0.4)                                    # but it is only 40% of the way now


@pytest.mark.unit
def test_a_snapshot_from_before_the_request_cannot_count_as_reaching_the_target(tmp_path):
    _snap(tmp_path, datetime(2026, 9, 20, 21, tzinfo=UTC), spot=55.0, bid=22.0)         # before the request
    _snap(tmp_path, datetime(2026, 10, 2, 21, tzinfo=UTC), spot=41.0, bid=10.5)
    out = picklog.score(_asked(), str(tmp_path), today=date(2026, 10, 3))
    assert out.target_touched is False and out.view == "toward" and out.target_progress == pytest.approx(0.1)


@pytest.mark.unit
def test_a_put_view_reaches_its_target_by_falling_and_has_no_share_comparison(tmp_path):
    put = {**PICK, "symbol": "IREN270115P00045000"}
    snap_quote = Quote(put["symbol"], "P", 45.0, date(2027, 1, 15), 9.0, 9.4, 0.8, -0.5, None, None, None, 500.0, 10.0)
    meta = {"source": "Cboe delayed quotes", "as_of": "x", "spot": 34.0, "iv30": 70.0, "session": "2026-10-10", "feed": "delayed"}
    record_chain_snapshot(str(tmp_path), "IREN", meta, [snap_quote], fetched_at=datetime(2026, 10, 10, 21, tzinfo=UTC))
    out = picklog.score(_asked(spot=40.0, target=35.0, right="P", pick=put), str(tmp_path), today=date(2026, 10, 11))
    assert out.view == "reached" and out.shares_pnl_per_100 is None and out.shares_ret is None
    assert out.pnl_per_contract == pytest.approx((9.0 - PICK["ask"]) * 100)


@pytest.mark.unit
def test_after_the_exit_date_a_target_never_reached_is_missed(tmp_path):
    _snap(tmp_path, datetime(2026, 12, 31, 21, tzinfo=UTC), spot=43.0, bid=9.0)
    out = picklog.score(_asked(), str(tmp_path), today=date(2027, 1, 5))
    assert out.status == "settled" and out.view == "missed" and out.days_left == -5


@pytest.mark.unit
def test_a_stock_that_has_not_moved_is_flat_and_one_that_fell_is_moving_away(tmp_path):
    _snap(tmp_path, datetime(2026, 10, 2, 21, tzinfo=UTC), spot=40.1, bid=10.0)
    assert picklog.score(_asked(), str(tmp_path), today=date(2026, 10, 3)).view == "flat"
    _snap(tmp_path, datetime(2026, 10, 3, 21, tzinfo=UTC), spot=37.0, bid=8.0)
    assert picklog.score(_asked(), str(tmp_path), today=date(2026, 10, 4)).view == "away"


@pytest.mark.unit
def test_a_request_that_named_no_contract_still_says_how_the_stock_did_against_the_view(tmp_path):
    _snap(tmp_path, datetime(2026, 10, 15, 21, tzinfo=UTC), spot=45.0, bid=12.0)
    out = picklog.score(_asked(pick=None), str(tmp_path), today=date(2026, 10, 16))
    assert out.status == "no contract" and out.contract is None and out.pnl_per_contract is None
    assert out.view == "toward" and out.shares_pnl_per_100 == pytest.approx(500.0)
    assert "no contract paid even if the view was right" in out.reason


@pytest.mark.unit
def test_the_summary_counts_views_whether_or_not_a_contract_was_named(tmp_path):
    _snap(tmp_path, datetime(2026, 10, 10, 21, tzinfo=UTC), spot=51.0, bid=20.0)
    picklog.record_pick(str(tmp_path), {**_entry(spot=40.0), "pick": PICK}, now=datetime(2026, 9, 30, tzinfo=UTC))
    picklog.record_pick(str(tmp_path), {**_entry(spot=40.0), "pick": None}, now=datetime(2026, 9, 30, 1, tzinfo=UTC))
    data = picklog.review_payload(str(tmp_path), date(2026, 10, 11))
    assert data["runs"] == 2 and data["runs_without_pick"] == 1
    assert data["summary"]["picks"] == 1 and data["summary"]["views"] == 2
    assert data["summary"]["views_tracked"] == 2 and data["summary"]["views_reached"] == 2


@pytest.mark.unit
def test_the_terminal_review_lays_each_request_out_as_then_and_now(tmp_path):
    _snap(tmp_path, datetime(2026, 10, 15, 21, tzinfo=UTC), spot=45.0, bid=12.0)
    picklog.record_pick(str(tmp_path), {**_entry(spot=40.0), "pick": PICK}, now=datetime(2026, 9, 30, 12, tzinfo=UTC))
    text = picklog.render_review(picklog.review(str(tmp_path), date(2026, 10, 16)), 1)
    assert "asked   2026-09-30 12:00 UTC   stock 40.00   contract ask 10.00   view: 50.00 by 2026-12-31" in text
    assert "contract bid 12.00 (+20.0%)" in text and "stock 45.00 (+12.5%)" in text
    assert "+200.00 $ per contract" in text and "100 shares +500.00 $" in text
    assert "moving toward your target (50% of the way)" in text and "76 days left" in text


@pytest.mark.unit
def test_live_refresh_archives_a_fresh_chain_per_open_ticker_and_survives_a_failed_fetch(tmp_path, monkeypatch):
    picklog.record_pick(str(tmp_path), _entry("IREN", exit_date="2026-12-31"), now=datetime(2026, 9, 30, tzinfo=UTC))
    picklog.record_pick(str(tmp_path), _entry("BAD", exit_date="2026-12-31"), now=datetime(2026, 9, 30, tzinfo=UTC))
    from tradingagents.dataflows.vendors import options as vendor

    def fetch(sym):
        if sym == "BAD":
            raise RuntimeError("no chain")
        return {"source": "Cboe delayed quotes", "as_of": "x", "spot": 44.0, "iv30": 70.0, "session": "2026-10-05",
                "feed": "delayed"}, [_quote(9.0)]

    monkeypatch.setattr(vendor, "fetch_chain", fetch)
    assert picklog.refresh_open(str(tmp_path), date(2026, 10, 5)) == ["IREN"]
    from tradingagents.options.archive import list_snapshots

    assert len(list_snapshots(str(tmp_path), "IREN")) == 1 and list_snapshots(str(tmp_path), "BAD") == []


@pytest.mark.unit
def test_a_move_that_rounds_to_zero_is_printed_without_a_sign():
    assert picklog._pct(-0.0000024) == "0.0%" and picklog._pct(0.1234) == "+12.3%" and picklog._pct(-0.031) == "-3.1%"
