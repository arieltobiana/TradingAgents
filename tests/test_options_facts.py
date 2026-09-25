"""The options section of the fact sheet, and the check on the chosen contract."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from tradingagents.agents.facts import check_decision, check_option, option_task_block
from tradingagents.agents.rating import parse_rating
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating, render_pm_decision
from tradingagents.dataflows.vendors import options
from tradingagents.dataflows.vendors.options import Quote, option_facts, parse_occ, usable
from tradingagents.dataflows.vendors.yahoo.facts import Fact, _Sheet

TODAY = "2026-09-25"
SPOT = 46.0


def _occ(expiry: date, right: str, strike: float) -> str:
    return f"IREN{expiry:%y%m%d}{right}{int(strike * 1000):08d}"


def _q(expiry, right, strike, bid, ask, iv=0.8, delta=None, oi=500.0):
    if delta is None:
        # A rough monotone delta so the band pickers have something to find.
        m = (SPOT - strike) / SPOT
        delta = max(0.02, min(0.98, 0.5 + 2.2 * m)) if right == "C" else -max(0.02, min(0.98, 0.5 - 2.2 * m))
    return Quote(_occ(expiry, right, strike), right, strike, expiry, bid, ask, iv, delta,
                 0.04, -0.07, 0.05, oi, 100.0)


def _chain():
    today = date.fromisoformat(TODAY)
    chain = []
    for days in (7, 21, 56, 84, 147):
        e = today + timedelta(days=days)
        for strike in range(36, 60):
            mid = max(0.2, (SPOT - strike) + 2.0) if strike < SPOT else max(0.2, 3.0 - (strike - SPOT) * 0.3)
            chain.append(_q(e, "C", strike, round(mid * 0.97, 2), round(mid * 1.03, 2)))
            pmid = max(0.2, (strike - SPOT) + 2.0) if strike > SPOT else max(0.2, 3.0 - (SPOT - strike) * 0.3)
            chain.append(_q(e, "P", strike, round(pmid * 0.97, 2), round(pmid * 1.03, 2)))
    # Junk the section must drop rather than believe.
    e = today + timedelta(days=21)
    chain.append(Quote(_occ(e, "C", 43.5), "C", 43.5, e, 0.0, 0.0, 1e-5, 0.9, None, None, None, 0, 2))
    chain.append(Quote(_occ(e, "C", 44.5), "C", 44.5, e, 3.0, 2.0, 0.7, 0.8, None, None, None, 0, 2))
    return chain


@pytest.fixture
def market(monkeypatch, tmp_path):
    monkeypatch.setattr(options, "ny_today", lambda: TODAY)
    monkeypatch.setattr(options, "fetch_cboe", lambda s: (
        {"source": "Cboe delayed quotes", "as_of": f"{TODAY} 03:44", "spot": SPOT, "iv30": 76.0,
         "session": "2026-09-24"}, _chain()))
    report = pd.Timestamp("2026-11-05 16:00", tz="America/New_York")
    past = [(pd.Timestamp(d, tz="America/New_York"), "after close") for d in ("2026-08-27 16:00", "2026-05-08 16:00")]
    monkeypatch.setattr(options, "earnings_events", lambda s: [(report, "after close")] + past)
    monkeypatch.setattr(options, "earnings_reactions", lambda s, d, ev: [("2026-08-27", -12.5), ("2026-05-08", 7.7)])
    monkeypatch.setattr(options, "fetch_alpaca", lambda s, syms=None: [])
    return tmp_path


def _sheet(tmp_path, right="C", day=TODAY):
    sheet = _Sheet()
    sheet.add("Latest close", SPOT, "price", "bars")
    sheet.add("Annualized volatility, last 20 sessions", 78.0, "pct", "bars")
    option_facts(sheet, "IREN", day, right, str(tmp_path))
    return sheet


def _candidates(sheet):
    return [f for f in sheet.facts if f.label.startswith("Candidate ")]


@pytest.mark.unit
def test_occ_symbols_parse_and_bad_quotes_are_unusable():
    assert parse_occ("IREN261120C00049000") == ("IREN", date(2026, 11, 20), "C", 49.0)
    assert parse_occ("not an option") is None
    e = date(2026, 10, 16)
    assert not usable(Quote("X", "C", 43.5, e, 0.0, 0.0, 1e-5, 0.9, None, None, None, 0, 0))   # no market
    assert not usable(Quote("X", "C", 44.5, e, 3.0, 2.0, 0.7, 0.8, None, None, None, 0, 0))    # crossed
    assert not usable(Quote("X", "C", 45.0, e, 1.0, 1.1, None, 0.5, None, None, None, 0, 0))   # no IV
    assert usable(Quote("X", "C", 45.0, e, 1.0, 1.1, 0.7, 0.5, None, None, None, 0, 0))


@pytest.mark.unit
def test_candidates_are_one_per_expiry_and_delta_with_the_buyers_arithmetic(market):
    sheet = _sheet(market)
    cands = _candidates(sheet)

    assert cands and all("C" in f.label.split()[1][10:11] for f in cands)
    expiries = {f.label.split()[1][4:10] for f in cands}
    assert len(expiries) == 5
    row = next(f for f in cands if "(2026-11-20 49C)" in f.label)
    mid = (row.value.split("mid $")[1]).split(" ")[0]
    assert f"breakeven at expiry ${49 + float(mid):.2f}" in row.value
    assert "spans earnings: yes" in row.value
    near = next(f for f in cands if f.label.split()[1][4:10] == "261002")
    assert "spans earnings: no" in near.value
    assert all("43.5C" not in f.label and "44.5C" not in f.label for f in cands)


@pytest.mark.unit
def test_chain_level_volatility_and_earnings_facts(market):
    labels = {f.label: f for f in _sheet(market).facts}

    assert labels["30-day implied volatility (IV30)"].value == 76.0
    assert labels["IV30 vs 20-day realized volatility"].value == pytest.approx(76 / 78)
    assert labels["Next earnings report"].value == "2026-11-05 (after close)"
    assert labels["Average absolute earnings-day move, last 2 reports"].value == pytest.approx(10.1)
    assert any(label.startswith("Expiry 2026-11-20 (56 days, spans earnings: yes)") for label in labels)
    assert any("WHOLE period" in label for label in labels)


@pytest.mark.unit
def test_a_past_date_gets_a_gap_not_todays_chain(market):
    sheet = _sheet(market, day="2026-06-01")

    assert not _candidates(sheet)
    assert any(g.startswith("options (quotes and greeks are live-only") for g in sheet.gaps)


@pytest.mark.unit
def test_iv_rank_waits_for_enough_saved_history(market):
    sheet = _sheet(market)
    assert any(g.startswith("IV rank (1 of 60") for g in sheet.gaps)

    path = market / "options_history" / "IREN.csv"
    start = date(2026, 6, 1)
    rows = [f"{start + timedelta(days=i)},{50 + i}" for i in range(70)]
    stale = [f"{date(2025, 1, 1) + timedelta(days=i)},{500 + i}" for i in range(70)]  # over a year old
    path.write_text("date,iv30\n" + "\n".join(stale + rows) + "\n")
    sheet = _sheet(market)
    rank = next(f for f in sheet.facts if f.label.startswith("IV rank over"))
    # 76.0 against this year's 50..119, not against last year's 500s.
    assert rank.label.startswith("IV rank over 71 saved sessions")
    assert rank.value == pytest.approx((76 - 50) / (119 - 50) * 100)


@pytest.mark.unit
def test_a_damaged_history_file_is_read_around_and_rewritten_clean(market):
    path = market / "options_history" / "IREN.csv"
    path.parent.mkdir(parents=True)
    # Two headers (two first runs racing), a torn line, a duplicate session.
    path.write_text("date,iv30,spot\ndate,iv30,spot\n2026-09-20,70.0,46\n2026-09-2\n2026-09-20,71.0,46\n")

    history = options.record_iv(str(market), "IREN", "2026-09-24", 76.0, SPOT)

    assert history == [("2026-09-20", 71.0), ("2026-09-24", 76.0)]
    assert options.record_iv(str(market), "IREN", "2026-09-24", 99.0, SPOT)[-1] == ("2026-09-24", 76.0)
    assert path.read_text().splitlines()[0] == "date,iv30"


@pytest.mark.unit
def test_a_second_source_that_disagrees_is_named_on_the_row(market, monkeypatch):
    def alpaca(symbol, syms=None):
        return [Quote(s, "C", 49.0, date(2026, 11, 20), 5.0, 5.2, 0.60, 0.20, None, None, None, None, None)
                for s in syms]

    monkeypatch.setattr(options, "fetch_alpaca", alpaca)
    rows = _candidates(_sheet(market))

    assert all("SOURCES DISAGREE" in f.value for f in rows)


@pytest.mark.unit
def test_an_unreachable_second_source_is_a_gap(market, monkeypatch):
    def down(symbol, syms=None):
        raise RuntimeError("no keys")

    monkeypatch.setattr(options, "fetch_alpaca", down)
    sheet = _sheet(market)

    assert any(g.startswith("second-source cross-check") for g in sheet.gaps)


@pytest.mark.unit
@pytest.mark.parametrize("when, spans", [
    ("before open", "yes"), ("after close", "no"), ("time not confirmed", "ambiguous"),
])
def test_a_report_on_the_expiry_day_spans_it_only_if_it_comes_before_the_close(market, monkeypatch, when, spans):
    expiry = date(2026, 10, 16)  # one of the chain's expiries (21 days out)
    monkeypatch.setattr(options, "earnings_events",
                        lambda s: [(pd.Timestamp(f"{expiry} 12:00", tz="America/New_York"), when)])
    sheet = _sheet(market)
    label = next(f.label for f in sheet.facts if f.label.startswith(f"Expiry {expiry}") and "straddle" in f.label)

    assert f"spans earnings: {spans}" in label


@pytest.mark.unit
def test_earnings_reactions_measure_the_right_session_and_skip_unconfirmed(monkeypatch):
    days = pd.bdate_range("2026-08-24", "2026-09-04")
    closes = pd.DataFrame({"Date": days, "Close": [100.0 + i for i in range(len(days))]})
    monkeypatch.setattr(options, "load_ohlcv", lambda *a, **k: closes.copy())
    ny = "America/New_York"
    events = [(pd.Timestamp("2026-08-27 16:00", tz=ny), "after close"),     # 27th close -> 28th close
              (pd.Timestamp("2026-09-01 07:00", tz=ny), "before open"),     # 31st close -> 1st close
              (pd.Timestamp("2026-09-02 15:00", tz=ny), "time not confirmed")]

    got = dict(options.earnings_reactions("IREN", "2026-09-10", events))

    close = dict(zip(closes["Date"].dt.strftime("%Y-%m-%d"), closes["Close"], strict=True))
    assert got["2026-08-27"] == pytest.approx((close["2026-08-28"] / close["2026-08-27"] - 1) * 100)
    assert got["2026-09-01"] == pytest.approx((close["2026-09-01"] / close["2026-08-31"] - 1) * 100)
    assert "2026-09-02" not in got


@pytest.mark.unit
def test_the_earnings_day_move_is_backed_out_of_the_term_structure(market, monkeypatch):
    today = date.fromisoformat(TODAY)
    pre, post = today + timedelta(days=21), today + timedelta(days=56)
    chain = [_q(pre, "C", 46, 3.0, 3.1, iv=0.70), _q(pre, "P", 46, 3.0, 3.1, iv=0.70),
             _q(post, "C", 46, 5.0, 5.2, iv=0.85), _q(post, "P", 46, 5.0, 5.2, iv=0.85)]
    monkeypatch.setattr(options, "fetch_cboe", lambda s: (
        {"source": "Cboe delayed quotes", "as_of": TODAY, "spot": SPOT, "iv30": 76.0, "session": TODAY}, chain))
    facts = {f.label: f for f in _sheet(market).facts}

    sd = next(f for label, f in facts.items() if label.startswith("Earnings-day move implied by options, 1 standard"))
    import math
    assert sd.value == pytest.approx(math.sqrt(56 / 365 * (0.85 ** 2 - 0.70 ** 2)) * 100)


# ------------------------------------------------------------ decision side

def _state(tmp_path, question="call"):
    from dataclasses import asdict
    sheet = _sheet(tmp_path)
    return {"fact_sheet": {"facts": [asdict(f) for f in sheet.facts], "gaps": sheet.gaps},
            "option_question": question}


def _row(state, label_part):
    return next(f for f in state["fact_sheet"]["facts"] if label_part in f["label"])


@pytest.mark.unit
def test_a_chosen_candidate_passes_and_is_cited(market):
    state = _state(market)
    row = _row(state, "(2026-11-20 49C)")
    sym = row["label"].split()[1]
    ask = float(row["value"].split("ask ")[1].split(")")[0])

    got = check_option(f"**Option**: {sym}, limit ${ask:.2f}.", state)   # a trailing period, too

    assert got["problems"] == [] and got["cited"] == {sym: row["id"]}


@pytest.mark.unit
@pytest.mark.parametrize("text, problem", [
    ("**Option**: IREN261120C00099000 (limit 1.00 per share)", "not a candidate"),
    ("Buy some calls.", "has no '**Option**:' line"),
    # "none" in prose is not an answer, and a padded symbol is still read.
    ("None of the risks look binding. Buy the IREN  261120C00099000 at 9.", "not a candidate"),
    ("**Option**: IREN261120C00049000", "no limit price"),
])
def test_an_invented_contract_or_no_answer_is_a_problem(market, text, problem):
    got = check_option(text, _state(market))

    assert any(problem in p for p in got["problems"])


@pytest.mark.unit
def test_a_limit_above_the_ask_is_a_problem(market):
    state = _state(market)
    sym = _row(state, "(2026-11-20 49C)")["label"].split()[1]

    got = check_option(f"**Option**: {sym} (limit 99.00 per share)", state)

    assert any("above its ask" in p for p in got["problems"])


@pytest.mark.unit
def test_saying_none_is_an_answer(market):
    state = _state(market)
    sym = _row(state, "(2026-11-20 49C)")["label"].split()[1]

    got = check_option(f"**Option**: none — {sym} was too expensive for an Underweight view.", state)

    assert got["problems"] == [] and got["none"] and got["chosen"] == []


@pytest.mark.unit
def test_with_no_candidates_none_is_accepted_without_a_revision():
    state = {"fact_sheet": {"facts": [{"id": "F1", "label": "Latest close", "value": 46.0, "unit": "price",
                                       "source": "s", "currency": "USD"}], "gaps": []},
             "option_question": "call"}

    got = check_option("**Rating**: Hold\n\n**Option**: none", state)

    assert got["problems"] == [] and got["notes"]


@pytest.mark.unit
def test_an_option_problem_triggers_the_revision_and_the_footer_reports_it(market):
    state = _state(market)
    good = _row(state, "(2026-11-20 49C)")["label"].split()[1]

    class Reviser:
        def invoke(self, prompt):
            assert "OPTION:" in prompt and "THE QUESTION FOR THIS RUN" in prompt
            return type("R", (), {"content": f"**Rating**: Buy\n\n**Option**: {good} (limit 2.00 per share)"})()

    text, result = check_decision("**Rating**: Buy\n\n**Option**: IREN261120C00099000 (limit 1.00)", state, Reviser())

    assert result["revised"] and result["option"]["problems"] == []
    assert f"chose {good}" in text and parse_rating(text) == "Buy"


@pytest.mark.unit
def test_the_option_task_only_appears_when_asked():
    assert option_task_block({"option_question": ""}) == ""
    assert "which put to buy" in option_task_block({"option_question": "put"})


@pytest.mark.unit
def test_the_decision_renders_the_option_lines():
    d = PortfolioDecision(rating=PortfolioRating.BUY, executive_summary="s", investment_thesis="t",
                          option_contract="IREN261120C00049000", option_limit_price=5.3,
                          option_plan="1 contract; exit at +80% or 21 days before expiry")
    text = render_pm_decision(d)

    assert "**Option**: IREN261120C00049000 (limit 5.30 per share)" in text
    assert "**Option Plan**: 1 contract" in text
    assert "**Option**" not in render_pm_decision(
        PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t"))


def test_fact_display_is_unchanged_for_text_rows():
    assert Fact("F1", "x", "a; b", "text", "s").display() == "a; b"
