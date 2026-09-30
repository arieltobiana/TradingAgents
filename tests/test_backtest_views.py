"""Scoring the portfolio manager's stock view at the horizon it states.

A rating is sizing advice; the view is a forecast for the stock with its own
horizon. Before building an option selector on the model's direction, the
backtest has to say whether that direction predicts returns, so each view is
scored against the stock's return over exactly the trading days it names.
"""

from __future__ import annotations

import pytest

import tradingagents.backtest as bt
from tradingagents.agents.schemas import VIEW_FLAT_SD, PortfolioDecision, PortfolioRating, render_pm_decision
from tradingagents.backtest import parse_view, summarize_views
from tradingagents.decision_log import TradingMemoryLog


def _decision(direction=None, days=None) -> str:
    return render_pm_decision(PortfolioDecision(
        rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t",
        view_direction=direction, view_horizon_days=days,
    ))


def _log(tmp_path, rows):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "m.md")})
    for ticker, date, decision in rows:
        log.store_decision(ticker, date, decision)
    return tmp_path / "m.md"


@pytest.fixture
def returns(monkeypatch):
    """Stub fetch_returns: ``table[(ticker, date)] = raw`` or None for pending."""
    table: dict = {}
    calls: list = []

    def fake(ticker, trade_date, holding_days=5, benchmark="SPY"):
        calls.append((ticker, trade_date, holding_days, benchmark))
        raw = table.get((ticker, trade_date))
        if raw is None:
            return None, None, None, None
        return raw, raw - 0.01, holding_days, "2026-03-01"

    monkeypatch.setattr(bt, "fetch_returns", fake)
    # A 2% flat band for every stock, unless a test sets its own.
    monkeypatch.setattr(bt, "flat_band", lambda ticker, trade_date, days: 0.02)
    return table, calls


@pytest.mark.unit
@pytest.mark.parametrize("direction, days", [("up", 20), ("down", 1), ("flat", 252)])
def test_the_rendered_view_parses_back(direction, days):
    assert parse_view(_decision(direction, days)) == (direction, days)


@pytest.mark.unit
@pytest.mark.parametrize("direction, days", [
    ("bullish", 20), ("up", "3 months"), ("up", 0), ("up", 400), ("up", 2.5), (None, 20), ("up", None),
])
def test_junk_view_fields_are_dropped_not_fatal(direction, days):
    decision = PortfolioDecision(rating="Buy", executive_summary="s", investment_thesis="t",
                                 view_direction=direction, view_horizon_days=days)
    assert "**View**: not provided" in render_pm_decision(decision)
    assert parse_view(render_pm_decision(decision)) is None


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "", "Rating: Buy\n\nno view here",
    "**View**: not provided",
    "**View**: bullish over 20 trading days",
    "**View**: up over the next quarter",
    "**View**: up over 0 trading days",
    "**View**: up over 300 trading days",
    "**View**: up over " + "9" * 5000 + " trading days",  # would overflow int() parsing
    "**View**: up over 20 trading days, maybe down",
    "**View**: up over 20 trading days\n\n**View**: not provided",  # the last word stands
])
def test_a_malformed_or_missing_view_is_unreadable(text):
    assert parse_view(text) is None


@pytest.mark.unit
def test_a_freetext_view_in_the_rendered_shape_is_read():
    assert parse_view("- **View**: Down over 10 trading days.") == ("down", 10)


@pytest.mark.unit
def test_the_view_line_does_not_disturb_the_rating_or_the_fact_checker():
    from tradingagents.agents.facts import _claims
    from tradingagents.agents.rating import extract_rating

    text = _decision("up", 20)
    assert extract_rating(text) == "Hold"
    assert _claims(text) == []  # "20 trading days" is neither a % nor a multiple


@pytest.mark.unit
def test_each_direction_is_scored_on_its_own_hit_rule(tmp_path, returns):
    table, _ = returns
    log = _log(tmp_path, [
        ("A", "2026-01-05", _decision("up", 20)),
        ("B", "2026-01-05", _decision("up", 20)),
        ("C", "2026-01-05", _decision("down", 10)),
        ("D", "2026-01-05", _decision("flat", 5)),
        ("E", "2026-01-05", _decision("flat", 5)),
    ])
    table.update({("A", "2026-01-05"): 0.05, ("B", "2026-01-05"): -0.01,
                  ("C", "2026-01-05"): -0.03,
                  ("D", "2026-01-05"): 0.015, ("E", "2026-01-05"): -0.025})

    summary = summarize_views(log, {"benchmark_map": {"": "SPY"}})

    assert summary.resolved == 5 and summary.pending == 0 and summary.unscored == 0
    up, down, flat = (summary.by_direction[d] for d in ("up", "down", "flat"))
    assert up.count == 2 and up.hit_rate == 0.5 and round(up.mean_raw, 4) == 0.02
    assert round(up.mean_alpha, 4) == 0.01
    assert down.count == 1 and down.hit_rate == 1.0
    assert flat.count == 2 and flat.hit_rate == 0.5  # inside the (stubbed) 2% band, then outside it
    assert "right 50%" in summary.render()


@pytest.mark.unit
def test_a_view_is_measured_at_its_own_horizon_not_the_holding_period(tmp_path, returns):
    table, calls = returns
    log = _log(tmp_path, [("NVDA", "2026-01-05", _decision("up", 42)),
                          ("7203.T", "2026-01-05", _decision("down", 3))])
    table[("NVDA", "2026-01-05")] = 0.1

    summarize_views(log, {"holding_period_days": 5, "benchmark_map": {"": "SPY", ".T": "^N225"}})

    assert ("NVDA", "2026-01-05", 42, "SPY") in calls
    assert ("7203.T", "2026-01-05", 3, "^N225") in calls


@pytest.mark.unit
def test_a_view_whose_horizon_has_not_traded_is_pending(tmp_path, returns):
    table, _ = returns
    log = _log(tmp_path, [("NVDA", "2026-01-05", _decision("up", 20)),
                          ("NVDA", "2026-01-12", _decision("up", 200))])
    table[("NVDA", "2026-01-05")] = 0.02

    summary = summarize_views(log, {})

    assert summary.resolved == 1 and summary.pending == 1
    assert summary.by_direction["up"].count == 1


@pytest.mark.unit
def test_a_decision_without_a_readable_view_is_never_scored(tmp_path, returns):
    table, calls = returns
    log = _log(tmp_path, [
        ("A", "2026-01-05", "Rating: Buy\n\nold decision, no view"),
        ("B", "2026-01-05", _decision()),
        ("C", "2026-01-05", "Rating: Buy\n\n**View**: to the moon"),
    ])
    table.update({("A", "2026-01-05"): 0.1, ("B", "2026-01-05"): 0.1, ("C", "2026-01-05"): 0.1})

    summary = summarize_views(log, {})

    assert calls == []  # nothing fetched, so nothing can leak into a score
    assert summary.resolved == 0 and summary.pending == 0 and summary.unscored == 3
    assert summary.by_direction == {}


@pytest.mark.unit
def test_a_run_that_wrote_no_log_has_no_views(tmp_path, returns):
    result = bt.BacktestResult(run_id="r", log_path=tmp_path / "missing.md")
    assert summarize_views(result, {}).resolved == 0


@pytest.mark.unit
def test_the_flat_band_is_a_share_of_the_stocks_own_move_over_the_horizon(monkeypatch):
    import math

    import pandas as pd

    # Alternating +1% / -1% log moves: a daily standard deviation of about 1.026%.
    moves = [0.01 if i % 2 else -0.01 for i in range(40)]
    closes = pd.Series([100 * math.exp(sum(moves[:i])) for i in range(41)])
    seen = {}

    def fake_closes(ticker, start, end):
        seen["window"] = (start, end)
        return closes

    monkeypatch.setattr(bt, "get_closes", fake_closes)
    rets = moves[-20:]
    mean = sum(rets) / 20
    daily = math.sqrt(sum((r - mean) ** 2 for r in rets) / 19)

    assert bt.flat_band("X", "2026-03-02", 16) == pytest.approx(VIEW_FLAT_SD * daily * 4)
    assert seen["window"][1] == "2026-03-03"  # the analysis day's close is included, nothing after it

    monkeypatch.setattr(bt, "get_closes", lambda *a: closes[:10])
    assert bt.flat_band("X", "2026-03-02", 16) is None

    gap = closes.copy()
    gap.iloc[-5] = float("nan")  # a missing session inside the window
    monkeypatch.setattr(bt, "get_closes", lambda *a: gap)
    assert bt.flat_band("X", "2026-03-02", 16) is None


@pytest.mark.unit
def test_a_flat_view_without_volatility_stays_pending(tmp_path, returns, monkeypatch):
    table, _ = returns
    monkeypatch.setattr(bt, "flat_band", lambda *a: None)
    log = _log(tmp_path, [("A", "2026-01-05", _decision("flat", 5))])
    table[("A", "2026-01-05")] = 0.0

    summary = summarize_views(log, {})

    assert summary.resolved == 0 and summary.pending == 1


@pytest.mark.unit
def test_a_wide_band_turns_a_big_move_into_a_right_flat_call(tmp_path, returns, monkeypatch):
    table, _ = returns
    monkeypatch.setattr(bt, "flat_band", lambda ticker, d, days: 0.10 if ticker == "IREN" else 0.01)
    log = _log(tmp_path, [("IREN", "2026-01-05", _decision("flat", 60)),
                          ("SPY", "2026-01-05", _decision("flat", 2))])
    table.update({("IREN", "2026-01-05"): 0.06, ("SPY", "2026-01-05"): 0.015})

    flat = summarize_views(log, {}).by_direction["flat"]

    assert flat.count == 2 and flat.hit_rate == 0.5  # 6% is flat for IREN; 1.5% is not for SPY


@pytest.fixture
def prior(monkeypatch):
    """Stub prior_return: ``table[(ticker, date)] = the last 20 sessions' return``."""
    table: dict = {}
    monkeypatch.setattr(bt, "prior_return", lambda ticker, d: table.get((ticker, d)))
    return table


@pytest.mark.unit
def test_up_down_calls_are_scored_against_always_up_and_momentum(tmp_path, returns, prior):
    table, _ = returns
    log = _log(tmp_path, [
        ("A", "2026-01-05", _decision("up", 20)),     # stock rose and had been rising
        ("B", "2026-01-05", _decision("down", 20)),   # stock rose; had been falling
        ("C", "2026-01-05", _decision("up", 20)),     # stock fell; had been rising
        ("D", "2026-01-05", _decision("flat", 20)),   # no direction claimed: not in the comparison
    ])
    table.update({("A", "2026-01-05"): 0.05, ("B", "2026-01-05"): 0.04,
                  ("C", "2026-01-05"): -0.03, ("D", "2026-01-05"): 0.0})
    prior.update({("A", "2026-01-05"): 0.10, ("B", "2026-01-05"): -0.10, ("C", "2026-01-05"): 0.10})

    d = summarize_views(log, {}).directional

    assert d["the model's up/down calls"] == (3, pytest.approx(1 / 3))   # only A is right
    assert d["always up"] == (3, pytest.approx(2 / 3))                   # A and B rose
    assert d["follow the last 20 sessions"] == (3, pytest.approx(1 / 3))  # A right; B, C wrong
    assert "always up: n=3, right 67%" in summarize_views(log, {}).render()


@pytest.mark.unit
def test_a_cell_without_a_momentum_signal_is_left_out_of_all_three_rows(tmp_path, returns, prior):
    table, _ = returns
    log = _log(tmp_path, [("A", "2026-01-05", _decision("up", 5)), ("B", "2026-01-05", _decision("up", 5)),
                          ("C", "2026-01-05", _decision("up", 5))])
    table.update({("A", "2026-01-05"): 0.02, ("B", "2026-01-05"): 0.02, ("C", "2026-01-05"): 0.02})
    prior[("A", "2026-01-05")] = 0.03   # B has no history; C's last 20 sessions were flat
    prior[("C", "2026-01-05")] = 0.0

    d = summarize_views(log, {}).directional

    assert d == {"the model's up/down calls": (1, 1.0), "always up": (1, 1.0),
                 "follow the last 20 sessions": (1, 1.0)}


@pytest.mark.unit
def test_several_logs_are_scored_together(tmp_path, returns, prior):
    table, _ = returns
    one = TradingMemoryLog({"memory_log_path": str(tmp_path / "one.md")})
    two = TradingMemoryLog({"memory_log_path": str(tmp_path / "two.md")})
    one.store_decision("A", "2026-01-05", _decision("up", 20))
    two.store_decision("B", "2026-01-05", _decision("down", 20))
    table.update({("A", "2026-01-05"): 0.05, ("B", "2026-01-05"): -0.02})
    prior.update({("A", "2026-01-05"): 0.01, ("B", "2026-01-05"): 0.01})

    summary = summarize_views([tmp_path / "one.md", tmp_path / "two.md"], {})

    assert summary.resolved == 2 and summary.directional["the model's up/down calls"] == (2, 1.0)


@pytest.mark.unit
def test_prior_return_is_the_move_over_the_window_and_none_without_history(monkeypatch):
    import pandas as pd

    closes = pd.Series([100.0 + i for i in range(21)])  # 100 -> 120 over 20 sessions
    monkeypatch.setattr(bt, "get_closes", lambda *a: closes)
    assert bt.prior_return("X", "2026-03-02") == pytest.approx(0.20)
    monkeypatch.setattr(bt, "get_closes", lambda *a: closes[:5])
    assert bt.prior_return("X", "2026-03-02") is None
