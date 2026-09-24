"""The fact sheet is computed in code, and the final decision is checked against it.

The regression these pin: a fundamentals analyst compared receivables "+341%"
with revenue "4.7x", called receivables the faster of the two, and three later
agents built a thesis on it, one restating 4.7x as "470%".
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.agents import facts as agent_facts
from tradingagents.agents.facts import check_decision, render_fact_sheet, unverified_claims
from tradingagents.dataflows.vendors.yahoo import facts as yahoo_facts
from tradingagents.graph.setup import GraphSetup

QUARTERS = pd.to_datetime(["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"])


def _frame(rows: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=QUARTERS).T


class _Ticker:
    quarterly_income_stmt = _frame({
        "Total Revenue": [8965e6, 5950e6, 3025e6, 2308e6, 1901e6],
        "Net Income": [6903e6, 3615e6, 803e6, 112e6, -23e6],
        "Gross Profit": [7582e6, 4662e6, 1541e6, 687e6, 498e6],
    })
    quarterly_balance_sheet = _frame({
        "Accounts Receivable": [4708e6, 2726e6, 1239e6, 1193e6, 1068e6],
        "Cash And Cash Equivalents": [4762e6, 3735e6, 1539e6, 1442e6, 1481e6],
        "Total Debt": [177e6, 200e6, 603e6, 1351e6, 2042e6],
    })
    quarterly_cashflow = _frame({"Free Cash Flow": [7083e6, 2993e6, 980e6, 438e6, 49e6]})
    insider_transactions = pd.DataFrame({
        "Start Date": pd.to_datetime(["2026-09-17", "2026-09-15", "2026-09-17"]),
        "Insider": ["CEO PERSON", "CFO PERSON", "CEO PERSON"],
        "Position": ["Chief Executive Officer", "Chief Financial Officer", "Chief Executive Officer"],
        "Text": ["Sale at price 1567.26 - 1587.00 per share.", "Sale at price 1568.83 per share.",
                 "Stock Award(Grant) at price 0.00 per share."],
        "Value": [53_272_704.0, 1_568_830.0, 0.0],
    })
    calendar = {}


def _prices(end="2026-09-24", n=260):
    dates = pd.bdate_range(end=end, periods=n)
    close = pd.Series(range(n), dtype=float) + 1000.0
    return pd.DataFrame({"Date": dates, "Open": close, "High": close + 10, "Low": close - 10,
                         "Close": close, "Volume": 1_000_000})


@pytest.fixture
def vendor(monkeypatch):
    monkeypatch.setattr(yahoo_facts, "load_ohlcv", lambda *a, **k: _prices())
    monkeypatch.setattr(yahoo_facts.yf, "Ticker", lambda s: _Ticker())
    monkeypatch.setattr(yahoo_facts, "get_current_date", lambda: "2026-09-24")


def _by_label(sheet):
    return {f["label"]: f for f in sheet["facts"]}


@pytest.mark.unit
def test_growth_is_stated_as_percent_and_as_multiple_and_the_comparison_is_computed(vendor):
    facts = _by_label(yahoo_facts.build_fact_sheet("SNDK", "2026-09-24"))

    assert facts["Revenue growth year over year (percent)"]["value"] == pytest.approx(371.6, abs=0.1)
    multiple = next(f for label, f in facts.items() if label.startswith("Revenue year over year as a multiple"))
    assert multiple["value"] == pytest.approx(4.716, abs=0.001)
    assert facts["Accounts receivable growth year over year (percent)"]["value"] == pytest.approx(340.8, abs=0.1)
    assert "SLOWER" in facts["Receivables vs revenue, year over year"]["value"]
    assert facts["Days sales outstanding, latest quarter"]["value"] == pytest.approx(47.8, abs=0.1)
    assert facts["Insider open-market purchases, last 90 days (value)"]["value"] == 0
    assert facts["Insider sales by CEO PERSON (Chief Executive Officer)"]["value"] == pytest.approx(53_272_704)


@pytest.mark.unit
def test_a_past_run_does_not_see_a_quarter_that_was_not_yet_filed(vendor):
    # 2026-06-30 ends 30 days before this run: inside the reporting lag.
    facts = _by_label(yahoo_facts.build_fact_sheet("SNDK", "2026-07-30"))

    assert "Revenue, latest quarter (2026-03-31)" in facts
    assert not any("2026-06-30" in label for label in facts)


@pytest.mark.unit
def test_a_section_that_fails_is_a_gap_not_a_failed_run(monkeypatch):
    monkeypatch.setattr(yahoo_facts, "load_ohlcv", lambda *a, **k: _prices())

    def broken(symbol):
        raise RuntimeError("vendor down")

    monkeypatch.setattr(yahoo_facts.yf, "Ticker", broken)
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")

    assert sheet["facts"], "price facts still computed"
    assert any(g.startswith("fundamentals") for g in sheet["gaps"])
    assert "Not computed" in render_fact_sheet(sheet)


@pytest.mark.unit
def test_the_checker_flags_the_misread_multiple_and_passes_the_true_numbers(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")
    text = ("Receivables grew 341% against a 470% revenue surge. Revenue is 4.7x a year ago, "
            "up 372%. Trim 45% of the position.")

    flagged = unverified_claims(text, sheet, references=[])

    assert [f.split(" ")[0] for f in flagged] == ["470%", "45%"]


@pytest.mark.unit
def test_numbers_the_analysts_reported_are_not_flagged(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")

    assert unverified_claims("a -7.5% pullback", sheet, ["closed -7.5% from the high"]) == []


class _Reviser:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("R", (), {"content": self.reply})()


@pytest.mark.unit
def test_an_unsupported_number_is_revised_once_and_what_remains_is_listed(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")
    state = {"fact_sheet": sheet}
    decision = "**Rating**: Underweight\n\nReceivables outpace a 470% revenue surge. Trim 45%."
    llm = _Reviser("**Rating**: Hold\n\nReceivables grew slower than revenue [F1]. Trim 45%.")

    text, result = check_decision(decision, state, llm)

    assert len(llm.prompts) == 1 and "470%" in llm.prompts[0]
    assert result["revised"] and (result["rating_before"], result["rating_after"]) == ("Underweight", "Hold")
    assert [r.split(" ")[0] for r in result["remaining"]] == ["45%"]
    assert "Underweight → Hold" in text and "Still not found" in text


@pytest.mark.unit
def test_a_revision_without_a_rating_is_discarded(vendor):
    state = {"fact_sheet": yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")}
    decision = "**Rating**: Sell\n\nA 470% surge."

    text, result = check_decision(decision, state, _Reviser("I cannot help with that."))

    assert not result["revised"]
    assert text.startswith(decision)
    assert result["remaining"] and result["remaining"][0].startswith("470%")


@pytest.mark.unit
def test_no_fact_sheet_means_the_check_says_it_did_not_run():
    text, result = check_decision("**Rating**: Hold", {"fact_sheet": {}}, llm=None)

    assert result["status"] == "skipped"
    assert "Not run" in text


@pytest.mark.unit
def test_every_arguing_agent_sees_the_fact_sheet():
    import inspect

    from tradingagents.agents.analysts import fundamentals_analyst
    from tradingagents.agents.managers import portfolio_manager, research_manager
    from tradingagents.agents.researchers import bear_researcher, bull_researcher
    from tradingagents.agents.risk_mgmt import (
        aggressive_debator,
        conservative_debator,
        neutral_debator,
    )
    from tradingagents.agents.trader import trader

    for module in (fundamentals_analyst, portfolio_manager, research_manager, bear_researcher,
                   bull_researcher, aggressive_debator, conservative_debator, neutral_debator, trader):
        assert "fact_sheet_block(state)" in inspect.getsource(module), module.__name__


@pytest.mark.unit
@pytest.mark.parametrize("enabled", [True, False])
def test_the_fact_check_node_follows_the_config(enabled):
    from tradingagents.graph.conditional_logic import ConditionalLogic

    workflow = GraphSetup(object(), object(), ConditionalLogic(), fact_check=enabled).setup_graph(("market",))

    assert ("Fact Check" in workflow.nodes) is enabled


def test_rules_explain_the_multiple_confusion():
    assert "4.72x" in agent_facts.FACT_RULES and "470%" in agent_facts.FACT_RULES
