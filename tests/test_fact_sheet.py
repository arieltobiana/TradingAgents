"""The fact sheet is computed in code, and the final decision is checked against it.

The regression these pin: a fundamentals analyst compared receivables "+341%"
with revenue "4.7x", called receivables the faster of the two, and three later
agents built a thesis on it, one restating 4.7x as "470%".
"""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.agents import facts as agent_facts
from tradingagents.agents.facts import (
    check_decision,
    classify_claims,
    render_fact_sheet,
    unverified_claims,
)
from tradingagents.agents.rating import parse_rating
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
        "Start Date": pd.to_datetime(["2026-09-17", "2026-09-15", "2026-09-17", "2026-03-02"]),
        "Insider": ["CEO PERSON", "CFO PERSON", "CEO PERSON", "CFO PERSON"],
        "Position": ["Chief Executive Officer", "Chief Financial Officer", "Chief Executive Officer",
                     "Chief Financial Officer"],
        "Text": ["Sale at price 1567.26 - 1587.00 per share.", "Sale at price 1568.83 per share.",
                 "Stock Award(Grant) at price 0.00 per share.", "Sale at price 900.00 per share."],
        "Value": [53_272_704.0, 1_568_830.0, 0.0, 900_000.0],
    })
    income_stmt = pd.DataFrame(columns=pd.to_datetime(["2026-06-30", "2025-06-30"]))
    info = {"financialCurrency": "USD"}
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
    assert facts["Receivables vs revenue, year over year"]["value"] == (
        "receivables grew SLOWER than revenue: days sales outstanding fell from 51 to 48 days")
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

    got = classify_claims(text, sheet, references=[])

    assert [f.split(" ")[0] for f in got["unsupported"]] == ["470%"]
    assert [f.split(" ")[0] for f in got["verified"]] == ["341%", "4.7x", "372%"]
    assert [f.split(" ")[0] for f in got["proposal"]] == ["45%"]


def _id(sheet, prefix):
    return next(f["id"] for f in sheet["facts"] if f["label"].startswith(prefix))


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "receivables grew 4.7x faster than revenue",            # right number, wrong subject
    "revenue fell 50.7% from the prior quarter",           # right number, wrong direction
    "revenue grew 470 percent",                            # spelled out
    "revenue is 6.2 times its year-ago level",             # spelled out multiple
])
def test_the_checker_catches_what_a_bare_number_match_would_pass(vendor, text):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")

    assert unverified_claims(text, sheet, references=[]), text


@pytest.mark.unit
def test_a_cited_number_must_agree_with_the_fact_it_cites(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")
    yoy = _id(sheet, "Revenue growth year over year")
    qoq = _id(sheet, "Revenue growth vs prior quarter")

    assert unverified_claims(f"revenue grew 371.6% [{yoy}]", sheet, []) == []
    assert unverified_claims(f"revenue grew 371.6% [{qoq}]", sheet, [])


@pytest.mark.unit
def test_locale_and_range_forms_are_read(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")

    assert unverified_claims("le chiffre d'affaires est 4,7x", sheet, []) == []
    assert [f.split(" ")[0] for f in unverified_claims("somewhere 300-372%", sheet, [])] == ["300%"]


@pytest.mark.unit
def test_a_number_found_only_in_a_report_is_sourced_not_verified(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")

    got = classify_claims("a 12.3% pullback", sheet, ["closed 12.3% below the high"])

    assert got["unsupported"] == [] and got["verified"] == []
    assert got["report"][0].startswith("12.3%")


@pytest.mark.unit
def test_the_footer_never_reads_as_a_rating(vendor):
    state = {"fact_sheet": yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")}
    decision = "**Rating**: Buy\n\nOur rating: 470% upside."

    text, _ = check_decision(decision, state, llm=None)

    assert parse_rating(text) == "Buy"
    assert "rating" not in text[len(decision):].lower()


@pytest.mark.unit
def test_insider_history_the_vendor_does_not_reach_is_a_gap(vendor, monkeypatch):
    recent_only = _Ticker.insider_transactions.iloc[:3]
    monkeypatch.setattr(yahoo_facts.yf, "Ticker", lambda s: type("T", (_Ticker,), {"insider_transactions": recent_only})())

    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")

    assert not any(f["label"].startswith("Insider") for f in sheet["facts"])
    assert any(g.startswith("insider transactions before") for g in sheet["gaps"])


@pytest.mark.unit
def test_statement_money_is_labelled_in_its_own_currency(vendor, monkeypatch):
    monkeypatch.setattr(yahoo_facts.yf, "Ticker",
                        lambda s: type("T", (_Ticker,), {"info": {"financialCurrency": "JPY"}})())

    sheet = yahoo_facts.build_fact_sheet("7203.T", "2026-09-24")
    rev = next(f for f in sheet["facts"] if f["label"].startswith("Revenue, latest"))

    assert rev["currency"] == "JPY" and "JPY" in render_fact_sheet(sheet)
    insider = next(f for f in sheet["facts"] if f["label"].startswith("Insider open-market sales"))
    assert insider["currency"] == "USD"


@pytest.mark.unit
def test_a_fiscal_year_end_quarter_waits_for_the_10k(vendor):
    # 2026-06-30 is a fiscal year end here: 60 days on, a 10-Q quarter would
    # be public but a 10-K quarter is not yet.
    facts = _by_label(yahoo_facts.build_fact_sheet("SNDK", "2026-08-29"))

    assert not any("2026-06-30" in label for label in facts)
    assert "Revenue, latest quarter (2026-03-31)" in facts


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
    assert result["revised"] and (result["call_before"], result["call_after"]) == ("Underweight", "Hold")
    assert result["final"]["unsupported"] == []
    assert [r.split(" ")[0] for r in result["final"]["proposal"]] == ["45%"]
    assert "from Underweight to Hold" in text
    # The run's signal is parsed from the text with the footer attached.
    assert parse_rating(text) == "Hold"


@pytest.mark.unit
def test_a_revision_without_a_rating_is_discarded(vendor):
    state = {"fact_sheet": yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")}
    decision = "**Rating**: Sell\n\nA 470% surge."

    text, result = check_decision(decision, state, _Reviser("I cannot help with that."))

    assert not result["revised"]
    assert text.startswith(decision)
    assert result["final"]["unsupported"][0].startswith("470%")
    assert parse_rating(text) == "Sell"


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


@pytest.mark.unit
def test_a_proposal_referred_back_to_is_still_a_proposal(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")
    text = "Trim to 60-70% of a full position. A break lower would justify going below the 60-70% band."

    got = classify_claims(text, sheet, [])

    assert got["unsupported"] == []
    assert len(got["proposal"]) == 4


@pytest.mark.unit
def test_receivables_off_a_tiny_base_get_no_growth_rate_and_no_comparison(vendor, monkeypatch):
    # IREN: receivables $1.6M -> $21.1M read as +1,247% "faster than revenue"
    # while days sales outstanding showed collections were normal.
    class Tiny(_Ticker):
        quarterly_balance_sheet = _frame({"Accounts Receivable": [21.1e6, 69.1e6, 9.6e6, 24.1e6, 1.6e6]})

    monkeypatch.setattr(yahoo_facts.yf, "Ticker", lambda s: Tiny())
    sheet = yahoo_facts.build_fact_sheet("IREN", "2026-09-24")
    labels = {f["label"] for f in sheet["facts"]}

    assert "Accounts receivable growth year over year (percent)" not in labels
    assert "Receivables vs revenue, year over year" not in labels
    assert "Days sales outstanding, latest quarter" in labels
    assert any(g.startswith("accounts receivable growth rate") for g in sheet["gaps"])


@pytest.mark.unit
def test_citation_lists_are_read_and_a_citation_belongs_to_the_number_before_it(vendor):
    sheet = yahoo_facts.build_fact_sheet("SNDK", "2026-09-24")
    yoy, mult = _id(sheet, "Revenue growth year over year"), _id(sheet, "Revenue year over year as a multiple")
    five = next(f for f in sheet["facts"] if f["label"] == "Price change over last 5 sessions")
    text = (f"Revenue rose 371.6% [{yoy}, {mult}] to 4.72x [{yoy} and {mult}]. "
            f"Trim to 55-58% of a position into strength (+{five['value']:.1f}% over 5 sessions [{five['id']}]).")

    got = classify_claims(text, sheet, [])

    assert got["unsupported"] == []
    assert [v.split(" ")[0] for v in got["verified"]] == ["371.6%", "4.72x", f"{five['value']:.1f}%"]
    assert [p.split(" ")[0] for p in got["proposal"]] == ["55%", "58%"]
