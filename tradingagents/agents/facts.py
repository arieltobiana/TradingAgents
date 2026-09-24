"""The fact sheet in the agents' prompts, and the check on the final decision.

Two halves of one rule — the model reads and writes, code computes:

* :func:`fact_sheet_block` puts the code-computed facts (see
  ``dataflows/vendors/yahoo/facts.py``) in front of every agent that argues or
  decides, with the instruction to cite them rather than derive their own
  growth rates and comparisons.
* :func:`create_fact_checker` runs after the Portfolio Manager. It extracts
  every percentage and multiple the decision states, matches each against the
  fact sheet and the analysts' reports, and when something matches neither it
  asks the model once to correct or withdraw it. What still matches nothing is
  listed under the decision, so an unverified number is never presented as a
  checked one.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

from tradingagents.agents.rating import is_review, parse_rating

logger = logging.getLogger(__name__)

_DISPLAY_UNITS = {"usd", "price", "pct", "x", "days", "count", "date", "text"}

FACT_RULES = (
    "Rules for numbers:\n"
    "1. For any growth rate, ratio, comparison between two figures, or price level, use the "
    "fact sheet value and cite its id, e.g. [F12].\n"
    "2. Do not compute a growth rate or compare two growth rates yourself. If the fact sheet "
    "does not have it, say it is not available.\n"
    "3. A percentage change and a multiple are different things: +371.6% growth is 4.72x the "
    "base, not 471.6% or 470%. Copy whichever form the fact sheet states.\n"
    "4. If a report or argument conflicts with the fact sheet, the fact sheet is right; say so."
)


def _display(fact: Mapping[str, Any]) -> str:
    # Rendering lives with the Fact dataclass; import lazily so this module does
    # not pull the data layer into agents that never render a sheet.
    from tradingagents.dataflows.vendors.yahoo.facts import Fact
    return Fact(**fact).display()


def render_fact_sheet(sheet: Mapping[str, Any] | None) -> str:
    """Markdown table of the facts, or an explicit notice that there are none."""
    facts = list((sheet or {}).get("facts") or [])
    if not facts:
        return ("FACT SHEET: not available in this run. Treat every growth rate and comparison "
                "in the reports as unverified.")
    lines = ["FACT SHEET (computed in code from the raw data; authoritative)", "",
             "| ID | Fact | Value |", "|---|---|---:|"]
    for f in facts:
        lines.append(f"| {f['id']} | {f['label']} | {_display(f)} |")
    sources = sorted({f["source"] for f in facts})
    lines += ["", "Sources: " + "; ".join(sources) + "."]
    gaps = (sheet or {}).get("gaps") or []
    if gaps:
        lines.append("Not computed (data unavailable, not zero): " + ", ".join(gaps) + ".")
    return "\n".join(lines)


def fact_sheet_block(state: Mapping[str, Any]) -> str:
    """The fact sheet plus the rules for using it, for insertion into a prompt."""
    return render_fact_sheet(state.get("fact_sheet")) + "\n\n" + FACT_RULES


# ------------------------------------------------------------------ checking

# A number followed by % or a multiple sign. The lookbehind keeps "F12" ids and
# the tail of longer tokens out.
_CLAIM = re.compile(
    r"(?<![\w.])([-+−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+−]?\d+(?:\.\d+)?)\s*(%|x\b|×)"
)


def _claims(text: str) -> list[tuple[float, str, int, str]]:
    """(value, unit, decimals, snippet) for each % or x number in ``text``."""
    out = []
    for m in _CLAIM.finditer(text or ""):
        raw = m.group(1).replace(",", "").replace("−", "-")
        unit = "pct" if m.group(2) == "%" else "x"
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        start, end = max(0, m.start() - 80), min(len(text), m.end() + 40)
        snippet = " ".join(text[start:end].split())
        out.append((float(raw), unit, decimals, snippet))
    return out


def _reference_numbers(sheet: Mapping[str, Any] | None, references: Iterable[str]) -> dict[str, list[float]]:
    known: dict[str, list[float]] = {"pct": [], "x": []}
    for f in (sheet or {}).get("facts") or []:
        v = f.get("value")
        if isinstance(v, str):
            for value, unit, _, _ in _claims(v):
                known[unit].append(value)
        elif f.get("unit") in ("pct", "x"):
            known[f["unit"]].append(float(v))
    for text in references:
        for value, unit, _, _ in _claims(text):
            known[unit].append(value)
    return known


def _matches(value: float, decimals: int, candidates: list[float]) -> bool:
    # Half a unit in the last digit the claim shows, or 1.5% relative, whichever
    # is looser: "6.6%" matches 6.57, "4.7x" matches 4.72, "470%" matches
    # nothing near 371.6. The sign is compared loosely because prose carries it
    # in words ("a 7.5% pullback").
    for c in candidates:
        tol = max(0.5 * 10 ** (-decimals), 0.015 * abs(c))
        if abs(abs(value) - abs(c)) <= tol:
            return True
    return False


def unverified_claims(text: str, sheet: Mapping[str, Any] | None, references: Iterable[str]) -> list[str]:
    """Snippets around each % or x number in ``text`` that nothing supports."""
    known = _reference_numbers(sheet, references)
    flagged, seen = [], set()
    for value, unit, decimals, snippet in _claims(text):
        if _matches(value, decimals, known[unit]):
            continue
        key = (value, unit)
        if key not in seen:
            seen.add(key)
            flagged.append(f"{value:g}{'%' if unit == 'pct' else 'x'} — “{snippet}”")
    return flagged


def _footer(result: Mapping[str, Any]) -> str:
    lines = ["", "---", "**Fact check**"]
    if result["status"] == "skipped":
        lines.append(f"- Not run: {result['reason']}")
        return "\n".join(lines)
    lines.append(f"- Numbers checked against the fact sheet and analyst reports: {result['checked']}.")
    if result["revised"]:
        lines.append(f"- The decision was revised once to correct {len(result['initial'])} unsupported number(s).")
        if result["rating_before"] != result["rating_after"]:
            lines.append(f"- The rating changed in revision: {result['rating_before']} → {result['rating_after']}.")
    if result["remaining"]:
        lines.append("- Still not found in the data (a proposal such as a position size, or unverified):")
        lines += [f"  - {r}" for r in result["remaining"]]
    else:
        lines.append("- Every percentage and multiple matches the fact sheet or an analyst report.")
    return "\n".join(lines)


def _revision_prompt(decision: str, flagged: list[str], state: Mapping[str, Any]) -> str:
    items = "\n".join(f"- {f}" for f in flagged)
    return f"""You wrote the trading decision below. An automated check found numbers in it that match neither the fact sheet nor any analyst report:

{items}

For each one:
- If it is a factual claim about the company or the price, replace it with the correct fact sheet value (cite the id), or remove the claim.
- If it is your own proposal (a position size, a trim fraction, a level you are recommending), keep it.
- If a corrected fact was a reason for the rating, reconsider the rating and say in one sentence why it stands or changes.

Keep the same structure and headings, starting with the **Rating** line. Return only the revised decision.

{fact_sheet_block(state)}

---
DECISION TO REVISE:
{decision}"""


def check_decision(decision: str, state: Mapping[str, Any], llm: Any | None) -> tuple[str, dict]:
    """Check (and at most once revise) ``decision``; returns (text with footer, result)."""
    sheet = state.get("fact_sheet")
    if not (sheet or {}).get("facts"):
        result = {"status": "skipped", "reason": "no fact sheet was computed for this run"}
        return decision + _footer(result), result

    references = [state.get(k) or "" for k in
                  ("market_report", "fundamentals_report", "news_report", "sentiment_report")]
    initial = unverified_claims(decision, sheet, references)
    result = {"status": "checked", "checked": len(_claims(decision)), "initial": initial,
              "revised": False, "rating_before": parse_rating(decision)}
    result["rating_after"] = result["rating_before"]

    if initial and llm is not None:
        try:
            revised = llm.invoke(_revision_prompt(decision, initial, state)).content
        except Exception as exc:  # noqa: BLE001 — a failed revision keeps the original, flagged
            logger.warning("fact check: revision failed (%s); keeping the original decision", exc)
            revised = ""
        if isinstance(revised, str) and revised.strip() and not is_review(parse_rating(revised)):
            decision = revised.strip()
            result["revised"] = True
            result["rating_after"] = parse_rating(decision)
            result["checked"] = len(_claims(decision))

    result["remaining"] = unverified_claims(decision, sheet, references) if result["revised"] else initial
    return decision + _footer(result), result


def create_fact_checker(llm):
    def fact_check_node(state) -> dict:
        decision, result = check_decision(state["final_trade_decision"], state, llm)
        return {"final_trade_decision": decision, "fact_check": result}

    return fact_check_node
