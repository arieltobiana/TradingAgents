"""The fact sheet in the agents' prompts, and the check on the final decision.

Two halves of one rule — the model reads and writes, code computes:

* :func:`fact_sheet_block` puts the code-computed facts (see
  ``dataflows/vendors/yahoo/facts.py``) in front of every agent that argues or
  decides, with the instruction to cite them rather than derive their own
  growth rates and comparisons.
* :func:`create_fact_checker` runs after the Portfolio Manager. It extracts
  every percentage and multiple the decision states and sorts each into one of
  four classes: *verified* (it cites a fact id and agrees with it, or matches a
  fact's value and direction), *from a report* (it appears only in an analyst
  report, which is itself model-written, so it is sourced but not checked),
  *proposal* (a trim fraction, a size, an ATR multiple for a stop) and
  *unsupported* (anything else, including a comparison that cites no fact).
  Unsupported numbers get one revision pass; whatever remains is listed under
  the decision, so an unverified number is never presented as a checked one.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

from tradingagents.agents.context import get_language_instruction
from tradingagents.agents.rating import is_review, parse_rating

logger = logging.getLogger(__name__)

FACT_RULES = (
    "Rules for numbers:\n"
    "1. For any growth rate, ratio, comparison between two figures, or price level, use the "
    "fact sheet value and cite its id, e.g. [F12].\n"
    "2. Do not compute a growth rate or compare two growth rates yourself. If the fact sheet "
    "does not have it, say it is not available.\n"
    "3. A percentage change and a multiple are different things: +371.6% growth is 4.72x the "
    "base, not 471.6% or 470%. Copy whichever form the fact sheet states.\n"
    "4. If a report or argument conflicts with the fact sheet, the fact sheet is right; say so.\n"
    "   A claim the fact sheet does not cover is UNVERIFIED, not false: say it is unverified and "
    "weigh it by how plausible it is. Do not treat its absence from the sheet as evidence against "
    "it.\n"
    "5. When a sentence compares two figures (faster, slower, outpaced, more than), cite the "
    "fact that states the comparison."
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

# A number: thousands separators, or a decimal comma (one or two digits, as
# written in most non-English locales), or a decimal point.
_NUM = r"[-+−]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+,\d{1,2}(?![\d,])|\d+(?:\.\d+)?)"
_UNIT = r"(?:%|\s?per\s?cent\b|\s?percent\b|x\b|×|\s?times\b|-?fold\b)"
_CLAIM = re.compile(
    rf"(?<![\w.,])(?P<a>{_NUM})"
    rf"(?:\s*(?:%|percent)?\s*(?:-|–|—|to)\s*(?P<b>{_NUM}))?"
    rf"\s*(?P<unit>{_UNIT})",
    re.IGNORECASE,
)
# "[F12]", "[F14, F15]", "[F27 and F28]", "[F31-F32]".
_CITE = re.compile(r"\[(F\d+(?:\s*(?:,|;|&|and|-|\u2013)\s*F?\d+)*)\]")
_CITE_WINDOW = 70


def _cited_ids(text: str) -> list[int]:
    return [int(n) for group in _CITE.findall(text) for n in re.findall(r"\d+", group)]
_CLAUSE_BREAK = re.compile(r"[.;:,\n()—]")
_PROPOSAL = re.compile(
    r"\b(trim|sell|reduce|cut|size|sizing|allocat\w*|position|stake|weight|exposure|stop|"
    r"target|add|buy|scale|risk no more)\b", re.IGNORECASE)
_COMPARISON = re.compile(
    r"\b(outpac\w*|outgr[eo]w\w*|faster than|slower than|more than|less than|exceed\w*|ahead of)\b",
    re.IGNORECASE)
_DOWN = re.compile(r"\b(fell|fall\w*|down|declin\w*|drop\w*|lower|decreas\w*|shr[ai]nk\w*|"
                   r"contract\w*|lost|loss|pullback|pulled back|retrac\w*|below|off)\b", re.IGNORECASE)
_UP = re.compile(r"\b(grew|grow\w*|up|rose|rise|rising|increas\w*|surg\w*|gain\w*|jump\w*|"
                 r"above|rall\w*|climb\w*)\b", re.IGNORECASE)
_SIGNED_LABEL = ("change", "growth", " vs ")


def _to_float(raw: str) -> float:
    raw = raw.replace("−", "-")
    if re.fullmatch(r"[-+]?\d+,\d{1,2}", raw):
        return float(raw.replace(",", "."))
    return float(raw.replace(",", ""))


def _decimals(raw: str) -> int:
    if re.fullmatch(r"[-+\u2212]?\d+,\d{1,2}", raw):
        return len(raw.split(",")[1])
    return len(raw.split(".")[1]) if "." in raw else 0


def _claims(text: str) -> list[dict]:
    """Each % or multiple number in ``text``, with the context the checker needs."""
    out = []
    text = text or ""
    matches = list(_CLAIM.finditer(text))
    for i, m in enumerate(matches):
        unit = "pct" if re.search(r"%|cent", m.group("unit"), re.IGNORECASE) else "x"
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line_end = len(text) if line_end == -1 else line_end
        before = text[line_start:m.start()]
        clause = _CLAUSE_BREAK.split(before)[-1][-60:]
        sentence_end = min((i for i in (text.find(". ", m.end()), line_end) if i != -1), default=line_end)
        after = text[m.end():min(sentence_end, m.end() + 80)]
        # A citation belongs to this number only if it comes before the next
        # number: "+6.1% over 5 sessions [F4]" does not cite a trim size
        # stated earlier in the same sentence.
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        cite_window = text[m.end():min(sentence_end, next_start, m.end() + _CITE_WINDOW)]
        sentence = text[max(line_start, text.rfind(". ", 0, m.start()) + 1):sentence_end]
        cites = _cited_ids(cite_window) or _cited_ids(clause)
        snippet = " ".join(text[max(line_start, m.start() - 80):min(line_end, m.end() + 40)].split())
        for raw in filter(None, (m.group("a"), m.group("b"))):
            value = _to_float(raw)
            direction = (-1 if value < 0 or _DOWN.search(clause) else
                         1 if raw.startswith("+") or _UP.search(clause) else 0)
            out.append({
                "value": abs(value), "unit": unit, "decimals": _decimals(raw), "snippet": snippet,
                "direction": direction, "cites": cites,
                "proposal": bool(_PROPOSAL.search(clause)) or bool(re.match(r"\s*ATR\b", after)),
                "comparison": bool(_COMPARISON.search(sentence)),
            })
    return out


def _fact_forms(fact: Mapping[str, Any]) -> list[tuple[str, float, int]]:
    """(unit, value, sign) forms a fact may legitimately be quoted in."""
    v, unit = fact.get("value"), fact.get("unit")
    if isinstance(v, str):
        return [(c["unit"], c["value"], 0) for c in _claims(v)]
    if unit not in ("pct", "x"):
        return []
    signed = unit == "pct" and any(w in fact.get("label", "").lower() for w in _SIGNED_LABEL)
    sign = (1 if v > 0 else -1 if v < 0 else 0) if signed else 0
    return [(unit, abs(float(v)), sign)]


def _close(value: float, decimals: int, candidate: float) -> bool:
    # Half a unit in the last digit shown, or 1.5% relative: "6.6%" matches
    # 6.57, "4.7x" matches 4.72, "470%" matches nothing near 371.6.
    return abs(value - candidate) <= max(0.5 * 10 ** (-decimals), 0.015 * abs(candidate))


def _agrees(claim: dict, forms: list[tuple[str, float, int]]) -> bool:
    for unit, value, sign in forms:
        if unit != claim["unit"] or not _close(claim["value"], claim["decimals"], value):
            continue
        if sign and claim["direction"] and sign != claim["direction"]:
            continue
        return True
    return False


def classify_claims(text: str, sheet: Mapping[str, Any] | None,
                    references: Iterable[str]) -> dict[str, list[str]]:
    """Sort every % / multiple in ``text`` into verified, report, proposal, unsupported."""
    facts = {int(f["id"][1:]): f for f in (sheet or {}).get("facts") or []}
    all_forms = [form for f in facts.values() for form in _fact_forms(f)]
    report_forms = [(c["unit"], c["value"], 0) for t in references for c in _claims(t)]
    out: dict[str, list[str]] = {"verified": [], "report": [], "proposal": [], "unsupported": []}
    seen = set()
    claims = _claims(text)
    # A proposal is often referred back to later ("below the 60-70% band"),
    # where the clause no longer carries the action word.
    proposed = {(c["value"], c["unit"]) for c in claims if c["proposal"] and not c["cites"]}
    for c in claims:
        label = f"{c['value']:g}{'%' if c['unit'] == 'pct' else 'x'}"
        if c["cites"]:
            if any(i in facts and _agrees(c, _fact_forms(facts[i])) for i in c["cites"]):
                kind = "verified"
            else:
                cited = ", ".join(f"F{i}" for i in c["cites"])
                kind, label = "unsupported", f"{label} (does not match the cited [{cited}])"
        elif c["proposal"] or (c["value"], c["unit"]) in proposed:
            kind = "proposal"
        elif c["comparison"]:
            kind, label = "unsupported", f"{label} (a comparison that cites no fact)"
        elif _agrees(c, all_forms):
            kind = "verified"
        elif _agrees(c, report_forms):
            kind = "report"
        else:
            kind = "unsupported"
        key = (kind, label, c["snippet"])
        if key not in seen:
            seen.add(key)
            out[kind].append(f"{label} — “{c['snippet']}”")
    return out


def unverified_claims(text: str, sheet: Mapping[str, Any] | None, references: Iterable[str]) -> list[str]:
    """The unsupported claims in ``text`` (see :func:`classify_claims`)."""
    return classify_claims(text, sheet, references)["unsupported"]


_REPORT_LIST_LIMIT = 8


def _footer(result: Mapping[str, Any]) -> str:
    lines = ["", "---", "**Fact check**"]
    if result["status"] == "skipped":
        lines.append(f"- Not run: {result['reason']}")
    else:
        final = result["final"]
        lines.append(
            f"- Percentages and multiples: {len(final['verified'])} verified against the fact sheet, "
            f"{len(final['report'])} taken from an analyst report (not independently checked), "
            f"{len(final['proposal'])} proposals, {len(final['unsupported'])} unsupported.")
        if result["revised"]:
            lines.append(f"- Revised once to correct {len(result['initial'])} unsupported number(s).")
            if result["call_before"] != result["call_after"]:
                lines.append(f"- The call changed in revision, from {result['call_before']} "
                             f"to {result['call_after']}.")
        if final["unsupported"]:
            lines.append("- Unsupported (not in the fact sheet or any report):")
            lines += [f"  - {r}" for r in final["unsupported"]]
        if final["report"]:
            lines.append("- From analyst reports only:")
            lines += [f"  - {r}" for r in final["report"][:_REPORT_LIST_LIMIT]]
            if len(final["report"]) > _REPORT_LIST_LIMIT:
                lines.append(f"  - and {len(final['report']) - _REPORT_LIST_LIMIT} more")
    # The rating parser reads the LAST "rating: X" in the text; nothing in the
    # footer may look like one, or it would overrule the decision it annotates.
    return re.sub(r"rating", "grade", "\n".join(lines), flags=re.IGNORECASE)


def _revision_prompt(decision: str, flagged: list[str], state: Mapping[str, Any]) -> str:
    items = "\n".join(f"- {f}" for f in flagged)
    return f"""You wrote the trading decision below. An automated check found numbers in it that match neither the fact sheet nor any analyst report, or that compare two figures without citing the fact that states the comparison:

{items}

For each one:
- If it is a factual claim about the company or the price, replace it with the correct fact sheet value and cite the id (e.g. [F12]), or remove the claim.
- If it is your own proposal (a position size, a trim fraction, a level you are recommending), keep it.
- If a corrected fact was a reason for the rating, reconsider the rating and say in one sentence why it stands or changes.

Keep the same structure and headings, starting with the **Rating** line. Return only the revised decision.

{fact_sheet_block(state)}

---
DECISION TO REVISE:
{decision}""" + get_language_instruction()


def check_decision(decision: str, state: Mapping[str, Any], llm: Any | None) -> tuple[str, dict]:
    """Check (and at most once revise) ``decision``; returns (text with footer, result)."""
    sheet = state.get("fact_sheet")
    if not (sheet or {}).get("facts"):
        result = {"status": "skipped", "reason": "no fact sheet was computed for this run"}
        return decision + _footer(result), result

    references = [state.get(k) or "" for k in
                  ("market_report", "fundamentals_report", "news_report", "sentiment_report")]
    initial = classify_claims(decision, sheet, references)
    result = {"status": "checked", "initial": initial["unsupported"], "revised": False,
              "call_before": parse_rating(decision)}
    final = initial

    if initial["unsupported"] and llm is not None:
        try:
            revised = llm.invoke(_revision_prompt(decision, initial["unsupported"], state)).content
        except Exception as exc:  # noqa: BLE001 — a failed revision keeps the original, flagged
            logger.warning("fact check: revision failed (%s); keeping the original decision", exc)
            revised = ""
        if isinstance(revised, str) and revised.strip() and not is_review(parse_rating(revised)):
            decision = revised.strip()
            result["revised"] = True
            final = classify_claims(decision, sheet, references)

    result["call_after"] = parse_rating(decision)
    result["final"] = final
    return decision + _footer(result), result


def create_fact_checker(llm):
    def fact_check_node(state) -> dict:
        decision, result = check_decision(state["final_trade_decision"], state, llm)
        return {"final_trade_decision": decision, "fact_check": result}

    return fact_check_node
