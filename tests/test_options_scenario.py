"""Ranking long options for a view the user states: direction, target, date."""

from __future__ import annotations

from datetime import date

import pytest

from tradingagents.dataflows.vendors.options import Quote
from tradingagents.options import scenario
from tradingagents.options.scenario import bs_price, rank_contracts

TODAY = date(2026, 9, 30)
EXIT = date(2026, 12, 31)
SPOT = 1000.0


def _quote(strike, expiry, right="C", iv=0.5, spread=0.02, oi=500.0):
    years = (expiry - TODAY).days / 365.0
    mid = bs_price(right, SPOT, strike, years, iv)
    return Quote(f"X{expiry:%y%m%d}{right}{int(strike * 1000):08d}", right, strike, expiry,
                 mid * (1 - spread / 2), mid * (1 + spread / 2), iv, 0.5, None, None, None, oi, 10.0)


def _chain(**kw):
    return [_quote(k, e, **kw) for e in (date(2027, 1, 15), date(2027, 3, 19), date(2027, 6, 18))
            for k in (900, 1000, 1100, 1200, 1300)]


@pytest.mark.unit
def test_put_call_parity_and_intrinsic_at_expiry():
    c, p = bs_price("C", 100, 95, 0.5, 0.3, 0.04), bs_price("P", 100, 95, 0.5, 0.3, 0.04)
    assert c - p == pytest.approx(100 - 95 * 2.718281828 ** (-0.04 * 0.5), abs=1e-6)
    assert bs_price("C", 120, 100, 0.0, 0.3) == 20 and bs_price("P", 120, 100, 0.0, 0.3) == 0
    assert bs_price("C", 100, 100, 1.0, 0.4) > bs_price("C", 100, 100, 1.0, 0.2)


@pytest.mark.unit
def test_with_no_opinion_the_expected_return_is_about_the_cost_of_the_trade():
    rows, _ = rank_contracts(_chain(), SPOT, "C", target=1000.0, exit_date=EXIT, today=TODAY)
    assert rows
    for r in rows:
        # getting in at the ask and out below the model value costs roughly the spread, never a gain or a disaster
        assert -0.12 < r.ev_market < 0.01, (r.symbol, r.ev_market)


@pytest.mark.unit
def test_an_expiry_too_close_to_the_exit_is_refused():
    chain = _chain() + [_quote(1100, date(2027, 1, 8))]          # 8 days after the exit
    rows, ctx = rank_contracts(chain, SPOT, "C", 1100.0, EXIT, TODAY)
    assert all(r.expiry != date(2027, 1, 8) for r in rows) and ctx["dropped"]["expires too soon"] == 1


@pytest.mark.unit
def test_being_right_loses_money_on_a_call_with_no_time_left_but_not_on_a_longer_one():
    rows, _ = rank_contracts(_chain(), SPOT, "C", target=1150.0, exit_date=EXIT, today=TODAY)
    by = {(r.expiry, r.strike): r for r in rows}
    near, far = by[(date(2027, 1, 15), 1200.0)], by[(date(2027, 6, 18), 1200.0)]
    assert far.roi_target > near.roi_target      # more time left on the exit date keeps more of the value
    assert near.roi_target < 0                   # the stock reaches the target and the call still loses


@pytest.mark.unit
def test_breakeven_is_where_the_return_is_zero():
    rows, _ = rank_contracts(_chain(), SPOT, "C", 1100.0, EXIT, TODAY)
    r = next(r for r in rows if r.strike == 1000.0 and r.expiry == date(2027, 3, 19))
    rows2, _ = rank_contracts(_chain(), SPOT, "C", r.breakeven, EXIT, TODAY)
    same = next(x for x in rows2 if x.symbol == r.symbol)
    assert same.roi_target == pytest.approx(0.0, abs=0.01)


@pytest.mark.unit
def test_puts_profit_from_a_fall_and_the_wrong_side_is_ignored():
    chain = _chain() + _chain(iv=0.5)[:0] + [_quote(k, date(2027, 3, 19), "P") for k in (900, 1000, 1100)]
    rows, ctx = rank_contracts(chain, SPOT, "P", target=850.0, exit_date=EXIT, today=TODAY)
    assert rows and all(r.symbol.endswith(f"P{int(r.strike * 1000):08d}") for r in rows)
    assert ctx["dropped"]["wrong side"] == len(_chain())
    assert all(r.roi_target > r.roi_flat for r in rows)       # the lower the stock, the better a put does


@pytest.mark.unit
def test_wide_thin_and_distant_strikes_are_filtered_and_counted():
    chain = [_quote(1000, date(2027, 3, 19), spread=0.30), _quote(1000, date(2027, 6, 18), oi=10.0),
             _quote(2000, date(2027, 3, 19)), _quote(1000, date(2027, 3, 19))]
    rows, ctx = rank_contracts(chain, SPOT, "C", 1100.0, EXIT, TODAY)
    assert len(rows) == 1
    d = ctx["dropped"]
    assert d["wide spread"] == 1 and d["thin open interest"] == 1 and d["far from spot"] == 1


@pytest.mark.unit
def test_no_usable_expiry_after_the_exit_date_is_reported_not_raised():
    rows, ctx = rank_contracts([_quote(1000, date(2026, 11, 20))], SPOT, "C", 1100.0, EXIT, TODAY)
    assert rows == [] and "error" in ctx
    assert "no contract passes" in scenario.render(rows, ctx, "X", "C", SPOT, 1100.0, EXIT)


@pytest.mark.unit
def test_a_volatility_drop_lowers_every_calls_return():
    a, _ = rank_contracts(_chain(), SPOT, "C", 1100.0, EXIT, TODAY)
    b, _ = rank_contracts(_chain(), SPOT, "C", 1100.0, EXIT, TODAY, iv_shift=-0.10)
    base = {r.symbol: r for r in a}
    assert all(r.roi_target < base[r.symbol].roi_target for r in b)
