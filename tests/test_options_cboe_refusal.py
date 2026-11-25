"""fetch_chain: one Cboe 403 per run is logged once and skips Cboe for later symbols."""
import logging

import pytest
import requests

from tradingagents.dataflows.vendors import options


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(options, "_cboe_refused", None)
    monkeypatch.setattr(options, "fetch_alpaca", lambda s: [])


def _http_error(code):
    resp = requests.Response()
    resp.status_code = code
    return requests.HTTPError(f"{code} Client Error", response=resp)


def test_a_403_is_logged_once_and_cboe_is_not_asked_again(monkeypatch, caplog):
    calls = []

    def cboe(symbol):
        calls.append(symbol)
        raise _http_error(403)

    monkeypatch.setattr(options, "fetch_cboe", cboe)
    with caplog.at_level(logging.WARNING, logger=options.logger.name):
        for sym in ("QQQ", "SPY", "NVDA"):
            meta, _ = options.fetch_chain(sym)
            assert meta["feed"] == "indicative"
    assert calls == ["QQQ"]
    lines = [r.getMessage() for r in caplog.records if "Cboe" in r.getMessage()]
    assert len(lines) == 1 and "403" in lines[0] and "rest of this run" in lines[0]


def test_other_failures_are_per_symbol_and_retried(monkeypatch, caplog):
    calls = []

    def cboe(symbol):
        calls.append(symbol)
        raise _http_error(503)

    monkeypatch.setattr(options, "fetch_cboe", cboe)
    with caplog.at_level(logging.WARNING, logger=options.logger.name):
        for sym in ("QQQ", "SPY"):
            options.fetch_chain(sym)
    assert calls == ["QQQ", "SPY"]
    assert sum("Cboe unavailable" in r.getMessage() for r in caplog.records) == 2
