from quant_research_stack.strategy_benchmark.data import UNIVERSES


def test_expanded_universe_set() -> None:
    names = {u.name for u in UNIVERSES}
    for required in {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "EW_BASKET"}:
        assert required in names
    assert len(UNIVERSES) >= 10
    assert all(len(u.tickers) >= 1 for u in UNIVERSES)
    assert len(names) == len(UNIVERSES)
