from quantex.datasource import YahooDataSource


def test_datasource():
    source = YahooDataSource("NVDA", start="2020-01-01", end="2020-12-31")
    assert len(source) == 252  ## 1 year of data
    assert len(source.Open) == 252
    source._current = 10
    assert len(source) == 10
    assert len(source.Open) == 10
