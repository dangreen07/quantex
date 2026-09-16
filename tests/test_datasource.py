from quantex.datasource import DataSource, PricingData, YahooDataSource
import pandas as pd


def test_datasource():
    source = YahooDataSource("NVDA", start="2020-01-01", end="2020-12-31")
    assert len(source) == 252  ## 1 year of data
    assert len(source.Open) == 252
    source._current = 0
    assert len(source) == 0
    source._current = 10
    assert len(source) == 10
    assert len(source.Open) == 10


def test_pricing_data():
    data = pd.read_parquet("tests/data/NVDA.parquet")
    source1 = DataSource("NVDA", data)  ## 6 years of daily data
    data = pd.read_parquet("tests/data/MSFT.parquet")
    source2 = DataSource("MSFT", data)  ## 1 year of daily data
    context = PricingData()
    context.add_data(source1, "NVDA")
    assert len(context.datas["NVDA"]) == len(source1.Close)
    assert len(context.index) == len(source1.Timestamp)
    context.add_data(source2, "MSFT")
    assert len(context.datas["MSFT"]) == len(source1.Close)
    assert len(context.index) == len(source1.Timestamp)
