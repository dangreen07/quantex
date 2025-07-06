import pytest
from quantex.sources import DataSource

def test_get_row_raises():
    ds = DataSource()
    with pytest.raises(NotImplementedError):
        ds.get_row(0)