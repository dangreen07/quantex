from quantex import DataSource


class Strategy:
    """A strategy class that will be inherited by all strategies."""

    def __init__(self, data_sources: dict[str, DataSource]):
        self.data_sources = data_sources

    def run_strategy(self, idx: int):
        """Run the strategy."""
        raise NotImplementedError("Subclasses must implement this method.")
