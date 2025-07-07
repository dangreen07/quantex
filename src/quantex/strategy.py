from quantex import DataSource


class Strategy:
    """A strategy class that will be inherited by all strategies."""

    index: int = 0

    def __init__(self, data_sources: dict[str, DataSource]):
        self.data_sources = data_sources

    def _increment_index(self) -> None:
        """Increment the index of the strategy."""
        self.index += 1

    def run(self):
        """Run the strategy."""
        raise NotImplementedError("Subclasses must implement this method.")
