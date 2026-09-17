from quantex.enums import MarginType, OrderDirection


class Margin:
    def __init__(
        self,
        initial_margin: tuple[float, float] = (0, 0),
        initial_margin_type: MarginType = MarginType.FIXED_PER_CONTRACT,
        maintenance_margin: tuple[float, float] = (0, 0),
        maintenance_margin_type: MarginType = MarginType.FIXED_PER_CONTRACT,
    ):
        """
        This class is used to calculate margin for a strategy.

        Parameters:
            initial_margin:
                A tuple of (long, short) margins needed to open a position.
            initial_margin_type:
                The type of the initial margin.
            maintenance_margin:
                A tuple of (long, short) margins needed to keep a position open. If account equity drops below the maintenance margin, the position will be closed.
            maintenance_margin_type:
                The type of the maintenance margin.
        """
        self.initial_margin = initial_margin
        self.initial_margin_type = initial_margin_type
        self.maintenance_margin = maintenance_margin
        self.maintenance_margin_type = maintenance_margin_type

    def get_initial_margin(
        self,
        direction: "OrderDirection",
        amount: float,
        price: float,
        multiplier: float,
    ) -> float:
        if self.initial_margin_type == MarginType.FIXED_PER_CONTRACT:
            if direction == OrderDirection.BUY:
                return self.initial_margin[0] * amount
            else:
                return self.initial_margin[1] * amount
        elif self.initial_margin_type == MarginType.PERCENTAGE:
            if direction == OrderDirection.BUY:
                return self.initial_margin[0] * amount * price * multiplier
            else:
                return self.initial_margin[1] * amount * price * multiplier
        else:
            raise ValueError("Invalid initial margin type")

    def get_maintenance_margin(
        self,
        direction: "OrderDirection",
        amount: float,
        price: float,
        multiplier: float,
    ) -> float:
        if self.maintenance_margin_type == MarginType.FIXED_PER_CONTRACT:
            if direction == OrderDirection.BUY:
                return self.maintenance_margin[0] * amount
            else:
                return self.maintenance_margin[1] * amount
        elif self.maintenance_margin_type == MarginType.PERCENTAGE:
            if direction == OrderDirection.BUY:
                return self.maintenance_margin[0] * amount * price * multiplier
            else:
                return self.maintenance_margin[1] * amount * price * multiplier
        else:
            raise ValueError("Invalid maintenance margin type")
