from quantex.broker import Order
from enum import Enum


class ApplicationType(Enum):
    """
    The application type for a commission.
    """

    PER_ORDER = 1  ## Applies for each order
    PER_CONTRACT = 2  ## Applies for each contract


class Commission:
    def __init__(
        self,
        application: ApplicationType | None = None,
        fixed: float | None = None,
        percentage: float | None = None,
    ):
        """
        This object is used for calculating commissions for a strategy.

        Parameters:
            application:
                The application type for the commission.
            fixed:
                The fixed commission amount per order or contract. If None, don't use fixed commission.
            percentage:
                The percentage commission amount per order or contract. If None, don't use percentage commission.
        """
        self.application = application
        self.fixed = fixed
        self.percentage = percentage

    def calculate(self, order: Order) -> float:
        """
        Calculates the commission for an order.

        Parameters:
            order: The order to calculate the commission for.

        Returns:
            The commission for the order.
        """
        commission = 0
        if self.application == ApplicationType.PER_ORDER:
            if self.fixed is not None:
                commission += self.fixed
            if self.percentage is not None:
                commission += order.amount * order.fill_price * self.percentage
        elif self.application == ApplicationType.PER_CONTRACT:
            if self.fixed is not None:
                commission += self.fixed * order.amount
            if self.percentage is not None:
                commission += order.amount * order.fill_price * self.percentage
        return commission
