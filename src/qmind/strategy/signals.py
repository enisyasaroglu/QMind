from enum import Enum


class Signal(Enum):
    """
    Defines the types of trading signals a strategy can generate.
    """

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"  # For closing existing positions

    def __str__(self):
        return self.value
