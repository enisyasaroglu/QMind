from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

from qmind.utils.logging_config import get_logger
from qmind.strategy.signals import Signal  # Import our Signal Enum

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Defines the interface that all concrete strategies must implement.
    """

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        logger.info(f"BaseStrategy initialized: {self.name} with params: {self.params}")

    @abstractmethod
    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Abstract method to generate a series of trading signals (BUY, SELL, HOLD, CLOSE)
        based on historical market data.

        Args:
            historical_data (pd.DataFrame): A DataFrame containing historical OHLCV data,
                                            indexed by datetime, with 'Open', 'High', 'Low', 'Close', 'Volume' columns.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime,
                       representing the signal for each corresponding bar.
        """
        pass

    @abstractmethod
    def on_bar(self, current_bar: pd.Series) -> Signal:
        """
        Abstract method for real-time signal generation based on a single, new incoming bar.
        This method is typically used in live trading or event-driven backtesting.

        Args:
            current_bar (pd.Series): A Series representing the most recent OHLCV bar.

        Returns:
            Signal: The trading signal (BUY, SELL, HOLD, CLOSE) for the current bar.
        """
        pass
