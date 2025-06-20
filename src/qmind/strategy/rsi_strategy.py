import pandas as pd
import ta  # Import the technical analysis library

from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class RSIStrategy(BaseStrategy):
    """
    A trading strategy based on the Relative Strength Index (RSI).
    Generates BUY signal when RSI crosses above the oversold level.
    Generates SELL signal when RSI crosses below the overbought level.
    """

    def __init__(
        self, window: int = 14, oversold_level: int = 30, overbought_level: int = 70
    ):
        super().__init__(
            name="RSI Overbought/Oversold",
            params={
                "window": window,
                "oversold_level": oversold_level,
                "overbought_level": overbought_level,
            },
        )
        self.window = window
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level

        if oversold_level >= overbought_level:
            logger.warning("Oversold level should be less than overbought level.")
        if not (0 <= oversold_level <= 100 and 0 <= overbought_level <= 100):
            logger.warning("RSI levels should be between 0 and 100.")

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on RSI for historical data.

        Args:
            historical_data (pd.DataFrame): DataFrame with 'Close' prices and a datetime index.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        if historical_data.empty:
            logger.warning("Historical data is empty, cannot generate RSI signals.")
            return pd.Series(dtype="object")

        if len(historical_data) < self.window:
            logger.warning(
                f"Not enough data ({len(historical_data)} bars) for RSI window "
                f"({self.window}). Skipping signal generation."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(
            f"Generating signals for RSI Strategy (window={self.window}, "
            f"oversold={self.oversold_level}, overbought={self.overbought_level}) "
            f"on {len(historical_data)} bars."
        )

        # Calculate RSI using the 'ta' library
        historical_data["rsi"] = ta.momentum.RSIIndicator(
            close=historical_data["Close"], window=self.window, fillna=False
        ).rsi()

        # Initialize signals Series with HOLD
        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # Generate BUY signals: RSI crosses above oversold level
        # Condition: (RSI was below oversold) AND (RSI is now above oversold)
        buy_conditions = (historical_data["rsi"].shift(1) < self.oversold_level) & (
            historical_data["rsi"] > self.oversold_level
        )

        # Generate SELL signals: RSI crosses below overbought level
        # Condition: (RSI was above overbought) AND (RSI is now below overbought)
        sell_conditions = (historical_data["rsi"].shift(1) > self.overbought_level) & (
            historical_data["rsi"] < self.overbought_level
        )

        signals[buy_conditions] = Signal.BUY
        signals[sell_conditions] = Signal.SELL

        # Drop temporary RSI column
        historical_data.drop(columns=["rsi"], inplace=True, errors="ignore")

        logger.info(
            f"Generated {signals.value_counts().get(Signal.BUY, 0)} BUY signals, "
            f"{signals.value_counts().get(Signal.SELL, 0)} SELL signals."
        )
        return signals

    def on_bar(self, current_bar: pd.Series) -> Signal:
        """
        Placeholder for real-time signal generation for an incoming bar.
        """
        logger.debug(
            f"On-bar method called for {current_bar.name}. Returning HOLD for now."
        )
        return Signal.HOLD


# --- Test Block (only runs when rsi_strategy.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting RSI Strategy Test ---")

    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # Fetch some AAPL daily data for testing
    logger.info(
        "Fetching AAPL daily data for RSI strategy test (longer history for RSI calc)..."
    )
    aapl_df = ingestor.ingest_historical_data(
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",  # Need more data for 14-period RSI
        end_date="2024-06-01",
        overwrite=False,
    )

    if not aapl_df.empty:
        print("\nAAPL Data for Strategy (last 5 rows):")
        print(aapl_df.tail())

        strategy = RSIStrategy(window=14, oversold_level=30, overbought_level=70)
        signals = strategy.generate_signals(aapl_df.copy())

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))

    else:
        logger.error(
            "Could not fetch AAPL data for RSI strategy test. Check data ingestion."
        )

    logger.info("--- RSI Strategy Test Complete ---")
