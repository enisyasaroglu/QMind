import pandas as pd
import ta  # Import the technical analysis library

from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class BollingerBandsStrategy(BaseStrategy):
    """
    A mean reversion trading strategy based on Bollinger Bands.
    Generates BUY signal when price crosses below the lower band.
    Generates SELL signal when price crosses above the upper band.
    """

    def __init__(self, window: int = 20, window_dev: float = 2.0):
        super().__init__(
            name="Bollinger Bands Mean Reversion",
            params={
                "window": window,  # Period for SMA (Middle Band)
                "window_dev": window_dev,  # Number of standard deviations for bands
            },
        )
        self.window = window
        self.window_dev = window_dev

        if window <= 1 or window_dev <= 0:
            logger.error("Bollinger Bands window must be > 1 and window_dev > 0.")
            raise ValueError("Invalid Bollinger Bands parameters.")

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on Bollinger Bands crossover logic.

        Args:
            historical_data (pd.DataFrame): DataFrame with 'Close' prices and a datetime index.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        if historical_data.empty:
            logger.warning(
                "Historical data is empty, cannot generate Bollinger Bands signals."
            )
            return pd.Series(dtype="object")

        if len(historical_data) < self.window:
            logger.warning(
                f"Not enough data ({len(historical_data)} bars) for Bollinger Bands window "
                f"({self.window}). Skipping signal generation."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(
            f"Generating signals for Bollinger Bands Strategy (window={self.window}, "
            f"std_dev={self.window_dev}) on {len(historical_data)} bars."
        )

        # Calculate Bollinger Bands using the 'ta' library
        bb_indicator = ta.volatility.BollingerBands(
            close=historical_data["Close"],
            window=self.window,
            window_dev=self.window_dev,
            fillna=False,
        )
        historical_data["bb_lower"] = bb_indicator.bollinger_lband()
        historical_data["bb_upper"] = bb_indicator.bollinger_hband()

        # Initialize signals Series with HOLD
        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # Generate BUY signals: Close price crosses below the lower band
        # Condition: (Previous close was above or on lower band) AND (Current close is below lower band)
        buy_conditions = (
            historical_data["Close"].shift(1) >= historical_data["bb_lower"].shift(1)
        ) & (historical_data["Close"] < historical_data["bb_lower"])

        # Generate SELL signals: Close price crosses above the upper band
        # Condition: (Previous close was below or on upper band) AND (Current close is above upper band)
        sell_conditions = (
            historical_data["Close"].shift(1) <= historical_data["bb_upper"].shift(1)
        ) & (historical_data["Close"] > historical_data["bb_upper"])

        signals[buy_conditions] = Signal.BUY
        signals[sell_conditions] = Signal.SELL

        # Drop temporary BB columns
        historical_data.drop(
            columns=["bb_lower", "bb_upper"], inplace=True, errors="ignore"
        )

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


# --- Test Block (only runs when bollinger_bands_strategy.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting Bollinger Bands Strategy Test ---")

    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # Fetch some SPY daily data for testing (need enough for 20-period BB)
    logger.info("Fetching SPY daily data for Bollinger Bands strategy test...")
    spy_df = ingestor.ingest_historical_data(
        symbol="SPY",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2024-06-01",
        overwrite=False,
    )

    if not spy_df.empty:
        print("\nSPY Data for Strategy (last 5 rows):")
        print(spy_df.tail())

        strategy = BollingerBandsStrategy(
            window=20, window_dev=2.0
        )  # Common parameters
        signals = strategy.generate_signals(spy_df.copy())

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))

    else:
        logger.error(
            "Could not fetch SPY data for Bollinger Bands strategy test. Check data ingestion."
        )

    logger.info("--- Bollinger Bands Strategy Test Complete ---")
