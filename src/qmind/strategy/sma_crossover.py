import pandas as pd
from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class SMACrossoverStrategy(BaseStrategy):
    """
    A simple trading strategy based on the crossover of two Simple Moving Averages (SMA).
    Generates BUY signal when short_SMA crosses above long_SMA.
    Generates SELL signal when short_SMA crosses below long_SMA.
    """

    def __init__(self, short_window: int = 50, long_window: int = 200):
        super().__init__(
            name="SMA Crossover",
            params={"short_window": short_window, "long_window": long_window},
        )
        self.short_window = short_window
        self.long_window = long_window
        if short_window >= long_window:
            logger.warning(
                "Short SMA window should typically be less than Long SMA window."
            )

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on SMA crossover logic for historical data.

        Args:
            historical_data (pd.DataFrame): DataFrame with 'Close' prices and a datetime index.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        if historical_data.empty:
            logger.warning("Historical data is empty, cannot generate signals.")
            return pd.Series(dtype="object")  # Return empty Series of appropriate type

        # Ensure we have enough data for the longest window
        if len(historical_data) < self.long_window:
            logger.warning(
                f"Not enough data ({len(historical_data)} bars) for long SMA window "
                f"({self.long_window}). Skipping signal generation."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(
            f"Generating signals for SMA Crossover ({self.short_window}/{self.long_window}) "
            f"on {len(historical_data)} bars."
        )

        # Calculate SMAs
        historical_data["short_sma"] = (
            historical_data["Close"]
            .rolling(window=self.short_window, min_periods=1)
            .mean()
        )
        historical_data["long_sma"] = (
            historical_data["Close"]
            .rolling(window=self.long_window, min_periods=1)
            .mean()
        )

        # Initialize signals Series with HOLD
        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # Generate BUY signals: short_SMA crosses above long_SMA
        buy_conditions = (
            historical_data["short_sma"].shift(1) < historical_data["long_sma"].shift(1)
        ) & (historical_data["short_sma"] > historical_data["long_sma"])

        # Generate SELL signals: short_SMA crosses below long_SMA
        sell_conditions = (
            historical_data["short_sma"].shift(1) > historical_data["long_sma"].shift(1)
        ) & (historical_data["short_sma"] < historical_data["long_sma"])

        signals[buy_conditions] = Signal.BUY
        signals[sell_conditions] = Signal.SELL

        # Drop temporary SMA columns
        historical_data.drop(
            columns=["short_sma", "long_sma"], inplace=True, errors="ignore"
        )

        logger.info(
            f"Generated {signals.value_counts().get(Signal.BUY, 0)} BUY signals, "
            f"{signals.value_counts().get(Signal.SELL, 0)} SELL signals."
        )
        return signals

    def on_bar(self, current_bar: pd.Series) -> Signal:
        """
        This method is typically used in live trading or event-driven backtesting,
        where you'd maintain state (e.g., recent SMA values) and make a decision
        based on the incoming bar.

        For simplicity in this initial setup, we'll just return HOLD as a placeholder,
        as real-time state management is more complex and will be covered later
        with the backtesting engine.
        """
        logger.debug(
            f"On-bar method called for {current_bar.name}. Returning HOLD for now."
        )
        return Signal.HOLD


# --- Test Block (only runs when sma_crossover.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting SMA Crossover Strategy Test ---")

    # You need to initialize DataIngestor to get actual data
    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # Fetch some AAPL daily data for testing (e.g., last year)
    logger.info("Fetching AAPL daily data for strategy test...")
    aapl_df = ingestor.ingest_historical_data(
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-06-01",  # Go back further for 200 SMA
        end_date="2024-06-01",
        overwrite=False,  # Load from local if exists, only fetch new
    )

    if not aapl_df.empty:
        print("\nAAPL Data for Strategy (last 5 rows):")
        print(aapl_df.tail())

        # Initialize the SMA Crossover Strategy
        # Common SMA windows are 50/200 for daily, 9/20 for intraday
        strategy = SMACrossoverStrategy(short_window=50, long_window=200)

        # Generate signals
        signals = strategy.generate_signals(
            aapl_df.copy()
        )  # Use a copy to avoid modifying original df

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))  # Show only BUY/SELL signals

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))  # Show only BUY/SELL signals

        # Example: Check a specific date's signal if it exists
        # Example: check around a known crossover period if possible
        # latest_signal_date = signals.index[-1]
        # logger.info(f"Latest signal ({latest_signal_date.strftime('%Y-%m-%d')}): {signals.iloc[-1]}")

    else:
        logger.error(
            "Could not fetch AAPL data for strategy test. Check data ingestion."
        )

    logger.info("--- SMA Crossover Strategy Test Complete ---")
