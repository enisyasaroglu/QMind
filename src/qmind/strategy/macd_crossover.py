import pandas as pd
import ta  # Import the technical analysis library

from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class MACDCrossoverStrategy(BaseStrategy):
    """
    A trading strategy based on the Moving Average Convergence Divergence (MACD) indicator.
    Generates BUY signal when MACD line crosses above its Signal line.
    Generates SELL signal when MACD line crosses below its Signal line.
    """

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        super().__init__(
            name="MACD Crossover",
            params={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            },
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

        if not (0 < fast_period < slow_period and signal_period > 0):
            logger.error(
                "Invalid MACD periods: fast_period must be < slow_period, all must be positive."
            )
            raise ValueError("Invalid MACD parameters.")

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on MACD line crossing its Signal line.

        Args:
            historical_data (pd.DataFrame): DataFrame with 'Close' prices and a datetime index.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        if historical_data.empty:
            logger.warning("Historical data is empty, cannot generate MACD signals.")
            return pd.Series(dtype="object")

        required_data_length = (
            self.slow_period + self.signal_period
        )  # Approx minimum for full MACD
        if len(historical_data) < required_data_length:
            logger.warning(
                f"Not enough data ({len(historical_data)} bars) for MACD calculation "
                f"(need approx {required_data_length}). Skipping signal generation."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(
            f"Generating signals for MACD Crossover Strategy (fast={self.fast_period}, "
            f"slow={self.slow_period}, signal={self.signal_period}) "
            f"on {len(historical_data)} bars."
        )

        # Calculate MACD using the 'ta' library
        macd_indicator = ta.trend.MACD(
            close=historical_data["Close"],
            window_fast=self.fast_period,
            window_slow=self.slow_period,
            window_sign=self.signal_period,
            fillna=False,
        )
        historical_data["macd_line"] = macd_indicator.macd()
        historical_data["signal_line"] = macd_indicator.macd_signal()

        # Initialize signals Series with HOLD
        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # Generate BUY signals: MACD line crosses above Signal line
        # Condition: (Prev MACD < Prev Signal) AND (Current MACD > Current Signal)
        buy_conditions = (
            historical_data["macd_line"].shift(1)
            < historical_data["signal_line"].shift(1)
        ) & (historical_data["macd_line"] > historical_data["signal_line"])

        # Generate SELL signals: MACD line crosses below Signal line
        # Condition: (Prev MACD > Prev Signal) AND (Current MACD < Current Signal)
        sell_conditions = (
            historical_data["macd_line"].shift(1)
            > historical_data["signal_line"].shift(1)
        ) & (historical_data["macd_line"] < historical_data["signal_line"])

        signals[buy_conditions] = Signal.BUY
        signals[sell_conditions] = Signal.SELL

        # Drop temporary MACD columns
        historical_data.drop(
            columns=["macd_line", "signal_line"], inplace=True, errors="ignore"
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


# --- Test Block (only runs when macd_crossover.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting MACD Crossover Strategy Test ---")

    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # Fetch some SPY daily data for testing (need enough for MACD calculation)
    logger.info("Fetching SPY daily data for MACD strategy test...")
    spy_df = ingestor.ingest_historical_data(
        symbol="SPY",
        timeframe="1D",
        start_date="2023-01-01",  # Need sufficient history for 26-period slow MA and 9-period signal MA
        end_date="2024-06-01",
        overwrite=False,
    )

    if not spy_df.empty:
        print("\nSPY Data for Strategy (last 5 rows):")
        print(spy_df.tail())

        # Standard MACD parameters: (12, 26, 9)
        strategy = MACDCrossoverStrategy(
            fast_period=12, slow_period=26, signal_period=9
        )
        signals = strategy.generate_signals(spy_df.copy())

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))

    else:
        logger.error(
            "Could not fetch SPY data for MACD strategy test. Check data ingestion."
        )

    logger.info("--- MACD Crossover Strategy Test Complete ---")
