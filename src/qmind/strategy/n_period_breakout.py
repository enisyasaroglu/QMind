import pandas as pd

from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class NPeriodBreakoutStrategy(BaseStrategy):
    """
    A simple price action strategy that generates signals when the price breaks
    above the highest high or below the lowest low of the last N periods.
    """

    def __init__(self, lookback_period: int = 20):
        super().__init__(
            name="N-Period High/Low Breakout",
            params={"lookback_period": lookback_period},
        )
        self.lookback_period = lookback_period
        if lookback_period <= 0:
            logger.error("Lookback period must be greater than 0.")
            raise ValueError("Lookback period must be greater than 0.")

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on N-period high/low breakouts.

        Args:
            historical_data (pd.DataFrame): DataFrame with 'High' and 'Low' prices
                                            and a datetime index.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        if historical_data.empty:
            logger.warning(
                "Historical data is empty, cannot generate breakout signals."
            )
            return pd.Series(dtype="object")

        if len(historical_data) < self.lookback_period:
            logger.warning(
                f"Not enough data ({len(historical_data)} bars) for lookback period "
                f"({self.lookback_period}). Skipping signal generation."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(
            f"Generating signals for N-Period Breakout Strategy (lookback={self.lookback_period}) "
            f"on {len(historical_data)} bars."
        )

        # Calculate the highest high and lowest low over the lookback period
        historical_data["rolling_high"] = (
            historical_data["High"]
            .rolling(window=self.lookback_period, min_periods=1)
            .max()
            .shift(1)
        )  # Shift by 1 to avoid look-ahead bias
        historical_data["rolling_low"] = (
            historical_data["Low"]
            .rolling(window=self.lookback_period, min_periods=1)
            .min()
            .shift(1)
        )  # Shift by 1 to avoid look-ahead bias

        # Initialize signals Series with HOLD
        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # Generate BUY signals: Close breaks above rolling_high
        # Condition: (Current close > previous rolling high)
        buy_conditions = historical_data["Close"] > historical_data["rolling_high"]

        # Generate SELL signals: Close breaks below rolling_low
        # Condition: (Current close < previous rolling low)
        sell_conditions = historical_data["Close"] < historical_data["rolling_low"]

        signals[buy_conditions] = Signal.BUY
        signals[sell_conditions] = Signal.SELL

        # Drop temporary columns
        historical_data.drop(
            columns=["rolling_high", "rolling_low"], inplace=True, errors="ignore"
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


# --- Test Block (only runs when n_period_breakout.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting N-Period Breakout Strategy Test ---")

    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # Fetch some SPY daily data for testing
    logger.info("Fetching SPY daily data for N-Period Breakout strategy test...")
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

        # Test with a 20-period breakout (e.g., 20 trading days ~ 1 month)
        strategy = NPeriodBreakoutStrategy(lookback_period=20)
        signals = strategy.generate_signals(spy_df.copy())

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))

    else:
        logger.error(
            "Could not fetch SPY data for N-Period Breakout strategy test. Check data ingestion."
        )

    logger.info("--- N-Period Breakout Strategy Test Complete ---")
