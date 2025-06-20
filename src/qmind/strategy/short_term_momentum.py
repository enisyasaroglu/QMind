import pandas as pd
import ta  # For EMA calculation

from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class ShortTermMomentumStrategy(BaseStrategy):
    """
    A 'high-frequency-like' strategy aiming to capture rapid, short-term momentum
    bursts on 1-minute (or very short) timeframes.
    Generates BUY signal on strong bullish 1-minute bar breaking above a fast EMA.
    Generates SELL signal on strong bearish 1-minute bar breaking below a fast EMA.
    """

    def __init__(
        self, fast_ema_period: int = 3, bar_strength_threshold: float = 0.0005
    ):
        super().__init__(
            name="Short-Term Momentum Burst",
            params={
                "fast_ema_period": fast_ema_period,
                "bar_strength_threshold": bar_strength_threshold,  # Percentage of price range
            },
        )
        self.fast_ema_period = fast_ema_period
        self.bar_strength_threshold = (
            bar_strength_threshold  # e.g., 0.0005 means 0.05% move
        )

        if fast_ema_period < 1:
            logger.error("Fast EMA period must be at least 1.")
            raise ValueError("Invalid EMA period.")
        if bar_strength_threshold <= 0:
            logger.error("Bar strength threshold must be positive.")
            raise ValueError("Invalid bar strength threshold.")

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on rapid short-term momentum of 1-minute bars.

        Args:
            historical_data (pd.DataFrame): DataFrame with 'Open', 'Close', 'High', 'Low' prices
                                            and a datetime index (expected to be 1-minute bars).

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        if historical_data.empty:
            logger.warning(
                "Historical data is empty, cannot generate short-term momentum signals."
            )
            return pd.Series(dtype="object")

        if len(historical_data) < self.fast_ema_period:
            logger.warning(
                f"Not enough data ({len(historical_data)} bars) for fast EMA calculation "
                f"({self.fast_ema_period}). Skipping signal generation."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(
            f"Generating signals for Short-Term Momentum Strategy (EMA={self.fast_ema_period}, "
            f"threshold={self.bar_strength_threshold}) on {len(historical_data)} bars."
        )

        # Calculate a very fast EMA
        historical_data["fast_ema"] = ta.trend.EMAIndicator(
            close=historical_data["Close"], window=self.fast_ema_period, fillna=False
        ).ema_indicator()

        # Calculate bar strength (percentage change from Open to Close)
        historical_data["bar_strength"] = (
            historical_data["Close"] - historical_data["Open"]
        ) / historical_data["Open"]

        # Initialize signals Series with HOLD
        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # Generate BUY signals: Strong bullish bar AND Close above fast EMA
        buy_conditions = (
            historical_data["bar_strength"] > self.bar_strength_threshold
        ) & (historical_data["Close"] > historical_data["fast_ema"])

        # Generate SELL signals: Strong bearish bar AND Close below fast EMA
        sell_conditions = (
            historical_data["bar_strength"] < -self.bar_strength_threshold
        ) & (historical_data["Close"] < historical_data["fast_ema"])

        signals[buy_conditions] = Signal.BUY
        signals[sell_conditions] = Signal.SELL

        # Drop temporary columns
        historical_data.drop(
            columns=["fast_ema", "bar_strength"], inplace=True, errors="ignore"
        )

        logger.info(
            f"Generated {signals.value_counts().get(Signal.BUY, 0)} BUY signals, "
            f"{signals.value_counts().get(Signal.SELL, 0)} SELL signals."
        )
        return signals

    def on_bar(self, current_bar: pd.Series) -> Signal:
        """
        Placeholder for real-time signal generation for an incoming bar.
        In a true HFT-like scenario, this would involve very rapid calculations
        and potentially dynamic thresholds.
        """
        logger.debug(
            f"On-bar method called for {current_bar.name}. Returning HOLD for now."
        )
        return Signal.HOLD


# --- Test Block (only runs when short_term_momentum.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting Short-Term Momentum Strategy Test ---")

    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # IMPORTANT: For 1-minute data, pick a very recent past trading day.
    # Older 1-minute data can be very extensive, and APIs might limit how far back.
    # Ensure your chosen date was a trading day.
    # Example: Today is Friday, June 20, 2025. June 18, 2025 was a Wednesday.
    test_date = "2025-06-18"  # Adjust this to a recent past trading day

    logger.info(
        f"Fetching 1-minute data for SPY for {test_date} for short-term momentum test..."
    )
    spy_df_1min = ingestor.ingest_historical_data(
        symbol="SPY",
        timeframe="1Min",
        start_date=test_date,
        end_date=test_date,
        overwrite=True,  # Always re-fetch for fresh 1-min data if needed
    )

    if not spy_df_1min.empty:
        print("\nSPY 1-Minute Data for Strategy (last 5 rows):")
        print(spy_df_1min.tail())

        # Initialize the Short-Term Momentum Strategy
        # Fast EMA (e.g., 3-period) and a small bar strength threshold (e.g., 0.05%)
        strategy = ShortTermMomentumStrategy(
            fast_ema_period=3, bar_strength_threshold=0.0005
        )
        signals = strategy.generate_signals(spy_df_1min.copy())

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))

        logger.info(
            f"Total signals generated (excluding HOLD): {len(signals[signals != Signal.HOLD])}"
        )
        if len(signals[signals != Signal.HOLD]) == 0:
            logger.warning(
                "No BUY/SELL signals generated. Try adjusting parameters or test date."
            )

    else:
        logger.error(
            f"Could not fetch 1-minute SPY data for {test_date}. Check API key and date."
        )

    logger.info("--- Short-Term Momentum Strategy Test Complete ---")
