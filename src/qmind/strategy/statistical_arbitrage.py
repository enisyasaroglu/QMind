import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore  # For standardizing the spread

from qmind.strategy.base_strategy import BaseStrategy
from qmind.strategy.signals import Signal
from qmind.utils.logging_config import get_logger

logger = get_logger(__name__)


class PairsTradingStrategy(BaseStrategy):
    """
    A statistical arbitrage strategy focusing on pairs trading.
    It identifies a cointegrated (or highly correlated) pair and trades the deviation
    of their spread from its historical mean.

    Signals are generated based on the Z-score of the spread:
    - Long the spread (Long symbol1, Short symbol2) when Z-score is below -entry_zscore.
    - Short the spread (Short symbol1, Long symbol2) when Z-score is above +entry_zscore.
    - Close positions when Z-score reverts towards exit_zscore.
    """

    def __init__(
        self,
        symbol1: str,
        symbol2: str,
        lookback_window: int = 60,  # Window for calculating hedge ratio and Z-score
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
    ):
        super().__init__(
            name="Pairs Trading",
            params={
                "symbol1": symbol1,
                "symbol2": symbol2,
                "lookback_window": lookback_window,
                "entry_zscore": entry_zscore,
                "exit_zscore": exit_zscore,
            },
        )
        self.symbol1 = symbol1.upper()
        self.symbol2 = symbol2.upper()
        self.lookback_window = lookback_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore

        if lookback_window < 2:
            raise ValueError(
                "Lookback window must be at least 2 for spread calculation."
            )
        if entry_zscore <= exit_zscore:
            logger.warning(
                "Entry Z-score should be greater than exit Z-score for effective trading ranges."
            )

    def generate_signals(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals for a pair based on their spread's Z-score.

        Args:
            historical_data (pd.DataFrame): A DataFrame containing 'Close' prices for both
                                            symbol1 and symbol2. Columns should be named
                                            f'{self.symbol1}_Close' and f'{self.symbol2}_Close'.
                                            Indexed by datetime.

        Returns:
            pd.Series: A Series of Signal Enum members, indexed by datetime.
        """
        required_cols = [f"{self.symbol1}_Close", f"{self.symbol2}_Close"]
        if not all(col in historical_data.columns for col in required_cols):
            logger.error(
                f"Input DataFrame missing required columns for pair: {required_cols}"
            )
            return pd.Series(dtype="object")
        if historical_data.empty:
            logger.warning(
                "Historical data is empty, cannot generate pairs trading signals."
            )
            return pd.Series(dtype="object")
        if len(historical_data) < self.lookback_window:
            logger.warning(
                f"Not enough data ({len(historical_data)} bars) for lookback window "
                f"({self.lookback_window}). Skipping signal generation."
            )
            return pd.Series(
                index=historical_data.index, data=Signal.HOLD, dtype="object"
            )

        logger.info(
            f"Generating signals for {self.symbol1}/{self.symbol2} Pairs Trading Strategy "
            f"(lookback={self.lookback_window}, entry_z={self.entry_zscore}, exit_z={self.exit_zscore}) "
            f"on {len(historical_data)} bars."
        )

        # 1. Calculate the rolling hedge ratio (beta) using OLS regression
        # Dependent variable: symbol1_Close, Independent variable: symbol2_Close
        # We add a constant to the independent variable for the intercept in regression
        betas = (
            historical_data[[f"{self.symbol1}_Close", f"{self.symbol2}_Close"]]
            .rolling(window=self.lookback_window)
            .apply(
                lambda x: sm.OLS(x.iloc[:, 0], sm.add_constant(x.iloc[:, 1]))
                .fit()
                .params[1],
                raw=False,
            )
            .rename(columns={f"{self.symbol1}_Close": "beta"})["beta"]
        )  # Rename to 'beta' for clarity

        # 2. Calculate the spread: symbol1_Close - beta * symbol2_Close
        # Use fillna(0) for initial NaNs if rolling window starts from 0, or dropna()
        spread = (
            historical_data[f"{self.symbol1}_Close"]
            - betas * historical_data[f"{self.symbol2}_Close"]
        )

        # 3. Calculate the rolling Z-score of the spread
        # Rolling mean and standard deviation of the spread
        rolling_mean_spread = spread.rolling(window=self.lookback_window).mean()
        rolling_std_spread = spread.rolling(window=self.lookback_window).std()

        # Z-score: (current_spread - rolling_mean) / rolling_std
        z_scores = (spread - rolling_mean_spread) / rolling_std_spread
        historical_data["z_score"] = z_scores

        # Initialize signals Series with HOLD
        signals = pd.Series(
            index=historical_data.index, data=Signal.HOLD, dtype="object"
        )

        # State tracking for opening/closing positions (essential for pairs trading)
        # 0: No position, 1: Long spread, -1: Short spread
        position_state = pd.Series(0, index=historical_data.index, dtype=int)

        # Iterate through Z-scores to generate signals and manage state
        for i in range(1, len(historical_data)):
            current_z = historical_data["z_score"].iloc[i]
            prev_z = historical_data["z_score"].iloc[i - 1]
            prev_state = position_state.iloc[i - 1]

            # If no position
            if prev_state == 0:
                # Entry for Long Spread (Z-score drops below -entry_zscore)
                if current_z < -self.entry_zscore:
                    signals.iloc[i] = (
                        Signal.BUY
                    )  # Buy the spread (Long symbol1, Short symbol2)
                    position_state.iloc[i] = 1
                # Entry for Short Spread (Z-score rises above +entry_zscore)
                elif current_z > self.entry_zscore:
                    signals.iloc[i] = (
                        Signal.SELL
                    )  # Short the spread (Short symbol1, Long symbol2)
                    position_state.iloc[i] = -1
                else:
                    position_state.iloc[i] = 0  # Remain no position
            # If Long Spread
            elif prev_state == 1:
                # Exit Long Spread (Z-score reverts towards exit_zscore, or crosses 0)
                if (
                    current_z > -self.exit_zscore
                ):  # Or `current_z > 0` for more aggressive mean reversion
                    signals.iloc[i] = Signal.CLOSE  # Close long spread
                    position_state.iloc[i] = 0
                else:
                    position_state.iloc[i] = 1  # Remain long spread
            # If Short Spread
            elif prev_state == -1:
                # Exit Short Spread (Z-score reverts towards -exit_zscore, or crosses 0)
                if current_z < self.exit_zscore:  # Or `current_z < 0`
                    signals.iloc[i] = Signal.CLOSE  # Close short spread
                    position_state.iloc[i] = 0
                else:
                    position_state.iloc[i] = -1  # Remain short spread

        # Drop temporary columns
        historical_data.drop(columns=["z_score"], inplace=True, errors="ignore")

        # Count signals. Note: BUY/SELL refer to opening a spread position. CLOSE refers to closing it.
        logger.info(
            f"Generated {signals.value_counts().get(Signal.BUY, 0)} BUY spread signals, "
            f"{signals.value_counts().get(Signal.SELL, 0)} SELL spread signals, "
            f"{signals.value_counts().get(Signal.CLOSE, 0)} CLOSE spread signals."
        )
        return signals.iloc[
            self.lookback_window - 1 :
        ]  # Remove initial NaNs from rolling calculations

    def on_bar(self, current_bar: pd.Series) -> Signal:
        """
        Placeholder for real-time signal generation for an incoming bar.
        For pairs trading, this would involve updating the rolling beta and Z-score
        with the new bar data and comparing to thresholds, while maintaining position state.
        """
        logger.debug(
            f"On-bar method called for {current_bar.name}. Returning HOLD for now."
        )
        return Signal.HOLD


# --- Test Block (only runs when statistical_arbitrage.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting Pairs Trading Strategy Test ---")

    from qmind.data_management.ingestion import DataIngestor

    ingestor = DataIngestor()

    # Define a test pair (e.g., AAPL and MSFT often show some correlation)
    test_symbol1 = "AAPL"
    test_symbol2 = "MSFT"

    # Fetch historical data for both symbols
    logger.info(f"Fetching daily data for {test_symbol1}...")
    df1 = ingestor.ingest_historical_data(
        symbol=test_symbol1,
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2024-06-01",
        overwrite=False,
    )

    logger.info(f"Fetching daily data for {test_symbol2}...")
    df2 = ingestor.ingest_historical_data(
        symbol=test_symbol2,
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2024-06-01",
        overwrite=False,
    )

    if not df1.empty and not df2.empty:
        # Merge the close prices of the two symbols into a single DataFrame
        # This is how a Backtester would prepare data for a multi-asset strategy
        merged_df = pd.DataFrame(
            {
                f"{test_symbol1}_Close": df1["Close"],
                f"{test_symbol2}_Close": df2["Close"],
            }
        ).dropna()  # Drop rows where one symbol might have missing data

        print(f"\nMerged Data for {test_symbol1}/{test_symbol2} (last 5 rows):")
        print(merged_df.tail())

        # Initialize the Pairs Trading Strategy
        strategy = PairsTradingStrategy(
            symbol1=test_symbol1,
            symbol2=test_symbol2,
            lookback_window=60,  # 60 trading days ~ 3 months
            entry_zscore=2.0,
            exit_zscore=0.5,
        )

        # Generate signals
        signals = strategy.generate_signals(
            merged_df.copy()
        )  # Use copy to avoid modifying original df

        print("\nGenerated Signals (first 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].head(10))

        print("\nGenerated Signals (last 10, showing non-HOLD):")
        print(signals[signals != Signal.HOLD].tail(10))

    else:
        logger.error(
            "Could not fetch data for both symbols for Pairs Trading strategy test. Check data ingestion."
        )

    logger.info("--- Pairs Trading Strategy Test Complete ---")
