import pandas as pd
import numpy as np

from qmind.utils.logging_config import get_logger
from qmind.utils.constants import OHLCV_COLS

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Responsible for creating and managing technical indicators and other features
    from raw OHLCV data.
    """

    def __init__(self):
        logger.info("FeatureEngineer initialized. Ready to add features.")

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Placeholder method to add various technical indicators to a DataFrame.
        This method will be expanded later with actual feature calculations.

        Args:
            df (pd.DataFrame): The input DataFrame containing OHLCV data,
                               with a datetime index and columns: 'Open', 'High', 'Low', 'Close', 'Volume'.

        Returns:
            pd.DataFrame: The DataFrame with new feature columns added.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty, no features to add.")
            return df

        logger.info("Adding placeholder technical features...")

        # --- Placeholder for actual feature calculations ---
        # Example: Simple Moving Average (SMA) - will need libraries like `ta` or `talib` later
        # df['SMA_20'] = df['Close'].rolling(window=20).mean()

        # Example: Relative Strength Index (RSI) - will need libraries like `ta` or `talib` later
        # delta = df['Close'].diff()
        # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # rs = gain / loss
        # df['RSI_14'] = 100 - (100 / (1 + rs))

        # Placeholder for a simple price difference
        df["Price_Diff"] = df["Close"].diff()
        df["Daily_Return"] = df["Close"].pct_change() * 100

        logger.info("Placeholder features added.")
        return df.copy()  # Return a copy to avoid SettingWithCopyWarning


# --- Test Block (only runs when feature_engineer.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting FeatureEngineer Placeholder Test ---")

    # Create a dummy DataFrame for testing
    dummy_data = {
        "Open": np.random.rand(50) * 100 + 100,
        "High": np.random.rand(50) * 100 + 105,
        "Low": np.random.rand(50) * 100 + 95,
        "Close": np.random.rand(50) * 100 + 100,
        "Volume": np.random.randint(100000, 10000000, 50),
    }
    # Create a date range for the index, starting from a recent date
    dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq="D")
    dummy_df = pd.DataFrame(dummy_data, index=dates)
    dummy_df.index.name = "timestamp"  # Ensure index name is set

    print("\nOriginal Dummy DataFrame (first 5 rows):")
    print(dummy_df.head())

    # Initialize and use the FeatureEngineer
    fe = FeatureEngineer()
    features_df = fe.add_technical_features(dummy_df)

    print("\nDataFrame with Placeholder Features (first 5 rows):")
    print(features_df.head())
    print(f"Total columns after adding features: {len(features_df.columns)}")

    logger.info("--- FeatureEngineer Placeholder Test Complete ---")
