import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from qmind.config import settings
from qmind.utils.logging_config import get_logger
from qmind.utils.constants import OHLCV_COLS, DATE_FORMAT
from qmind.data_management.sources import AlpacaDataClient

logger = get_logger(__name__)


class DataIngestor:
    """
    Manages the ingestion, local storage, and loading of historical market data.
    """

    def __init__(self):
        self.data_client = AlpacaDataClient()  # Initialize the Alpaca data client
        self.base_data_path = (
            settings.DATA_STORAGE_PATH
        )  # Base path for local data storage
        os.makedirs(
            self.base_data_path, exist_ok=True
        )  # Ensure base data directory exists
        logger.info(f"DataIngestor initialized. Local data path: {self.base_data_path}")

    def _get_local_file_path(self, symbol: str, timeframe: str) -> str:
        """Constructs the local Parquet file path for a given symbol and timeframe."""
        filename = f"{symbol.upper()}_{timeframe.upper()}.parquet"
        return os.path.join(self.base_data_path, filename)

    def ingest_historical_data(
        self,
        symbol: str,
        timeframe: str = "1D",
        start_date: Optional[str] = None,  # YYYY-MM-DD
        end_date: Optional[str] = None,  # YYYY-MM-DD
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Ingests historical data for a symbol from Alpaca, stores it locally,
        and returns the data as a DataFrame.

        Args:
            symbol (str): The stock ticker symbol.
            timeframe (str): The time aggregation (e.g., "1D", "1H", "1Min").
            start_date (Optional[str]): Start date for data download. Uses settings if None.
            end_date (Optional[str]): End date for data download. Uses settings if None.
            overwrite (bool): If True, existing local data will be overwritten.
                              If False, data will be loaded from local if available.

        Returns:
            pd.DataFrame: A DataFrame with the ingested OHLCV data.
        """
        local_file_path = self._get_local_file_path(symbol, timeframe)
        df = pd.DataFrame(columns=OHLCV_COLS)

        if os.path.exists(local_file_path) and not overwrite:
            logger.info(
                f"Local data found for {symbol} ({timeframe}). Loading from {local_file_path}"
            )
            try:
                df = pd.read_parquet(local_file_path)
                logger.info(
                    f"Loaded {len(df)} records from local storage for {symbol} ({timeframe})."
                )

                # Optional: Check if local data needs updating (e.g., fetch newer bars)
                # For simplicity, we'll implement a full fetch if local data doesn't cover the requested range
                # or if the latest date is not recent enough.
                if not df.empty:
                    latest_local_date = df.index.max().strftime(DATE_FORMAT)
                    # If end_date is provided and newer than local, consider fetching more
                    if (
                        end_date
                        and datetime.strptime(end_date, DATE_FORMAT).date()
                        > df.index.max().date()
                    ):
                        logger.info(
                            f"Local data for {symbol} ends at {latest_local_date}, fetching new data up to {end_date}."
                        )
                        # Fetch new data from the day after the latest local data
                        fetch_start_date = (
                            df.index.max() + timedelta(days=1)
                        ).strftime(DATE_FORMAT)
                        new_data = self.data_client.get_historical_bars(
                            symbol,
                            timeframe,
                            start_date=fetch_start_date,
                            end_date=end_date,
                        )
                        if not new_data.empty:
                            df = (
                                pd.concat([df, new_data]).drop_duplicates().sort_index()
                            )
                            logger.info(
                                f"Appended {len(new_data)} new records. Total records: {len(df)}."
                            )
                            df.to_parquet(local_file_path)  # Save updated data
                        else:
                            logger.info(
                                f"No new data found for {symbol} between {fetch_start_date} and {end_date}."
                            )
                    else:
                        logger.info(
                            f"Local data for {symbol} appears up-to-date for the requested range (up to {latest_local_date})."
                        )
                else:  # If local file existed but was empty
                    logger.warning(
                        f"Local data file for {symbol} ({timeframe}) was empty. Fetching from Alpaca."
                    )
                    df = self.data_client.get_historical_bars(
                        symbol, timeframe, start_date, end_date
                    )
                    if not df.empty:
                        df.to_parquet(local_file_path)  # Save newly fetched data
            except Exception as e:
                logger.error(
                    f"Error loading local data for {symbol} ({timeframe}): {e}. Attempting to re-fetch from Alpaca.",
                    exc_info=True,
                )
                df = self.data_client.get_historical_bars(
                    symbol, timeframe, start_date, end_date
                )
                if not df.empty:
                    df.to_parquet(local_file_path)  # Save newly fetched data
        else:
            logger.info(
                f"No local data found or overwrite requested for {symbol} ({timeframe}). Fetching from Alpaca."
            )
            df = self.data_client.get_historical_bars(
                symbol, timeframe, start_date, end_date
            )
            if not df.empty:
                df.to_parquet(local_file_path)  # Save newly fetched data
            else:
                logger.warning(
                    f"Failed to fetch any data for {symbol} ({timeframe}) from Alpaca."
                )

        return df

    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Loads historical data from local storage for a given symbol and timeframe.
        Does NOT attempt to fetch from source if local file not found.

        Args:
            symbol (str): The stock ticker symbol.
            timeframe (str): The time aggregation (e.g., "1D", "1H", "1Min").

        Returns:
            pd.DataFrame: A DataFrame with OHLCV data, or empty if not found.
        """
        local_file_path = self._get_local_file_path(symbol, timeframe)
        if os.path.exists(local_file_path):
            logger.info(
                f"Loading local data for {symbol} ({timeframe}) from {local_file_path}."
            )
            try:
                df = pd.read_parquet(local_file_path)
                logger.info(
                    f"Successfully loaded {len(df)} records for {symbol} ({timeframe})."
                )
                return df
            except Exception as e:
                logger.error(
                    f"Error loading {symbol} ({timeframe}) data from {local_file_path}: {e}",
                    exc_info=True,
                )
                return pd.DataFrame(columns=OHLCV_COLS)
        else:
            logger.warning(
                f"No local data file found for {symbol} ({timeframe}) at {local_file_path}."
            )
            return pd.DataFrame(columns=OHLCV_COLS)


# --- Test Block (only runs when ingestion.py is executed directly) ---
if __name__ == "__main__":
    logger.info("--- Starting DataIngestor Test ---")
    ingestor = DataIngestor()

    # Test 1: Ingest AAPL daily data (first time or overwrite)
    logger.info("\n--- Test 1: Ingesting AAPL Daily Data (may overwrite/fetch) ---")
    aapl_daily_df = ingestor.ingest_historical_data(
        symbol="AAPL",
        timeframe="1D",
        start_date="2024-01-01",
        end_date="2024-06-01",  # Ensure it's not too far in future
        overwrite=True,  # Set to True to force fresh download
    )
    print("\nAAPL Daily Data (first 5 rows):")
    print(aapl_daily_df.head())
    print(f"Total AAPL daily records: {len(aapl_daily_df)}")

    # Test 2: Load AAPL daily data (should load from local, no re-fetch)
    logger.info("\n--- Test 2: Loading AAPL Daily Data (from local) ---")
    loaded_aapl_daily_df = ingestor.load_historical_data(symbol="AAPL", timeframe="1D")
    print("\nLoaded AAPL Daily Data (first 5 rows):")
    print(loaded_aapl_daily_df.head())
    print(f"Total loaded AAPL daily records: {len(loaded_aapl_daily_df)}")

    # Test 3: Ingest SPY 1-minute data (for a recent *past* day, e.g., June 18, 2025 which was a Wednesday)
    logger.info(
        "\n--- Test 3: Ingesting SPY 1-Minute Data (for a specific past day) ---"
    )
    # Adjust the date to a recent past trading day (e.g., one or two days before today, excluding weekends)
    # Today is Friday, June 20, 2025. June 18, 2025 was a Wednesday.
    test_minute_date = "2025-06-18"
    spy_minute_df = ingestor.ingest_historical_data(
        symbol="SPY",
        timeframe="1Min",
        start_date=test_minute_date,
        end_date=test_minute_date,
        overwrite=True,  # Force fresh download
    )
    print(f"\nSPY 1-Minute Data for {test_minute_date} (first 5 rows):")
    print(spy_minute_df.head())
    print(f"Total SPY 1-minute records for {test_minute_date}: {len(spy_minute_df)}")

    # Test 4: Attempt to ingest non-existent symbol
    logger.info("\n--- Test 4: Ingesting Non-Existent Symbol ---")
    non_existent_df = ingestor.ingest_historical_data(
        symbol="INVALIDSTOCK",
        timeframe="1D",
        start_date="2024-01-01",
        end_date="2024-01-05",
    )
    print(f"\nInvalid Stock Data Empty? {non_existent_df.empty}")

    logger.info("--- DataIngestor Test Complete ---")
