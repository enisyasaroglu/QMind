import os
import pandas as pd
from datetime import datetime
from alpaca_trade_api.rest import (
    REST,
    TimeFrame,
)  # Import Alpaca's API client and TimeFrame enum

from qmind.config import settings
from qmind.utils.logging_config import get_logger
from qmind.utils.constants import OHLCV_COLS, DATETIME_FORMAT_ISO

logger = get_logger(__name__)


class AlpacaDataClient:
    """
    A client to interact with Alpaca's historical and potentially real-time data APIs.
    """

    def __init__(self):
        # Initialize the Alpaca REST API client using credentials from settings
        self.api = REST(
            key_id=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            base_url=settings.ALPACA_BASE_URL,
            api_version="v2",  # Specify API version
        )
        logger.info("AlpacaDataClient initialized.")
        # Basic check to ensure API keys are loaded
        if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
            logger.error(
                "Alpaca API Key ID or Secret Key not found in .env or settings. Exiting."
            )
            raise ValueError(
                "Alpaca API credentials missing. Please check your .env file."
            )

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1D",  # Example: "1D", "1H", "1Min"
        start_date: str = None,  # YYYY-MM-DD
        end_date: str = None,  # YYYY-MM-DD
        limit: int = None,  # Max number of bars
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV bar data for a given symbol from Alpaca.

        Args:
            symbol (str): The stock ticker symbol (e.g., "AAPL").
            timeframe (str): The time aggregation of the bars (e.g., "1D", "1H", "1Min").
                             Currently supports "1Min", "5Min", "15Min", "1H", "1D".
            start_date (str, optional): Start date in YYYY-MM-DD format. Defaults to BACKTEST_START_DATE.
            end_date (str, optional): End date in YYYY-MM-DD format. Defaults to BACKTEST_END_DATE.
            limit (int, optional): The maximum number of bars to return. Defaults to None (Alpaca max).

        Returns:
            pd.DataFrame: A DataFrame with OHLCV data, indexed by datetime.
                          Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        try:
            # Convert string timeframe to Alpaca's TimeFrame object
            if timeframe.upper() == "1MIN":
                tf = TimeFrame.Minute
            elif timeframe.upper() == "5MIN":
                tf = TimeFrame.Minute
                # Alpaca's get_bars might interpret TimeFrame.Minute differently for 5min/15min
                # For specific multi-minute bars, you might need to fetch 1Min and resample.
                # For simplicity, let's use the core TimeFrame objects initially.
                logger.warning(
                    "For 5Min/15Min, Alpaca's TimeFrame.Minute might fetch 1Min bars. Consider resampling manually if needed."
                )
            elif timeframe.upper() == "15MIN":
                tf = TimeFrame.Minute
                logger.warning(
                    "For 5Min/15Min, Alpaca's TimeFrame.Minute might fetch 1Min bars. Consider resampling manually if needed."
                )
            elif timeframe.upper() == "1H":
                tf = TimeFrame.Hour
            elif timeframe.upper() == "1D":
                tf = TimeFrame.Day
            else:
                logger.error(f"Unsupported timeframe: {timeframe}. Defaulting to 1Day.")
                tf = TimeFrame.Day

            # Use dates from settings if not provided
            _start_date = start_date if start_date else settings.BACKTEST_START_DATE
            _end_date = end_date if end_date else settings.BACKTEST_END_DATE

            logger.info(
                f"Fetching {timeframe} bars for {symbol} from {_start_date} to {_end_date}"
            )

            # Fetch bars from Alpaca
            # Alpaca's get_bars returns a list of Bar objects
            bars = self.api.get_bars(
                symbol=symbol,
                timeframe=tf,
                start=_start_date,
                end=_end_date,
                limit=limit,
                adjustment="raw",  # 'raw', 'split', 'dividend'
            ).df  # Use .df to directly convert to pandas DataFrame

            if bars.empty:
                logger.warning(
                    f"No {timeframe} bars found for {symbol} in the given date range."
                )
                return pd.DataFrame(columns=OHLCV_COLS)

            # Alpaca DataFrame columns are typically ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            # Rename to our standard OHLCV_COLS if necessary (they are often lowercase by default)
            bars.columns = (
                bars.columns.str.capitalize()
            )  # Capitalize first letter (Open, High, etc.)
            bars = bars[OHLCV_COLS]  # Select only the columns we need

            # Set index to datetime and ensure it's timezone-aware (Alpaca returns UTC by default)
            bars.index = pd.to_datetime(bars.index, utc=True)
            logger.info(
                f"Successfully fetched {len(bars)} {timeframe} bars for {symbol}."
            )
            return bars

        except Exception as e:
            logger.error(
                f"Error fetching historical bars for {symbol}: {e}", exc_info=True
            )
            return pd.DataFrame(columns=OHLCV_COLS)


# Example of how you might use this (for direct testing, not typical import)
if __name__ == "__main__":
    # Ensure you have your .env file correctly set up with Alpaca keys!
    # You can temporarily add `load_dotenv()` here for quick testing if needed,
    # but in a real app, it's loaded by settings.py already when imported.

    # Test fetching daily data for AAPL
    alpaca_client = AlpacaDataClient()
    aapl_daily_bars = alpaca_client.get_historical_bars(
        symbol="AAPL",
        timeframe="1D",
        start_date="2024-01-01",  # Recent data for quick check
        end_date="2024-06-01",
    )
    print("\nAAPL Daily Bars (first 5 rows):")
    print(aapl_daily_bars.head())
    print(f"Total AAPL daily bars: {len(aapl_daily_bars)}")

    # Test fetching minute data for a short period
    spy_minute_bars = alpaca_client.get_historical_bars(
        symbol="SPY",
        timeframe="1Min",
        start_date="2025-06-19",  # Yesterday's data as per current time
        end_date="2025-06-19",
    )
    print("\nSPY 1-Minute Bars (first 5 rows):")
    print(spy_minute_bars.head())
    print(f"Total SPY 1-minute bars: {len(spy_minute_bars)}")

    # Test with an invalid symbol
    invalid_bars = alpaca_client.get_historical_bars(
        symbol="NONEXISTENTSTOCK",
        timeframe="1D",
        start_date="2024-01-01",
        end_date="2024-01-05",
    )
    print(f"\nInvalid symbol bars empty? {invalid_bars.empty}")
