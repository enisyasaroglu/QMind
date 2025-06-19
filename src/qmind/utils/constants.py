import pytz  # Will need to install this library later
from datetime import time

# --- DATETIME FORMATS ---
DATETIME_FORMAT_ISO = "%Y-%m-%dT%H:%M:%S"  # ISO 8601 format
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"

# --- MARKET / TRADING CONSTANTS ---
# Example: New York Stock Exchange (NYSE) timezone
MARKET_TIMEZONE = pytz.timezone("America/New_York")

# Example: Standard NYSE trading hours (Eastern Time)
MARKET_OPEN_TIME = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM ET

# --- FINANCIAL CONSTANTS ---
DEFAULT_SLIPPAGE_BPS = 2  # Default slippage in basis points (0.02%)
DEFAULT_COMMISSION_BPS = 1  # Default commission in basis points (0.01%)

# --- PROJECT-SPECIFIC CONSTANTS ---
# For example, names of columns in your DataFrames
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
