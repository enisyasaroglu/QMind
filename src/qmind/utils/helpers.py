from datetime import datetime
import pytz
# In /Users/enisyasaroglu/QMind/src/qmind/utils/helpers.py

from datetime import (
    time,
    datetime,
    timezone,
)  # Ensure you also import 'time' and 'timezone' if using them
from qmind.utils.constants import (
    MARKET_TIMEZONE,
    DATETIME_FORMAT_ISO,
)  # Import constants


def format_currency(amount: float, currency_symbol: str = "$") -> str:
    """Formats a float amount as a currency string."""
    return f"{currency_symbol}{amount:,.2f}"


def get_current_timestamp_utc() -> datetime:
    """Returns the current UTC datetime object."""
    return datetime.now(pytz.utc)


def is_market_open(current_dt: datetime) -> bool:
    """
    Checks if the market is currently open based on defined trading hours and timezone.
    Args:
        current_dt: A timezone-aware datetime object (preferably UTC).
    """
    market_dt = current_dt.astimezone(MARKET_TIMEZONE)
    market_time = market_dt.time()
    # Basic check: is it within trading hours on a weekday? (Weekend check not included here for simplicity)
    return (
        MARKET_TIMEZONE.localize(
            datetime(market_dt.year, market_dt.month, market_dt.day)
        ).weekday()
        < 5
        and MARKET_OPEN_TIME <= market_time <= MARKET_CLOSE_TIME
    )


# Define market open and close times in Eastern Time (ET)
# Assuming 9:30 AM ET and 4:00 PM ET for NYSE/NASDAQ
MARKET_OPEN_TIME = time(9, 30, 0)
MARKET_CLOSE_TIME = time(16, 0, 0)
