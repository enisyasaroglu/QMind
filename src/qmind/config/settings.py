import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This searches for .env in the current directory and its parents
# and loads any variables found there into os.environ.
load_dotenv()

# --- GENERAL SETTINGS ---
ENVIRONMENT = os.getenv(
    "ENVIRONMENT", "development"
)  # 'development', 'production', 'backtesting'
LOG_LEVEL = os.getenv(
    "LOG_LEVEL", "INFO"
)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

# --- DATA MANAGEMENT SETTINGS ---
# Base path for storing historical data (e.g., in parquet format)
DATA_STORAGE_PATH = os.getenv("DATA_STORAGE_PATH", "data/")
YAHOO_FINANCE_TICKER = os.getenv("YAHOO_FINANCE_TICKER", "^GSPC")  # Default to S&P 500

# Redis Cache Settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# --- TRADING ACCOUNT / BROKERAGE SETTINGS ---
# These should definitely come from .env for security!
BROKERAGE_API_KEY = os.getenv("BROKERAGE_API_KEY")
BROKERAGE_SECRET_KEY = os.getenv("BROKERAGE_SECRET_KEY")
BROKERAGE_BASE_URL = os.getenv(
    "BROKERAGE_BASE_URL", "https://paper-api.alpaca.markets"
)  # Example for Alpaca Paper Trading
BROKERAGE_ACCOUNT_ID = os.getenv("BROKERAGE_ACCOUNT_ID")

# --- STRATEGY SETTINGS ---
# Example: Common parameters for DRL strategies
DRL_MODEL_PATH = os.getenv("DRL_MODEL_PATH", "models/drl_agent/")
DRL_TRAINING_EPOCHS = int(os.getenv("DRL_TRAINING_EPOCHS", 100))
DRL_LEARNING_RATE = float(os.getenv("DRL_LEARNING_RATE", 0.001))

# --- SIMULATION & BACKTESTING SETTINGS ---
BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2020-01-01")
BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "2023-12-31")
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 100000.0))

# --- UI SETTINGS ---
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8501))

# --- ADD MORE SETTINGS AS YOUR PROJECT GROWS ---
# For example, specific thresholds for indicators, model hyper-parameters,
# notification settings (email, Telegram API keys), etc.
