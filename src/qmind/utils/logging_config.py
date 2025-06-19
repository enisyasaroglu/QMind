import os
import sys
from loguru import logger
from qmind.config import settings  # Import your settings


def setup_logging():
    """
    Configures the Loguru logger for the QMind trading system.
    Logs to console and a rotating file.
    """
    # Remove default handler to customize loguru's behavior
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        diagnose=True,  # Show traceback for errors
    )

    # Ensure logs directory exists
    log_dir = os.path.join(
        settings.DATA_STORAGE_PATH, "logs"
    )  # Use DATA_STORAGE_PATH for logs
    os.makedirs(log_dir, exist_ok=True)  # Create 'data/logs' if it doesn't exist

    # Add file logger (rotating, compressing)
    logger.add(
        os.path.join(log_dir, "qmind_{time:YYYY-MM-DD}.log"),
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",  # Rotate log file every day
        retention="7 days",  # Keep logs for 7 days
        compression="zip",  # Compress old log files
        enqueue=True,  # Use a queue for non-blocking logging, important in high-performance apps
    )

    # Set default log level for modules that use the standard logging library (e.g., some external libs)
    # loguru will intercept these logs
    import logging

    logging.basicConfig(level=settings.LOG_LEVEL, handlers=[logging.NullHandler()])


# Optional: If you want to get a logger instance in other modules
def get_logger(name: str):
    """Returns a named logger instance."""
    return logger.bind(name=name)


# When this module is imported, logging will be set up automatically
setup_logging()
