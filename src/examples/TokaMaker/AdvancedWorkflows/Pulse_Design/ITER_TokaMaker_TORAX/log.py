"""Shared logging setup for the ITER TokaMaker TORAX workflow scripts."""

import logging
import os


_LOGGER_ROOT_NAME = "iter_tokamaker_torax"
_HANDLER_NAME = "iter_tokamaker_torax_console"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DEFAULT_LOG_LEVEL = "INFO"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a timestamped console logger for this workflow.

    This helper keeps the workflow's logging behavior in one place so training, data generation, and visualization scripts can import the same logger without each script configuring its own formatter or handler. The default INFO level is intentionally conservative for long-running Modal/local jobs, and ITER_TOKAMAKER_LOG_LEVEL can be set when a run needs DEBUG or quieter WARNING output.
    """
    configure_logging()
    if not name or name == "__main__":
        return logging.getLogger(_LOGGER_ROOT_NAME)
    if name.startswith(_LOGGER_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_ROOT_NAME}.{name}")


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Configure the workflow root logger once and return it.

    The handler is owned by the workflow logger rather than the global root logger, which avoids surprising other libraries while still giving every local module a consistent timestamped line format. Repeated calls are safe because the handler is named and reused.
    """
    logger = logging.getLogger(_LOGGER_ROOT_NAME)
    logger.setLevel(_resolve_log_level(level))
    logger.propagate = False
    if not any(handler.get_name() == _HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    return logger


def _resolve_log_level(level: str | int | None) -> int:
    """Resolve an explicit level or ITER_TOKAMAKER_LOG_LEVEL into a logging level integer."""
    if isinstance(level, int):
        return level
    level_name = str(level or os.getenv("ITER_TOKAMAKER_LOG_LEVEL", _DEFAULT_LOG_LEVEL)).upper()
    return getattr(logging, level_name, logging.INFO)
