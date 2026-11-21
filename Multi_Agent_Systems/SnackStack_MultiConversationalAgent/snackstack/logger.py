"""
Central logging setup for SnackStack.

Every other module should get its logger from here, so all logs share
the same format, level, and output (console + log file).

Usage in any file:
    from snackstack.logger import setup_logger

    logger = setup_logger(__name__)
    logger.info("Order placed")
    logger.error("Something went wrong")
"""

import logging
import os
import sys
from pathlib import Path

# Folder and file where logs are saved (created automatically if missing).
# Anchored to the project root (the folder that contains `snackstack/`),
# so logs always land in the same place no matter where you run the app from.
#   __file__               -> .../snackstack/logger.py
#   .parent                -> .../snackstack/
#   .parent.parent         -> .../SnackStack_MultiConversationalAgent/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "snackstack.log"

# Format of each log line in the FILE (plain text), e.g.:
# 2026-11-21 23:57:46 | INFO    | menu_agent | menu_agent.py:80 | [menu_agent:model] tool_calls=False
# `%(levelname)-7s` pads the level to 7 chars (len("WARNING")) so the first columns line up.
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Console colours (ANSI escape codes — no extra package needed) ──────────
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
LEVEL_STYLES = {
    #                 colour            icon
    logging.DEBUG:    ("\033[36m",      "🔍"),   # cyan
    logging.INFO:     ("\033[32m",      "✅"),   # green
    logging.WARNING:  ("\033[33m",      "⚠️ "),  # yellow
    logging.ERROR:    ("\033[31m",      "❌"),   # red
    logging.CRITICAL: ("\033[1;41m",    "🔥"),   # white on red
}
NAME_COLOUR = "\033[35m"  # magenta for the module / agent name

# Colours only when writing to a real terminal; respect https://no-color.org
USE_COLOUR = sys.stderr.isatty() and not os.getenv("NO_COLOR")


class PlainFormatter(logging.Formatter):
    """File formatter: plain text, no colours (kept for symmetry / future fields)."""


class ColourFormatter(logging.Formatter):
    """Console formatter: same layout as the file, plus colours and a level icon.

    Example:
        2026-11-21 23:57:46 | ✅ INFO    | menu_agent | menu_agent.py:80 | [menu_agent:model] tool_calls=False
    """

    def format(self, record: logging.LogRecord) -> str:
        colour, icon = LEVEL_STYLES.get(record.levelno, ("", "•"))
        date = self.formatTime(record, DATE_FORMAT)
        location = f"{record.filename}:{record.lineno}"
        message = record.getMessage()
        if record.exc_info:  # keep tracebacks from logger.exception(...)
            message += "\n" + self.formatException(record.exc_info)

        if not USE_COLOUR:
            return f"{date} | {icon} {record.levelname:<7} | {record.name} | {location} | {message}"

        sep = f"{DIM}|{RESET}"
        msg_colour = colour if record.levelno >= logging.WARNING else ""
        return (
            f"{DIM}{date}{RESET} {sep} "
            f"{icon} {colour}{BOLD}{record.levelname:<7}{RESET} {sep} "
            f"{NAME_COLOUR}{record.name}{RESET} {sep} "
            f"{DIM}{location}{RESET} {sep} "
            f"{msg_colour}{message}{RESET}"
        )


def setup_logger(name: str = "snackstack", level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to both the console and a log file.

    Args:
        name:  Logger name. Pass __name__ so logs show which module they came from.
        level: Minimum level to record (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logger = logging.getLogger(name)

    # Always apply the requested level, even if the logger already exists,
    # so setup_logger(__name__, logging.DEBUG) works on a second call too.
    logger.setLevel(level)

    # If this logger was already set up, reuse it.
    # This prevents duplicate log lines when a module is imported more than once.
    if logger.handlers:
        return logger
    # Handler 1: print logs to the terminal (coloured, compact).
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColourFormatter())
    logger.addHandler(console_handler)

    # Handler 2: save logs to a file (plain text, full date) so they can be reviewed later.
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(PlainFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    # Don't pass logs up to the root logger (avoids printing them twice).
    logger.propagate = False

    return logger
