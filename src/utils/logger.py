import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(log_dir / "app.log", rotation="10 MB", level="INFO", mode="a")
    logger.add(log_dir / "errors.log", rotation="5 MB", level="ERROR", mode="a")


__all__ = ["logger", "setup_logging"]
