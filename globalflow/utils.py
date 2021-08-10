import logging
from contextlib import contextmanager
from typing import final


@contextmanager
def log_level(lev: int = logging.ERROR):
    logger = logging.getLogger("globalflow")
    prev_level = logger.level
    logger.setLevel(lev)
    try:
        yield
    finally:
        logger.setLevel(prev_level)
