import logging

from rich.logging import RichHandler

logger = logging.getLogger("mani_skill")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    ch = RichHandler(rich_tracebacks=True)
    ch.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(ch)
