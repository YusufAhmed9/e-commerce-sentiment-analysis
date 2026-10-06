import logging

from pythonjsonlogger.json import JsonFormatter


def init_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:                      
        handler = logging.StreamHandler()
        handler.setFormatter(
            JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False               

    return logger