import logging

logger = logging.getLogger()

def add(a: int, b: int) -> int:
    logger.debug("Add function called")
    return a + b


def subtract(a: int, b: int) -> int:
    return a - b
