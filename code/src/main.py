import sys

import helper
from loguru import logger

logger.remove()
logger.add(sys.stderr, level=5)


logger.trace("Trace")
logger.debug("Debug")
logger.info("Info")
logger.warning("Warning")
logger.error("Error")
logger.critical("Critical")
print("Hello")
print(helper.add(1, 2))
