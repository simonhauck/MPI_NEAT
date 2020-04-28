import sys

from loguru import logger

import helper
import test_class

logger.remove()
logger.add(sys.stderr, level=5)

newclass = test_class.TestClass()
newclass.add_class(5, 6)

logger.trace("Trace")
logger.debug("Debug")
logger.info("Info")
logger.warning("Warning")
logger.error("Error")
logger.critical("Critical")
print("Hello")
print(helper.add(1, 2))
