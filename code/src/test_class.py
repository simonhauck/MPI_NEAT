from loguru import logger


class TestClass:

    def add_class(self, a, b):
        logger.debug("TestClass {}, {}".format(a, b))
        return a + b
