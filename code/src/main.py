from loguru import logger

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time}</green> | {level}   | <level>{message}</level>",
           level="INFO")
