import sys,logging
from logging.handlers import RotatingFileHandler


def t1():
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    # 设置默认的level为DEBUG
    # 设置log的格式
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
    # )

    logger.info("debug")

def get_logger(filename=None):
    """
    获取Logger，将log同时打印到屏幕和文件。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter( '%(asctime)s %(filename)s:%(lineno)s [%(levelname)s] %(message)s')

    if filename is not None:
        file_handler = RotatingFileHandler(
            filename, maxBytes=50 * 1024 * 1024, backupCount=1)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

if __name__ == "__main__":
    LOG_FILE="log"
    LOGGER = get_logger()
    #LOGGER = get_logger(LOG_FILE)
    LOGGER.info("test log")