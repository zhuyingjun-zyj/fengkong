import logging


class Logging():

    def __init__(self, path, stream=False):
        self.path = path
        self.stream = stream

    def create_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # if not os.path.exists(path):
        #     os.mkdir(path)

        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        if self.stream:
            # 设置CMD日志
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel(logging.DEBUG)
            logger.addHandler(sh)

        # 设置文件日志s
        fh = logging.FileHandler(filename=self.path, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        return logger
