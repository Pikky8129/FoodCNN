from common.logger_factory import LoggerFactory


class AppBase:

    def __init__(self):
        self._logger = LoggerFactory.create_logger()

