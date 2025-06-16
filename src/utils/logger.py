import logging
from datetime import datetime

from src.utils.settings import Settings
from src.utils.singleton import Singleton


class Logger(metaclass=Singleton):
    """_summary_

    Args:
        metaclass (_type_, optional): Defaults to Singleton.

    Returns:
        _type_: Logger
    """

    _logger: logging.Logger = None
    _no_fd_logger: str = "Moving on without a file logger"

    def __init__(
        self, name: str = "maze-3d", log_lvl: int = logging.DEBUG
    ) -> None:
        """_summary_

        Args:
            name (str, optional): Logger name. Defaults to 'maze-3d'.
            log_lvl (int, optional): Logger level. Defaults to logging.DEBUG.
        """
        if Logger._logger is None:
            # Grab app settings
            settings = Settings().get_settings()

            # Create the logger instance
            Logger._logger = logging.getLogger(name)
            Logger._logger.setLevel(log_lvl)

            # Define the log format
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)8s | %(process)7d - %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )

            # Create the default console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_lvl)
            console_handler.setFormatter(formatter)

            Logger._logger.addHandler(console_handler)

            try:
                if settings["file_handler"]:
                    # Create a file handler
                    log_file = "{path}/logs/{date}.log".format(
                        path=settings["current_dir"],
                        date=datetime.now().strftime("%Y%m%d"),
                    )
                    file_handler = logging.FileHandler(log_file, "a+", "utf-8")
                    file_handler.setLevel(log_lvl)
                    file_handler.setFormatter(formatter)
                    Logger._logger.addHandler(file_handler)
                else:
                    self.get_logger().info(self._no_fd_logger)
            except Exception as e:
                self.get_logger().warning(e)
                self.get_logger().info(self._no_fd_logger)

    def get_logger(self) -> logging.Logger:
        """_summary_

        Returns:
            logging.Logger: Returns instantiated logger
        """
        return self._logger
