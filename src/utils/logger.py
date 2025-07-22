import inspect
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
                    Logger._logger.info(self._no_fd_logger)
            except Exception as e:
                Logger._logger.warning(e)
                Logger._logger.info(self._no_fd_logger)

    def _get_context(self) -> str:
        # Skip the logger methods and try to find the real caller
        for frame_info in inspect.stack()[2:]:
            caller = frame_info.frame.f_locals.get("self")

            if caller:
                return f"[{caller.__class__.__name__}]"
            # fallback to function name in non-class contexts
            if "function" in frame_info and frame_info.function != "<module>":
                return f"[{frame_info.function}]"
        return "[\\m/]"

    def debug(self, msg: str):
        context = self._get_context()
        Logger._logger.debug(f"{context} {msg}")

    def info(self, msg: str):
        context = self._get_context()
        Logger._logger.info(f"{context} {msg}")

    def warning(self, msg: str):
        context = self._get_context()
        Logger._logger.warning(f"{context} {msg}")

    def error(self, msg: str):
        context = self._get_context()
        Logger._logger.error(f"{context} {msg}")

    def exception(self, msg: str):
        context = self._get_context()
        Logger._logger.exception(f"{context} {msg}")
