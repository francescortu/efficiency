import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Literal, Optional
from rich.logging import RichHandler
from pathlib import Path


class LambdaLogger():
    @staticmethod
    def log(message: str, level: str = "INFO"):
        """
        Log a message to AWS CloudWatch Logs.
        args:
        message (str): The message to log.
        level (str): The logging level. Default is INFO.
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().log(log_level, message)
        
    def info(self, message: str):
        return LambdaLogger.log(message, "INFO")
    def warning(self, message: str):
        return LambdaLogger.log(message, "WARNING")
    def error(self, message: str):
        return LambdaLogger.log(message, "ERROR")
    

class Logger:
    """
    Logger class to log messages to a file and optionally to the console using rich for console output.
    """
    def __init__(
        self,
        logname: str,
        level: str = "INFO",
        disable_file: bool = False,
        disable_stdout: bool = False,
        log_file_path: Optional[str] = None,
    ):
        """
        args:
        logname (str): The name of the logger.
        level (str): The logging level. Default is INFO.
        disable_file (bool): If True, the logger will not log to a file. Default is False.
        disable_stdout (bool): If True, the logger will not log to the console. Default is False.
        log_file_path (str): The path to the log file. If None, the log file will be saved as {logname}.log.
        """
        self.logname = logname
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.file_log = log_file_path
        self.maxBytes = 1024 * 1024 * 10  # 10 MB
        self.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        if self.file_log: # if not just stdout
            self.file_logger = self._init_file_logger()
            self.file_logger.disabled = disable_file
        else:
            self.file_logger = None
        self.std_out_logger = self._init_stdout_logger()
        self.std_out_logger.disabled = disable_stdout

    def __call__(
        self,
        msg: str,
        level: Literal["info", "debug", "warning", "error"],
        std_out: bool = False,
    ):
        self.log(msg=msg, level=level, std_out=std_out)

    def _init_file_logger(self):
        logger = logging.getLogger(self.logname)
        logger.setLevel(self.level)
        if not self.file_log:
            self.file_log = f"{self.logname}.log"
        if not any(
            isinstance(handler, RotatingFileHandler) for handler in logger.handlers
        ):
            logging_handler = RotatingFileHandler(
                filename=Path(self.file_log),
                mode="a",
                maxBytes=self.maxBytes,
                backupCount=2,
                encoding=None,
                delay=False,
            )
            logging_handler.setFormatter(logging.Formatter(self.format))
            logger.addHandler(logging_handler)

        return logger

    def _init_stdout_logger(self):
        stdout_logger = logging.getLogger(f"{self.logname}_stdout")
        stdout_logger.setLevel(self.level)

        if not any(
            isinstance(handler, RichHandler)
            for handler in stdout_logger.handlers
        ):
            stdout_handler = RichHandler()
            stdout_handler.setFormatter(logging.Formatter(self.format))
            stdout_logger.addHandler(stdout_handler)

        return stdout_logger

    def log(
        self,
        msg: str,
        level: Literal["info", "debug", "warning", "error"],
        std_out: bool = False,
    ):
        log_method = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }.get(level, logging.INFO)

        if self.file_log and self.file_logger:
            self.file_logger.log(log_method, msg)
        if std_out:
            self.std_out_logger.log(log_method, msg)

    def info(self, msg: str, std_out: bool = False):
        self.log(msg, "info", std_out)
    
    def debug(self, msg: str, std_out: bool = False):
        self.log(msg, "debug", std_out)
        
    def warning(self, msg: str, std_out: bool = False):
        self.log(msg, "warning", std_out)
        
    def error(self, msg: str, std_out: bool = False):
        self.log(msg, "error", std_out)
