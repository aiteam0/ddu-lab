import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_LOGGING_INITIALIZED = False
_VERBOSE_MODE = False

def setup_logging(
    log_level: Optional[str] = None,
    log_to_file: bool = None,
    log_dir: str = None
) -> None:

    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    
    level_str = log_level or os.getenv("LOG_LEVEL", "DEBUG")  
    level = getattr(logging, level_str.upper(), logging.DEBUG)
    
    file_logging = log_to_file if log_to_file is not None else \
                   os.getenv("LOG_TO_FILE", "true").lower() == "true"
    
    log_directory = Path(log_dir or os.getenv("LOG_DIR", "logs"))
    log_directory.mkdir(exist_ok=True)
    
    (log_directory / "archived").mkdir(exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handlers = []
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    if file_logging:
        app_handler = RotatingFileHandler(
            log_directory / "app.log",
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(formatter)
        handlers.append(app_handler)
        
        if level <= logging.DEBUG:
            debug_handler = RotatingFileHandler(
                log_directory / "debug.log",
                maxBytes=50*1024*1024,
                backupCount=3,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(formatter)
            handlers.append(debug_handler)
        
        error_handler = RotatingFileHandler(
            log_directory / "error.log",
            maxBytes=10*1024*1024,
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        handlers.append(error_handler)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    for handler in handlers:
        root_logger.addHandler(handler)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    _LOGGING_INITIALIZED = True

def get_logger(name: str) -> logging.Logger:

    setup_logging()
    logger = logging.getLogger(name)
    
    logger.setLevel(logging.DEBUG)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    for handler in root_logger.handlers:
        if hasattr(handler, 'setLevel'):
            handler.setLevel(logging.DEBUG)
    
    if not hasattr(get_logger, '_first_logger_created'):
        get_logger._first_logger_created = True
        logger.info(f"[LOGGING_INIT] Logger system initialized - Root level: {root_logger.level}, Handlers: {len(root_logger.handlers)}")
        logger.info(f"[LOGGING_INIT] First logger created: {name} - Level: {logger.level}")
    
    return logger

def log_print(message: str, level: str = "INFO", logger_name: str = None) -> None:

    logger = get_logger(logger_name or "print_replacement")
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, message)

def setup_verbose_logging(verbose: bool = True) -> None:

    global _VERBOSE_MODE
    _VERBOSE_MODE = verbose
    
    if verbose:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.INFO)

def is_verbose() -> bool:

    return _VERBOSE_MODE

def init_project_logging():

    setup_logging()
    logger = get_logger("hddu.logging")
    logger.info("H-DDU 로깅 시스템 초기화 완료")
    logger.info(f"로그 디렉토리: {os.getenv('LOG_DIR', 'logs')}")
    logger.info(f"로그 레벨: {os.getenv('LOG_LEVEL', 'INFO')}")
    logger.info(f"파일 로그: {'활성화' if os.getenv('LOG_TO_FILE', 'true').lower() == 'true' else '비활성화'}")

def configure_logging(level: str = None):

    setup_logging(log_level=level)


if os.getenv("AUTO_INIT_LOGGING", "false").lower() == "true":
    init_project_logging() 