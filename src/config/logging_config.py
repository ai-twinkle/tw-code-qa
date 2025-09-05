"""
Logging configuration for the system
日誌配置
"""

import logging
import logging.config
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from typing_extensions import TypedDict


class LogLevel(Enum):
    """日誌級別"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """日誌格式類型"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"


class HandlerConfig(TypedDict):
    """日誌處理器配置類型"""
    handler_type: str
    level: str
    format_type: str
    filename: Optional[str]
    max_bytes: Optional[int]
    backup_count: Optional[int]
    when: Optional[str]
    interval: Optional[int]
    encoding: Optional[str]


class FormatterConfig(TypedDict):
    """日誌格式器配置類型"""
    format_string: str
    date_format: str
    style: str


class LoggerConfig(TypedDict):
    """日誌器配置類型"""
    level: str
    handlers: List[str]
    propagate: bool


class LoggingSystemConfig(TypedDict):
    """完整日誌系統配置類型"""
    version: int
    disable_existing_loggers: bool
    formatters: Dict[str, FormatterConfig]
    handlers: Dict[str, HandlerConfig]
    loggers: Dict[str, LoggerConfig]
    root: LoggerConfig


# 日誌格式定義
LOG_FORMATS: Dict[str, FormatterConfig] = {
    "simple": {
        "format_string": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "style": "%"
    },
    "detailed": {
        "format_string": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "style": "%"
    },
    "json": {
        "format_string": '{"timestamp":"%(asctime)s","logger":"%(name)s","level":"%(levelname)s","file":"%(filename)s","line":%(lineno)d,"function":"%(funcName)s","message":"%(message)s"}',
        "date_format": "%Y-%m-%dT%H:%M:%S",
        "style": "%"
    },
    "structured": {
        "format_string": "[%(asctime)s] %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "style": "%"
    }
}

# 預設日誌目錄
DEFAULT_LOG_DIR = Path("logs")

# 確保日誌目錄存在
def ensure_log_directory(log_dir: Union[str, Path] = DEFAULT_LOG_DIR) -> Path:
    """確保日誌目錄存在"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    return log_path

# 日誌處理器配置
def get_handler_configs(log_dir: Union[str, Path] = DEFAULT_LOG_DIR) -> Dict[str, HandlerConfig]:
    """獲取日誌處理器配置"""
    log_path = ensure_log_directory(log_dir)
    
    return {
        "console": {
            "handler_type": "StreamHandler",
            "level": LogLevel.INFO.value,
            "format_type": "simple",
            "filename": None,
            "max_bytes": None,
            "backup_count": None,
            "when": None,
            "interval": None,
            "encoding": None
        },
        "file_info": {
            "handler_type": "RotatingFileHandler",
            "level": LogLevel.INFO.value,
            "format_type": "detailed",
            "filename": str(log_path / "application.log"),
            "max_bytes": 10485760,  # 10MB
            "backup_count": 5,
            "when": None,
            "interval": None,
            "encoding": "utf-8"
        },
        "file_error": {
            "handler_type": "RotatingFileHandler",
            "level": LogLevel.ERROR.value,
            "format_type": "detailed",
            "filename": str(log_path / "error.log"),
            "max_bytes": 10485760,  # 10MB
            "backup_count": 3,
            "when": None,
            "interval": None,
            "encoding": "utf-8"
        },
        "file_debug": {
            "handler_type": "RotatingFileHandler",
            "level": LogLevel.DEBUG.value,
            "format_type": "structured",
            "filename": str(log_path / "debug.log"),
            "max_bytes": 20971520,  # 20MB
            "backup_count": 2,
            "when": None,
            "interval": None,
            "encoding": "utf-8"
        },
        "agent_workflow": {
            "handler_type": "TimedRotatingFileHandler",
            "level": LogLevel.INFO.value,
            "format_type": "json",
            "filename": str(log_path / "agent_workflow.log"),
            "max_bytes": None,
            "backup_count": 7,
            "when": "midnight",
            "interval": 1,
            "encoding": "utf-8"
        },
        "performance": {
            "handler_type": "TimedRotatingFileHandler",
            "level": LogLevel.INFO.value,
            "format_type": "json",
            "filename": str(log_path / "performance.log"),
            "max_bytes": None,
            "backup_count": 30,
            "when": "midnight",
            "interval": 1,
            "encoding": "utf-8"
        },
        "quality_assessment": {
            "handler_type": "RotatingFileHandler",
            "level": LogLevel.INFO.value,
            "format_type": "json",
            "filename": str(log_path / "quality_assessment.log"),
            "max_bytes": 52428800,  # 50MB
            "backup_count": 3,
            "when": None,
            "interval": None,
            "encoding": "utf-8"
        }
    }

# 日誌器配置
LOGGER_CONFIGS: Dict[str, LoggerConfig] = {
    "src.workflow": {
        "level": LogLevel.DEBUG.value,
        "handlers": ["console", "file_info", "agent_workflow"],
        "propagate": False
    },
    "src.workflow.nodes": {
        "level": LogLevel.DEBUG.value,
        "handlers": ["console", "file_info", "agent_workflow"],
        "propagate": False
    },
    "src.services": {
        "level": LogLevel.INFO.value,
        "handlers": ["console", "file_info"],
        "propagate": False
    },
    "src.services.llm_service": {
        "level": LogLevel.DEBUG.value,
        "handlers": ["console", "file_info", "performance"],
        "propagate": False
    },
    "src.core": {
        "level": LogLevel.INFO.value,
        "handlers": ["console", "file_info"],
        "propagate": False
    },
    "src.models": {
        "level": LogLevel.WARNING.value,
        "handlers": ["console", "file_info"],
        "propagate": False
    },
    "src.utils": {
        "level": LogLevel.INFO.value,
        "handlers": ["console", "file_info"],
        "propagate": False
    },
    "quality": {
        "level": LogLevel.INFO.value,
        "handlers": ["console", "file_info", "quality_assessment"],
        "propagate": False
    },
    "performance": {
        "level": LogLevel.INFO.value,
        "handlers": ["console", "performance"],
        "propagate": False
    },
    "error": {
        "level": LogLevel.ERROR.value,
        "handlers": ["console", "file_error"],
        "propagate": False
    },
    "security": {
        "level": LogLevel.WARNING.value,
        "handlers": ["console", "file_error"],
        "propagate": False
    }
}

# 獲取完整的日誌配置
def get_logging_config(
    log_dir: Union[str, Path] = DEFAULT_LOG_DIR,
    console_level: LogLevel = LogLevel.INFO,
    file_level: LogLevel = LogLevel.DEBUG,
    debug_mode: bool = False
) -> LoggingSystemConfig:
    """
    獲取完整的日誌系統配置
    
    Args:
        log_dir: 日誌目錄路徑
        console_level: 控制台日誌級別
        file_level: 文件日誌級別
        debug_mode: 是否啟用調試模式
    
    Returns:
        完整的日誌系統配置
    """
    handlers = get_handler_configs(log_dir)
    
    # 如果是調試模式，調整日誌級別
    if debug_mode:
        handlers["console"]["level"] = LogLevel.DEBUG.value
        for logger_config in LOGGER_CONFIGS.values():
            if logger_config["level"] == LogLevel.INFO.value:
                logger_config["level"] = LogLevel.DEBUG.value
    else:
        handlers["console"]["level"] = console_level.value
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": LOG_FORMATS,
        "handlers": handlers,
        "loggers": LOGGER_CONFIGS,
        "root": {
            "level": file_level.value,
            "handlers": ["console", "file_info", "file_error"],
            "propagate": False
        }
    }

# 開發環境配置
def get_development_logging_config() -> LoggingSystemConfig:
    """獲取開發環境的日誌配置"""
    return get_logging_config(
        console_level=LogLevel.DEBUG,
        file_level=LogLevel.DEBUG,
        debug_mode=True
    )

# 生產環境配置
def get_production_logging_config() -> LoggingSystemConfig:
    """獲取生產環境的日誌配置"""
    return get_logging_config(
        console_level=LogLevel.INFO,
        file_level=LogLevel.INFO,
        debug_mode=False
    )

# 測試環境配置
def get_testing_logging_config() -> LoggingSystemConfig:
    """獲取測試環境的日誌配置"""
    return get_logging_config(
        log_dir=Path("logs/test"),
        console_level=LogLevel.WARNING,
        file_level=LogLevel.DEBUG,
        debug_mode=True
    )


def setup_logging(log_level: str = "INFO", verbose: bool = False) -> None:
    """
    設定日誌系統
    
    Args:
        log_level: 日誌級別 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        verbose: 是否啟用詳細模式
    """
    try:
        # 轉換字符串到LogLevel枚舉
        level_enum = LogLevel(log_level.upper())
    except ValueError:
        level_enum = LogLevel.INFO
    
    # 根據verbose設定決定console和debug模式
    console_level = LogLevel.DEBUG if verbose else level_enum
    debug_mode = verbose or level_enum == LogLevel.DEBUG
    
    # 獲取當前環境的日誌配置
    from src.config.settings import get_environment
    environment = get_environment()
    
    if environment == "production":
        config = get_production_logging_config()
    elif environment == "testing":
        config = get_testing_logging_config()
    else:  # development or others
        config = get_development_logging_config()
    
    # 覆蓋console級別
    config["handlers"]["console"]["level"] = console_level.value
    config["root"]["level"] = level_enum.value
    
    # 確保日誌目錄存在
    ensure_log_directory()
    
    # 應用配置
    logging.config.dictConfig(config)


# 特殊用途的日誌器名稱常數
class LoggerNames:
    """日誌器名稱常數"""
    WORKFLOW = "src.workflow"
    NODES = "src.workflow.nodes"
    ANALYZER_DESIGNER = "src.workflow.nodes.analyzer_designer"
    REPRODUCER = "src.workflow.nodes.reproducer"
    EVALUATOR = "src.workflow.nodes.evaluator"
    LLM_SERVICE = "src.services.llm_service"
    DATA_LOADER = "src.services.data_loader"
    DATASET_MANAGER = "src.core.dataset_manager"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    ERROR = "error"
    SECURITY = "security"

# Agent 特定的日誌配置
AGENT_LOG_CONFIGS: Dict[str, Dict[str, Union[str, bool]]] = {
    "analyzer_designer": {
        "logger_name": LoggerNames.ANALYZER_DESIGNER,
        "log_translation_details": True,
        "log_terminology_decisions": True,
        "log_complexity_analysis": True
    },
    "reproducer": {
        "logger_name": LoggerNames.REPRODUCER,
        "log_qa_execution": True,
        "log_reasoning_steps": True,
        "log_comparison_results": False
    },
    "evaluator": {
        "logger_name": LoggerNames.EVALUATOR,
        "log_semantic_analysis": True,
        "log_quality_scores": True,
        "log_improvement_suggestions": True
    }
}

# 日誌過濾器配置
LOG_FILTERS = {
    "sensitive_data": {
        "api_keys": True,
        "user_tokens": True,
        "internal_paths": True
    },
    "performance_threshold": {
        "min_duration_ms": 100,
        "log_slow_operations": True
    },
    "quality_threshold": {
        "min_score_to_log": 5.0,
        "log_all_failures": True
    }
}

# 導出列表
__all__ = [
    # Enums and TypedDict classes
    "LogLevel",
    "LogFormat",
    "HandlerConfig",
    "FormatterConfig",
    "LoggerConfig",
    "LoggingSystemConfig",
    
    # Configuration dictionaries and constants
    "DEFAULT_LOG_DIR",
    "AGENT_LOG_CONFIGS",
    "LOG_FILTERS",
    
    # Functions
    "ensure_log_directory",
    "get_handler_configs",
    "get_logging_config",
    "get_development_logging_config",
    "get_production_logging_config", 
    "get_testing_logging_config",
    "setup_logging",
    
    # Constants class
    "LoggerNames",
]
