"""
System settings and configurations
系統設定與配置
"""

from enum import Enum
from pathlib import Path
from typing import Dict, Union, Optional

from typing_extensions import TypedDict


class Environment(Enum):
    """環境類型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """日誌級別"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# 類型定義
class MemoryManagementConfig(TypedDict):
    """記憶體管理配置類型"""
    batch_size_auto_adjust: bool
    memory_threshold_mb: int
    gc_frequency: int
    memory_warning_threshold: float


class ProcessingConfig(TypedDict):
    """處理配置類型"""
    max_concurrent_requests: int
    request_queue_size: int
    worker_pool_size: int
    async_processing: bool


class IOOptimizationConfig(TypedDict):
    """IO 優化配置類型"""
    buffer_size: int
    read_chunk_size: int
    write_buffer_size: int
    compression_enabled: bool
    compression_level: int


class CachingConfig(TypedDict):
    """快取配置類型"""
    cache_size_mb: int
    cache_ttl_seconds: int
    cache_cleanup_interval: int
    persistent_cache: bool


class PerformanceConfig(TypedDict):
    """效能配置類型"""
    memory_management: MemoryManagementConfig
    processing: ProcessingConfig
    io_optimization: IOOptimizationConfig
    caching: CachingConfig


class ApiKeysConfig(TypedDict):
    """API 金鑰配置類型"""
    encryption_enabled: bool
    key_rotation_days: int
    mask_in_logs: bool


class DataProtectionConfig(TypedDict):
    """資料保護配置類型"""
    sanitize_outputs: bool
    remove_sensitive_info: bool
    audit_trail: bool


class NetworkConfig(TypedDict):
    """網路配置類型"""
    ssl_verify: bool
    timeout_seconds: int
    max_redirects: int
    user_agent: str


class SecurityConfig(TypedDict):
    """安全配置類型"""
    api_keys: ApiKeysConfig
    data_protection: DataProtectionConfig
    network: NetworkConfig


class HealthCheckConfig(TypedDict):
    """健康檢查配置類型"""
    interval_seconds: int
    timeout_seconds: int
    retry_attempts: int


class MetricsConfig(TypedDict):
    """指標配置類型"""
    collection_interval: int
    retention_days: int
    export_format: str


class AlertsConfig(TypedDict):
    """告警配置類型"""
    error_rate_threshold: float
    response_time_threshold: int
    memory_usage_threshold: float
    disk_usage_threshold: float


class MonitoringConfig(TypedDict):
    """監控配置類型"""
    health_check: HealthCheckConfig
    metrics: MetricsConfig
    alerts: AlertsConfig


class ValidationConfig(TypedDict):
    """驗證配置類型"""
    strict_schema_validation: bool
    skip_invalid_records: bool
    max_error_rate: float


class TransformationConfig(TypedDict):
    """轉換配置類型"""
    preserve_original_format: bool
    normalize_encoding: bool
    remove_duplicates: bool


class OutputConfig(TypedDict):
    """輸出配置類型"""
    format: str
    compression: str
    split_large_files: bool
    max_file_size_mb: int


class DataProcessingConfig(TypedDict):
    """資料處理配置類型"""
    validation: ValidationConfig
    transformation: TransformationConfig
    output: OutputConfig


class EnvironmentConfig(TypedDict):
    """環境配置類型"""
    debug: bool
    log_level: str
    max_workers: int
    batch_size: int
    enable_monitoring: bool
    save_intermediate_results: bool
    verbose_logging: bool


class SystemSettings(TypedDict):
    """系統設定類型"""
    environment: str
    debug: bool
    log_level: str
    max_workers: int
    memory_limit_gb: int
    temp_dir: str
    output_dir: str
    cache_enabled: bool
    cache_ttl_hours: int
    auto_cleanup: bool
    progress_save_interval: int
    progress_log_interval: int


# 預設系統設定
DEFAULT_SETTINGS: SystemSettings = {
    "environment": Environment.DEVELOPMENT.value,
    "debug": True,
    "log_level": LogLevel.INFO.value,
    "max_workers": 4,
    "memory_limit_gb": 8,
    "temp_dir": "temp",
    "output_dir": "output",
    "cache_enabled": True,
    "cache_ttl_hours": 24,
    "auto_cleanup": True,
    "progress_save_interval": 100,
    "progress_log_interval": 10,
}

# 環境配置
ENVIRONMENT_CONFIGS: Dict[str, EnvironmentConfig] = {
    "development": {
        "debug": True,
        "log_level": LogLevel.DEBUG.value,
        "max_workers": 2,
        "batch_size": 10,
        "enable_monitoring": True,
        "save_intermediate_results": True,
        "verbose_logging": True,
    },
    "testing": {
        "debug": True,
        "log_level": LogLevel.WARNING.value,
        "max_workers": 1,
        "batch_size": 5,
        "enable_monitoring": False,
        "save_intermediate_results": False,
        "verbose_logging": False,
    },
    "staging": {
        "debug": False,
        "log_level": LogLevel.INFO.value,
        "max_workers": 4,
        "batch_size": 50,
        "enable_monitoring": True,
        "save_intermediate_results": True,
        "verbose_logging": False,
    },
    "production": {
        "debug": False,
        "log_level": LogLevel.WARNING.value,
        "max_workers": 8,
        "batch_size": 100,
        "enable_monitoring": True,
        "save_intermediate_results": False,
        "verbose_logging": False,
    },
}

# 系統路徑配置
SYSTEM_PATHS: Dict[str, str] = {
    "project_root": str(Path(__file__).parent.parent.parent),
    "data_dir": "data",
    "output_dir": "output",
    "logs_dir": "logs",
    "temp_dir": "temp",
    "cache_dir": "cache",
    "config_dir": "config",
    "scripts_dir": "scripts",
    "docs_dir": "docs",
    "tests_dir": "tests",
}

# 功能開關
FEATURE_FLAGS: Dict[str, bool] = {
    "enable_caching": True,
    "enable_progress_save": True,
    "enable_auto_retry": True,
    "enable_quality_monitoring": True,
    "enable_cost_tracking": True,
    "enable_performance_profiling": False,
    "enable_memory_optimization": True,
    "enable_parallel_processing": True,
    "enable_streaming_output": True,
    "enable_detailed_logging": False,
    "enable_metrics_collection": True,
    "enable_error_recovery": True,
}

# 效能配置
PERFORMANCE_CONFIGS: PerformanceConfig = {
    "memory_management": {
        "batch_size_auto_adjust": True,
        "memory_threshold_mb": 4096,
        "gc_frequency": 100,  # 每處理 N 筆記錄執行垃圾回收
        "memory_warning_threshold": 0.8,  # 記憶體使用警告閾值 (80%)
    },
    "processing": {
        "max_concurrent_requests": 10,
        "request_queue_size": 100,
        "worker_pool_size": 4,
        "async_processing": True,
    },
    "io_optimization": {
        "buffer_size": 8192,
        "read_chunk_size": 1024 * 1024,  # 1MB
        "write_buffer_size": 65536,
        "compression_enabled": True,
        "compression_level": 6,
    },
    "caching": {
        "cache_size_mb": 512,
        "cache_ttl_seconds": 3600,
        "cache_cleanup_interval": 300,
        "persistent_cache": True,
    },
}

# 安全配置
SECURITY_CONFIGS: SecurityConfig = {
    "api_keys": {
        "encryption_enabled": False,  # 生產環境應啟用
        "key_rotation_days": 90,
        "mask_in_logs": True,
    },
    "data_protection": {
        "sanitize_outputs": True,
        "remove_sensitive_info": True,
        "audit_trail": True,
    },
    "network": {
        "ssl_verify": True,
        "timeout_seconds": 30,
        "max_redirects": 3,
        "user_agent": "tw-code-qa/1.0",
    },
}

# 監控與告警配置
MONITORING_CONFIGS: MonitoringConfig = {
    "health_check": {
        "interval_seconds": 60,
        "timeout_seconds": 10,
        "retry_attempts": 3,
    },
    "metrics": {
        "collection_interval": 30,
        "retention_days": 30,
        "export_format": "json",
    },
    "alerts": {
        "error_rate_threshold": 0.05,  # 5%
        "response_time_threshold": 120,  # 120 秒
        "memory_usage_threshold": 0.9,  # 90%
        "disk_usage_threshold": 0.8,   # 80%
    },
}

# 資料處理配置
DATA_PROCESSING_CONFIGS: DataProcessingConfig = {
    "validation": {
        "strict_schema_validation": True,
        "skip_invalid_records": False,
        "max_error_rate": 0.1,  # 10%
    },
    "transformation": {
        "preserve_original_format": True,
        "normalize_encoding": True,
        "remove_duplicates": True,
    },
    "output": {
        "format": "jsonl",
        "compression": "gzip",
        "split_large_files": True,
        "max_file_size_mb": 100,
    },
}

# 當前環境設定 (透過配置文件設定，而非環境變數)
CURRENT_ENVIRONMENT = Environment.DEVELOPMENT.value

def get_environment() -> str:
    """取得當前環境"""
    return CURRENT_ENVIRONMENT

def set_environment(env: str) -> None:
    """設定當前環境"""
    global CURRENT_ENVIRONMENT
    if env in [e.value for e in Environment]:
        CURRENT_ENVIRONMENT = env
    else:
        raise ValueError(f"Invalid environment: {env}. Must be one of {[e.value for e in Environment]}")

def get_config_for_environment(env: Optional[str] = None) -> Union[SystemSettings, EnvironmentConfig]:
    """取得指定環境的配置"""
    if env is None:
        env = get_environment()
    
    if env in ENVIRONMENT_CONFIGS:
        return ENVIRONMENT_CONFIGS[env]
    else:
        return DEFAULT_SETTINGS

def is_development() -> bool:
    """是否為開發環境"""
    return get_environment() == Environment.DEVELOPMENT.value

def is_production() -> bool:
    """是否為生產環境"""
    return get_environment() == Environment.PRODUCTION.value


__all__ = [
    # Enums
    "Environment",
    "LogLevel",
    
    # TypedDict classes
    "MemoryManagementConfig",
    "ProcessingConfig", 
    "IOOptimizationConfig",
    "CachingConfig",
    "PerformanceConfig",
    "ApiKeysConfig",
    "DataProtectionConfig",
    "NetworkConfig",
    "SecurityConfig",
    "HealthCheckConfig",
    "MetricsConfig",
    "AlertsConfig",
    "MonitoringConfig",
    "ValidationConfig",
    "TransformationConfig",
    "OutputConfig",
    "DataProcessingConfig",
    "EnvironmentConfig",
    "SystemSettings",
    
    # Configuration dictionaries
    "DEFAULT_SETTINGS",
    "ENVIRONMENT_CONFIGS",
    "SYSTEM_PATHS",
    "FEATURE_FLAGS",
    "PERFORMANCE_CONFIGS",
    "SECURITY_CONFIGS",
    "MONITORING_CONFIGS",
    "DATA_PROCESSING_CONFIGS",
    "CURRENT_ENVIRONMENT",
    
    # Functions
    "get_environment",
    "set_environment",
    "get_config_for_environment",
    "is_development",
    "is_production",
]
