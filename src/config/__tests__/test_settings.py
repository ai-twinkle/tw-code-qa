"""
Settings configuration tests
設定配置測試模組

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config.settings import (
    Environment,
    LogLevel,
    MemoryManagementConfig,
    ProcessingConfig,
    IOOptimizationConfig,
    CachingConfig,
    PerformanceConfig,
    ApiKeysConfig,
    DataProtectionConfig,
    NetworkConfig,
    SecurityConfig,
    HealthCheckConfig,
    MetricsConfig,
    AlertsConfig,
    MonitoringConfig,
    ValidationConfig,
    TransformationConfig,
    OutputConfig,
    DataProcessingConfig,
    EnvironmentConfig,
    SystemSettings,
    DEFAULT_SETTINGS,
    ENVIRONMENT_CONFIGS,
    SYSTEM_PATHS,
    FEATURE_FLAGS,
    PERFORMANCE_CONFIGS,
    SECURITY_CONFIGS,
    MONITORING_CONFIGS,
    DATA_PROCESSING_CONFIGS,
    CURRENT_ENVIRONMENT,
    get_environment,
    set_environment,
    get_config_for_environment,
    is_development,
    is_production,
)


class TestEnvironmentEnum:
    """測試環境枚舉"""

    def test_environment_values(self) -> None:
        """測試環境值"""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_environment_members(self) -> None:
        """測試環境成員"""
        environments = list(Environment)
        assert len(environments) == 4
        assert Environment.DEVELOPMENT in environments
        assert Environment.TESTING in environments
        assert Environment.STAGING in environments
        assert Environment.PRODUCTION in environments


class TestLogLevelEnum:
    """測試日誌級別枚舉"""

    def test_log_level_values(self) -> None:
        """測試日誌級別值"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"

    def test_log_level_members(self) -> None:
        """測試日誌級別成員"""
        levels = list(LogLevel)
        assert len(levels) == 5
        assert LogLevel.DEBUG in levels
        assert LogLevel.INFO in levels
        assert LogLevel.WARNING in levels
        assert LogLevel.ERROR in levels
        assert LogLevel.CRITICAL in levels


class TestDefaultSettings:
    """測試預設設定"""

    def test_default_settings_structure(self) -> None:
        """測試預設設定結構"""
        assert isinstance(DEFAULT_SETTINGS, dict)
        
        # 檢查必要的鍵
        required_keys = {
            "environment", "debug", "log_level", "max_workers",
            "memory_limit_gb", "temp_dir", "output_dir", "cache_enabled",
            "cache_ttl_hours", "auto_cleanup", "progress_save_interval",
            "progress_log_interval"
        }
        assert set(DEFAULT_SETTINGS.keys()) == required_keys

    def test_default_settings_values(self) -> None:
        """測試預設設定值"""
        assert DEFAULT_SETTINGS["environment"] == Environment.DEVELOPMENT.value
        assert DEFAULT_SETTINGS["debug"] is True
        assert DEFAULT_SETTINGS["log_level"] == LogLevel.INFO.value
        assert DEFAULT_SETTINGS["max_workers"] == 4
        assert DEFAULT_SETTINGS["memory_limit_gb"] == 8
        assert DEFAULT_SETTINGS["temp_dir"] == "temp"
        assert DEFAULT_SETTINGS["output_dir"] == "output"
        assert DEFAULT_SETTINGS["cache_enabled"] is True
        assert DEFAULT_SETTINGS["cache_ttl_hours"] == 24
        assert DEFAULT_SETTINGS["auto_cleanup"] is True
        assert DEFAULT_SETTINGS["progress_save_interval"] == 100
        assert DEFAULT_SETTINGS["progress_log_interval"] == 10

    def test_default_settings_types(self) -> None:
        """測試預設設定類型"""
        assert isinstance(DEFAULT_SETTINGS["environment"], str)
        assert isinstance(DEFAULT_SETTINGS["debug"], bool)
        assert isinstance(DEFAULT_SETTINGS["log_level"], str)
        assert isinstance(DEFAULT_SETTINGS["max_workers"], int)
        assert isinstance(DEFAULT_SETTINGS["memory_limit_gb"], int)
        assert isinstance(DEFAULT_SETTINGS["temp_dir"], str)
        assert isinstance(DEFAULT_SETTINGS["output_dir"], str)
        assert isinstance(DEFAULT_SETTINGS["cache_enabled"], bool)
        assert isinstance(DEFAULT_SETTINGS["cache_ttl_hours"], int)
        assert isinstance(DEFAULT_SETTINGS["auto_cleanup"], bool)
        assert isinstance(DEFAULT_SETTINGS["progress_save_interval"], int)
        assert isinstance(DEFAULT_SETTINGS["progress_log_interval"], int)


class TestEnvironmentConfigs:
    """測試環境配置"""

    def test_environment_configs_structure(self) -> None:
        """測試環境配置結構"""
        assert isinstance(ENVIRONMENT_CONFIGS, dict)
        
        # 檢查是否包含所有環境
        for env in Environment:
            assert env.value in ENVIRONMENT_CONFIGS

    def test_environment_configs_values(self) -> None:
        """測試環境配置值"""
        # 測試開發環境配置
        dev_config = ENVIRONMENT_CONFIGS[Environment.DEVELOPMENT.value]
        assert dev_config["debug"] is True
        assert dev_config["log_level"] == LogLevel.DEBUG.value
        
        # 測試生產環境配置
        prod_config = ENVIRONMENT_CONFIGS[Environment.PRODUCTION.value]
        assert prod_config["debug"] is False
        assert prod_config["log_level"] == LogLevel.WARNING.value  # 實際配置是 WARNING

    def test_environment_configs_keys(self) -> None:
        """測試環境配置鍵"""
        for env_name, config in ENVIRONMENT_CONFIGS.items():
            assert isinstance(config, dict)
            # 檢查基本配置鍵
            required_keys = {"debug", "log_level", "max_workers", "batch_size"}
            assert required_keys.issubset(set(config.keys()))


class TestConfigurationDictionaries:
    """測試配置字典"""

    def test_system_paths_structure(self) -> None:
        """測試系統路徑結構"""
        assert isinstance(SYSTEM_PATHS, dict)
        assert "temp_dir" in SYSTEM_PATHS
        assert "output_dir" in SYSTEM_PATHS
        assert "logs_dir" in SYSTEM_PATHS  # 實際是 logs_dir 而不是 log_dir

    def test_feature_flags_structure(self) -> None:
        """測試功能開關結構"""
        assert isinstance(FEATURE_FLAGS, dict)
        for key, value in FEATURE_FLAGS.items():
            assert isinstance(key, str)
            assert isinstance(value, bool)

    def test_performance_configs_structure(self) -> None:
        """測試性能配置結構"""
        assert isinstance(PERFORMANCE_CONFIGS, dict)
        # PERFORMANCE_CONFIGS 是結構化配置，包含 memory_management, processing 等子配置
        expected_keys = {"memory_management", "processing", "io_optimization", "caching"}
        for key in expected_keys:
            assert key in PERFORMANCE_CONFIGS
            assert isinstance(PERFORMANCE_CONFIGS[key], dict)

    def test_security_configs_structure(self) -> None:
        """測試安全配置結構"""
        assert isinstance(SECURITY_CONFIGS, dict)
        # SECURITY_CONFIGS 是結構化配置，包含 api_keys, data_protection 等子配置
        expected_keys = {"api_keys", "data_protection", "network"}
        for key in expected_keys:
            assert key in SECURITY_CONFIGS
            assert isinstance(SECURITY_CONFIGS[key], dict)

    def test_monitoring_configs_structure(self) -> None:
        """測試監控配置結構"""
        assert isinstance(MONITORING_CONFIGS, dict)
        # MONITORING_CONFIGS 是結構化配置，包含 health_check, metrics 等子配置
        expected_keys = {"health_check", "metrics", "alerts"}
        for key in expected_keys:
            assert key in MONITORING_CONFIGS
            assert isinstance(MONITORING_CONFIGS[key], dict)

    def test_data_processing_configs_structure(self) -> None:
        """測試資料處理配置結構"""
        assert isinstance(DATA_PROCESSING_CONFIGS, dict)
        # DATA_PROCESSING_CONFIGS 是結構化配置，包含 validation, transformation 等子配置
        expected_keys = {"validation", "transformation", "output"}
        for key in expected_keys:
            assert key in DATA_PROCESSING_CONFIGS
            assert isinstance(DATA_PROCESSING_CONFIGS[key], dict)


class TestEnvironmentFunctions:
    """測試環境函數"""

    def test_get_environment(self) -> None:
        """測試取得當前環境"""
        current_env = get_environment()
        assert isinstance(current_env, str)
        assert current_env in [e.value for e in Environment]

    def test_set_environment_valid(self) -> None:
        """測試設定有效環境"""
        original_env = get_environment()
        
        try:
            # 測試設定不同的環境
            for env in Environment:
                set_environment(env.value)
                assert get_environment() == env.value
        finally:
            # 恢復原始環境
            set_environment(original_env)

    def test_set_environment_invalid(self) -> None:
        """測試設定無效環境"""
        with pytest.raises(ValueError) as exc_info:
            set_environment("invalid_environment")
        
        assert "Invalid environment" in str(exc_info.value)
        assert "Must be one of" in str(exc_info.value)

    def test_get_config_for_environment_current(self) -> None:
        """測試取得當前環境配置"""
        config = get_config_for_environment()
        assert isinstance(config, dict)
        
        current_env = get_environment()
        if current_env in ENVIRONMENT_CONFIGS:
            expected_config = ENVIRONMENT_CONFIGS[current_env]
            assert config == expected_config

    def test_get_config_for_environment_specific(self) -> None:
        """測試取得指定環境配置"""
        # 測試已知環境
        for env in Environment:
            config = get_config_for_environment(env.value)
            assert isinstance(config, dict)
            
            if env.value in ENVIRONMENT_CONFIGS:
                expected_config = ENVIRONMENT_CONFIGS[env.value]
                assert config == expected_config

    def test_get_config_for_environment_fallback(self) -> None:
        """測試取得不存在環境的配置（回退到預設）"""
        config = get_config_for_environment("non_existent_env")
        assert config == DEFAULT_SETTINGS

    def test_is_development(self) -> None:
        """測試是否為開發環境"""
        original_env = get_environment()
        
        try:
            # 設定為開發環境
            set_environment(Environment.DEVELOPMENT.value)
            assert is_development() is True
            
            # 設定為非開發環境
            set_environment(Environment.PRODUCTION.value)
            assert is_development() is False
        finally:
            set_environment(original_env)

    def test_is_production(self) -> None:
        """測試是否為生產環境"""
        original_env = get_environment()
        
        try:
            # 設定為生產環境
            set_environment(Environment.PRODUCTION.value)
            assert is_production() is True
            
            # 設定為非生產環境
            set_environment(Environment.DEVELOPMENT.value)
            assert is_production() is False
        finally:
            set_environment(original_env)


class TestTypeDefinitions:
    """測試類型定義"""

    def test_memory_management_config_type(self) -> None:
        """測試記憶體管理配置類型"""
        config: MemoryManagementConfig = {
            "batch_size_auto_adjust": True,
            "memory_threshold_mb": 1024,
            "gc_frequency": 100,
            "memory_warning_threshold": 0.8
        }
        
        assert isinstance(config["batch_size_auto_adjust"], bool)
        assert isinstance(config["memory_threshold_mb"], int)
        assert isinstance(config["gc_frequency"], int)
        assert isinstance(config["memory_warning_threshold"], (int, float))

    def test_processing_config_type(self) -> None:
        """測試處理配置類型"""
        config: ProcessingConfig = {
            "max_concurrent_requests": 10,
            "request_queue_size": 100,
            "worker_pool_size": 4,
            "async_processing": True
        }
        
        assert isinstance(config["max_concurrent_requests"], int)
        assert isinstance(config["request_queue_size"], int)
        assert isinstance(config["worker_pool_size"], int)
        assert isinstance(config["async_processing"], bool)

    def test_io_optimization_config_type(self) -> None:
        """測試IO優化配置類型"""
        config: IOOptimizationConfig = {
            "buffer_size": 8192,
            "read_chunk_size": 1024
        }
        
        assert isinstance(config["buffer_size"], int)
        assert isinstance(config["read_chunk_size"], int)

    def test_system_settings_type(self) -> None:
        """測試系統設定類型"""
        # 使用預設設定作為測試對象
        settings: SystemSettings = DEFAULT_SETTINGS
        
        assert isinstance(settings["environment"], str)
        assert isinstance(settings["debug"], bool)
        assert isinstance(settings["log_level"], str)
        assert isinstance(settings["max_workers"], int)
        assert isinstance(settings["memory_limit_gb"], int)
        assert isinstance(settings["temp_dir"], str)
        assert isinstance(settings["output_dir"], str)
        assert isinstance(settings["cache_enabled"], bool)
        assert isinstance(settings["cache_ttl_hours"], int)
        assert isinstance(settings["auto_cleanup"], bool)
        assert isinstance(settings["progress_save_interval"], int)
        assert isinstance(settings["progress_log_interval"], int)

    def test_environment_config_type(self) -> None:
        """測試環境配置類型"""
        # 使用已知的環境配置進行測試
        for env_name, config in ENVIRONMENT_CONFIGS.items():
            env_config: EnvironmentConfig = config
            
            assert isinstance(env_config["debug"], bool)
            assert isinstance(env_config["log_level"], str)
            assert isinstance(env_config["max_workers"], int)
            assert isinstance(env_config["batch_size"], int)
            assert isinstance(env_config["enable_monitoring"], bool)
            assert isinstance(env_config["save_intermediate_results"], bool)
            assert isinstance(env_config["verbose_logging"], bool)


class TestGlobalVariables:
    """測試全域變數"""

    def test_current_environment_variable(self) -> None:
        """測試當前環境變數"""
        assert isinstance(CURRENT_ENVIRONMENT, str)
        assert CURRENT_ENVIRONMENT in [e.value for e in Environment]

    def test_current_environment_consistency(self) -> None:
        """測試當前環境一致性"""
        # get_environment() 應該返回與 CURRENT_ENVIRONMENT 相同的值
        assert get_environment() == CURRENT_ENVIRONMENT


class TestConfigurationIntegration:
    """測試配置整合"""

    def test_environment_switching_integration(self) -> None:
        """測試環境切換整合"""
        original_env = get_environment()
        
        try:
            for env in Environment:
                set_environment(env.value)
                
                # 確認環境設定正確
                assert get_environment() == env.value
                
                # 確認配置獲取正確
                config = get_config_for_environment()
                if env.value in ENVIRONMENT_CONFIGS:
                    assert config == ENVIRONMENT_CONFIGS[env.value]
                else:
                    assert config == DEFAULT_SETTINGS
                
                # 確認環境檢查函數正確
                assert is_development() == (env == Environment.DEVELOPMENT)
                assert is_production() == (env == Environment.PRODUCTION)
        finally:
            set_environment(original_env)

    def test_configuration_completeness(self) -> None:
        """測試配置完整性"""
        # 確保所有環境都有對應的配置
        for env in Environment:
            config = get_config_for_environment(env.value)
            assert config is not None
            assert isinstance(config, dict)

    def test_configuration_consistency(self) -> None:
        """測試配置一致性"""
        # 檢查所有環境配置是否有一致的結構
        all_configs = list(ENVIRONMENT_CONFIGS.values()) + [DEFAULT_SETTINGS]
        
        if all_configs:
            base_keys = set(all_configs[0].keys())
            for config in all_configs[1:]:
                # 基本配置鍵應該一致（允許額外的鍵）
                common_keys = {"debug", "log_level", "max_workers"}
                assert common_keys.issubset(set(config.keys()))
