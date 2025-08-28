"""
Configuration module integration tests
配置模組整合測試

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# 測試所有模組匯入
from src.config import (
    # Settings - Enums
    Environment,
    LogLevel,
    
    # Settings - TypedDict classes
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
    
    # Settings - Configuration dictionaries
    DEFAULT_SETTINGS,
    ENVIRONMENT_CONFIGS,
    SYSTEM_PATHS,
    FEATURE_FLAGS,
    PERFORMANCE_CONFIGS,
    SECURITY_CONFIGS,
    MONITORING_CONFIGS,
    DATA_PROCESSING_CONFIGS,
    CURRENT_ENVIRONMENT,
    
    # Settings - Functions
    get_environment,
    set_environment,
    get_config_for_environment,
    is_development,
    is_production,
    
    # LLM Configuration
    LLMProvider,
    ModelTier,
    RetryConfig,
    RateLimitConfig,
    ModelConfig,
    AgentModelConfig,
    DEFAULT_LLM_CONFIGS,
    AGENT_MODEL_CONFIGS,
    COST_TRACKING_CONFIG,
    MODEL_PRICING,
    QUALITY_THRESHOLDS,
    get_model_config,
    get_agent_config,
    calculate_cost,
    
    # Logging Configuration
    LoggingLogLevel,
    LogFormat,
    HandlerConfig,
    FormatterConfig,
    LoggerConfig,
    LoggingSystemConfig,
    DEFAULT_LOG_DIR,
    AGENT_LOG_CONFIGS,
    LOG_FILTERS,
    ensure_log_directory,
    get_handler_configs,
    get_logging_config,
    get_development_logging_config,
    get_production_logging_config,
    get_testing_logging_config,
    LoggerNames,
)


class TestConfigModuleImports:
    """測試配置模組匯入"""

    def test_settings_enums_import(self) -> None:
        """測試設定枚舉匯入"""
        assert Environment is not None
        assert LogLevel is not None
        
        # 測試枚舉值
        assert hasattr(Environment, 'DEVELOPMENT')
        assert hasattr(LogLevel, 'INFO')

    def test_settings_typeddict_import(self) -> None:
        """測試設定 TypedDict 匯入"""
        typeddict_classes = [
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
        ]
        
        for cls in typeddict_classes:
            assert cls is not None

    def test_settings_configurations_import(self) -> None:
        """測試設定配置匯入"""
        configurations = [
            DEFAULT_SETTINGS,
            ENVIRONMENT_CONFIGS,
            SYSTEM_PATHS,
            FEATURE_FLAGS,
            PERFORMANCE_CONFIGS,
            SECURITY_CONFIGS,
            MONITORING_CONFIGS,
            DATA_PROCESSING_CONFIGS,
            CURRENT_ENVIRONMENT,
        ]
        
        for config in configurations:
            assert config is not None

    def test_settings_functions_import(self) -> None:
        """測試設定函數匯入"""
        functions = [
            get_environment,
            set_environment,
            get_config_for_environment,
            is_development,
            is_production,
        ]
        
        for func in functions:
            assert callable(func)

    def test_llm_enums_import(self) -> None:
        """測試 LLM 枚舉匯入"""
        assert LLMProvider is not None
        assert ModelTier is not None
        
        # 測試枚舉值
        assert hasattr(LLMProvider, 'OPENAI')
        assert hasattr(ModelTier, 'STANDARD')

    def test_llm_typeddict_import(self) -> None:
        """測試 LLM TypedDict 匯入"""
        typeddict_classes = [
            RetryConfig,
            RateLimitConfig,
            ModelConfig,
            AgentModelConfig,
        ]
        
        for cls in typeddict_classes:
            assert cls is not None

    def test_llm_configurations_import(self) -> None:
        """測試 LLM 配置匯入"""
        configurations = [
            DEFAULT_LLM_CONFIGS,
            AGENT_MODEL_CONFIGS,
            COST_TRACKING_CONFIG,
            MODEL_PRICING,
            QUALITY_THRESHOLDS,
        ]
        
        for config in configurations:
            assert config is not None

    def test_llm_functions_import(self) -> None:
        """測試 LLM 函數匯入"""
        functions = [
            get_model_config,
            get_agent_config,
            calculate_cost,
        ]
        
        for func in functions:
            assert callable(func)

    def test_logging_enums_import(self) -> None:
        """測試日誌枚舉匯入"""
        assert LoggingLogLevel is not None
        assert LogFormat is not None
        
        # 測試枚舉值
        assert hasattr(LoggingLogLevel, 'INFO')
        assert hasattr(LogFormat, 'SIMPLE')

    def test_logging_typeddict_import(self) -> None:
        """測試日誌 TypedDict 匯入"""
        typeddict_classes = [
            HandlerConfig,
            FormatterConfig,
            LoggerConfig,
            LoggingSystemConfig,
        ]
        
        for cls in typeddict_classes:
            assert cls is not None

    def test_logging_configurations_import(self) -> None:
        """測試日誌配置匯入"""
        configurations = [
            DEFAULT_LOG_DIR,
            AGENT_LOG_CONFIGS,
            LOG_FILTERS,
        ]
        
        for config in configurations:
            assert config is not None

    def test_logging_functions_import(self) -> None:
        """測試日誌函數匯入"""
        functions = [
            ensure_log_directory,
            get_handler_configs,
            get_logging_config,
            get_development_logging_config,
            get_production_logging_config,
            get_testing_logging_config,
        ]
        
        for func in functions:
            assert callable(func)

    def test_logging_constants_import(self) -> None:
        """測試日誌常數匯入"""
        assert LoggerNames is not None
        assert hasattr(LoggerNames, 'WORKFLOW')


class TestConfigModuleIntegration:
    """測試配置模組整合"""

    def test_environment_logging_integration(self) -> None:
        """測試環境與日誌整合"""
        original_env = get_environment()
        
        try:
            # 設定開發環境
            set_environment(Environment.DEVELOPMENT.value)
            dev_logging_config = get_development_logging_config()
            
            # 設定生產環境
            set_environment(Environment.PRODUCTION.value)
            prod_logging_config = get_production_logging_config()
            
            # 驗證不同環境有不同的日誌配置
            assert dev_logging_config != prod_logging_config
            
        finally:
            set_environment(original_env)

    def test_llm_model_pricing_integration(self) -> None:
        """測試 LLM 模型與價格整合"""
        # 檢查是否有模型配置與價格配置的對應關係
        for model_name in DEFAULT_LLM_CONFIGS.keys():
            model_config = get_model_config(model_name)
            assert model_config is not None
            
            # 如果有價格配置，計算成本應該有效
            if model_name in MODEL_PRICING:
                cost = calculate_cost(model_name, 1000, 1000)
                assert cost > 0

    def test_settings_environment_integration(self) -> None:
        """測試設定與環境整合"""
        # 檢查當前環境是否在環境配置中
        current_env = get_environment()
        env_config = get_config_for_environment(current_env)
        
        assert env_config is not None
        assert isinstance(env_config, dict)

    def test_cross_module_enum_consistency(self) -> None:
        """測試跨模組枚舉一致性"""
        # 檢查日誌級別枚舉的一致性
        settings_log_levels = [level.value for level in LogLevel]
        logging_log_levels = [level.value for level in LoggingLogLevel]
        
        # 應該有共同的日誌級別
        common_levels = set(settings_log_levels) & set(logging_log_levels)
        assert len(common_levels) > 0

    def test_configuration_types_consistency(self) -> None:
        """測試配置類型一致性"""
        # 檢查系統設定中的日誌級別是否有效
        log_level = DEFAULT_SETTINGS["log_level"]
        assert log_level in [level.value for level in LogLevel]

    def test_all_imports_accessible(self) -> None:
        """測試所有匯入都可訪問"""
        # 檢查所有匯入的項目都不是 None
        import src.config as config_module
        
        all_exports = config_module.__all__
        
        for export_name in all_exports:
            assert hasattr(config_module, export_name)
            export_value = getattr(config_module, export_name)
            assert export_value is not None


class TestConfigModuleFunctionality:
    """測試配置模組功能性"""

    def test_environment_switching_functionality(self) -> None:
        """測試環境切換功能"""
        original_env = get_environment()
        
        try:
            # 測試切換到不同環境
            for env in Environment:
                set_environment(env.value)
                assert get_environment() == env.value
                
                # 測試環境檢查函數
                assert is_development() == (env == Environment.DEVELOPMENT)
                assert is_production() == (env == Environment.PRODUCTION)
                
        finally:
            set_environment(original_env)

    def test_logging_configuration_functionality(self) -> None:
        """測試日誌配置功能"""
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 測試日誌目錄創建
            log_dir = ensure_log_directory(temp_dir)
            assert log_dir.exists()
            
            # 測試處理器配置
            handlers = get_handler_configs(temp_dir)
            assert isinstance(handlers, dict)
            assert len(handlers) > 0
            
            # 測試完整日誌配置
            config = get_logging_config(log_dir=temp_dir)
            assert isinstance(config, dict)
            assert "handlers" in config
            assert "formatters" in config

    def test_llm_configuration_functionality(self) -> None:
        """測試 LLM 配置功能"""
        # 測試模型配置獲取
        for model_name in DEFAULT_LLM_CONFIGS.keys():
            config = get_model_config(model_name)
            assert config is not None
            assert config["model_name"] == model_name
        
        # 測試成本計算
        for model_name in MODEL_PRICING.keys():
            cost = calculate_cost(model_name, 1000, 1000)
            assert cost > 0
            assert isinstance(cost, float)

    def test_configuration_completeness(self) -> None:
        """測試配置完整性"""
        # 檢查默認設定的完整性
        assert isinstance(DEFAULT_SETTINGS, dict)
        assert len(DEFAULT_SETTINGS) > 0
        
        # 檢查環境配置的完整性
        assert isinstance(ENVIRONMENT_CONFIGS, dict)
        for env in Environment:
            assert env.value in ENVIRONMENT_CONFIGS
        
        # 檢查 LLM 配置的完整性
        assert isinstance(DEFAULT_LLM_CONFIGS, dict)
        assert len(DEFAULT_LLM_CONFIGS) > 0
        
        # 檢查日誌配置的完整性
        dev_config = get_development_logging_config()
        prod_config = get_production_logging_config()
        test_config = get_testing_logging_config()
        
        for config in [dev_config, prod_config, test_config]:
            assert isinstance(config, dict)
            assert "handlers" in config
            assert "formatters" in config


class TestConfigModuleErrorHandling:
    """測試配置模組錯誤處理"""

    def test_invalid_environment_handling(self) -> None:
        """測試無效環境處理"""
        with pytest.raises(ValueError):
            set_environment("invalid_environment")

    def test_non_existent_model_handling(self) -> None:
        """測試不存在模型處理"""
        config = get_model_config("non_existent_model")
        assert config is None
        
        cost = calculate_cost("non_existent_model", 1000, 1000)
        assert cost == 0.0

    def test_non_existent_agent_handling(self) -> None:
        """測試不存在 Agent 處理"""
        config = get_agent_config("non_existent_agent")
        assert config is None

    def test_fallback_configuration_handling(self) -> None:
        """測試回退配置處理"""
        # 測試不存在環境的配置回退
        config = get_config_for_environment("non_existent_env")
        assert config == DEFAULT_SETTINGS


class TestConfigModulePerformance:
    """測試配置模組性能"""

    def test_configuration_loading_performance(self) -> None:
        """測試配置加載性能"""
        import time
        
        # 測試環境配置加載時間
        start_time = time.time()
        for _ in range(100):
            get_config_for_environment()
        end_time = time.time()
        
        # 100 次調用應該在合理時間內完成
        assert (end_time - start_time) < 1.0  # 1 秒

    def test_model_config_loading_performance(self) -> None:
        """測試模型配置加載性能"""
        import time
        
        start_time = time.time()
        for _ in range(100):
            for model_name in DEFAULT_LLM_CONFIGS.keys():
                get_model_config(model_name)
        end_time = time.time()
        
        # 應該在合理時間內完成
        assert (end_time - start_time) < 1.0  # 1 秒

    def test_cost_calculation_performance(self) -> None:
        """測試成本計算性能"""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            calculate_cost("gpt-4o", 1000, 1000)
        end_time = time.time()
        
        # 1000 次計算應該很快
        assert (end_time - start_time) < 1.0  # 1 秒


class TestConfigModuleConstants:
    """測試配置模組常數"""

    def test_default_values_reasonable(self) -> None:
        """測試預設值合理性"""
        # 檢查系統設定預設值
        assert DEFAULT_SETTINGS["max_workers"] > 0
        assert DEFAULT_SETTINGS["memory_limit_gb"] > 0
        assert DEFAULT_SETTINGS["cache_ttl_hours"] > 0
        
        # 檢查成本追蹤預設值
        assert COST_TRACKING_CONFIG["daily_budget_usd"] > 0
        assert 0 <= COST_TRACKING_CONFIG["alert_threshold"] <= 1

    def test_configuration_dictionaries_structure(self) -> None:
        """測試配置字典結構"""
        configs_to_test = [
            (SYSTEM_PATHS, str),
            (FEATURE_FLAGS, bool),
            (PERFORMANCE_CONFIGS, dict),
            (SECURITY_CONFIGS, dict),
            (MONITORING_CONFIGS, dict),
            (DATA_PROCESSING_CONFIGS, dict),
        ]
        
        for config_dict, expected_value_type in configs_to_test:
            assert isinstance(config_dict, dict)
            for key, value in config_dict.items():
                assert isinstance(key, str)
                if expected_value_type != dict:
                    assert isinstance(value, expected_value_type)

    def test_model_configurations_validity(self) -> None:
        """測試模型配置有效性"""
        for model_name, config in DEFAULT_LLM_CONFIGS.items():
            # 檢查必要字段
            assert config["provider"] in [p.value for p in LLMProvider]
            assert 0 <= config["temperature"] <= 2.0
            assert config["max_tokens"] > 0
            assert 0 <= config["top_p"] <= 1.0
            assert config["timeout_seconds"] > 0
            
            # 檢查重試配置
            retry_config = config["retry_config"]
            assert retry_config["max_retries"] >= 0
            assert retry_config["initial_delay"] > 0
            assert retry_config["max_delay"] >= retry_config["initial_delay"]
            
            # 檢查速率限制配置
            rate_limit = config["rate_limit"]
            assert rate_limit["requests_per_minute"] > 0
            assert rate_limit["tokens_per_minute"] > 0
            assert rate_limit["concurrent_requests"] > 0
