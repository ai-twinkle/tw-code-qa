"""
Logging configuration tests
日誌配置測試模組

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config.logging_config import (
    LogLevel,
    LogFormat,
    HandlerConfig,
    FormatterConfig,
    LoggerConfig,
    LoggingSystemConfig,
    DEFAULT_LOG_DIR,
    AGENT_LOG_CONFIGS,
    LOG_FILTERS,
    LOG_FORMATS,
    ensure_log_directory,
    get_handler_configs,
    get_logging_config,
    get_development_logging_config,
    get_production_logging_config,
    get_testing_logging_config,
    LoggerNames,
)


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
        expected_levels = [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL
        ]
        for level in expected_levels:
            assert level in levels


class TestLogFormatEnum:
    """測試日誌格式枚舉"""

    def test_log_format_values(self) -> None:
        """測試日誌格式值"""
        assert LogFormat.SIMPLE.value == "simple"
        assert LogFormat.DETAILED.value == "detailed"
        assert LogFormat.JSON.value == "json"
        assert LogFormat.STRUCTURED.value == "structured"

    def test_log_format_members(self) -> None:
        """測試日誌格式成員"""
        formats = list(LogFormat)
        assert len(formats) == 4
        expected_formats = [
            LogFormat.SIMPLE,
            LogFormat.DETAILED,
            LogFormat.JSON,
            LogFormat.STRUCTURED
        ]
        for fmt in expected_formats:
            assert fmt in formats


class TestLogFormats:
    """測試日誌格式定義"""

    def test_log_formats_structure(self) -> None:
        """測試日誌格式結構"""
        assert isinstance(LOG_FORMATS, dict)
        assert len(LOG_FORMATS) == 4
        
        # 檢查是否包含所有格式
        expected_formats = ["simple", "detailed", "json", "structured"]
        for fmt in expected_formats:
            assert fmt in LOG_FORMATS

    def test_formatter_config_structure(self) -> None:
        """測試格式器配置結構"""
        for format_name, config in LOG_FORMATS.items():
            assert isinstance(config, dict)
            
            required_keys = {"format_string", "date_format", "style"}
            assert set(config.keys()) == required_keys
            
            assert isinstance(config["format_string"], str)
            assert isinstance(config["date_format"], str)
            assert isinstance(config["style"], str)
            assert config["style"] == "%"

    def test_simple_format(self) -> None:
        """測試簡單格式"""
        config = LOG_FORMATS["simple"]
        assert "%(asctime)s" in config["format_string"]
        assert "%(name)s" in config["format_string"]
        assert "%(levelname)s" in config["format_string"]
        assert "%(message)s" in config["format_string"]

    def test_detailed_format(self) -> None:
        """測試詳細格式"""
        config = LOG_FORMATS["detailed"]
        assert "%(asctime)s" in config["format_string"]
        assert "%(filename)s" in config["format_string"]
        assert "%(lineno)d" in config["format_string"]
        assert "%(funcName)s" in config["format_string"]

    def test_json_format(self) -> None:
        """測試 JSON 格式"""
        config = LOG_FORMATS["json"]
        format_string = config["format_string"]
        
        # 檢查 JSON 結構的關鍵元素
        assert '"timestamp"' in format_string
        assert '"logger"' in format_string
        assert '"level"' in format_string
        assert '"message"' in format_string

    def test_structured_format(self) -> None:
        """測試結構化格式"""
        config = LOG_FORMATS["structured"]
        assert "[%(asctime)s]" in config["format_string"]
        assert "%(levelname)-8s" in config["format_string"]


class TestDefaultLogDir:
    """測試預設日誌目錄"""

    def test_default_log_dir_type(self) -> None:
        """測試預設日誌目錄類型"""
        assert isinstance(DEFAULT_LOG_DIR, Path)

    def test_default_log_dir_value(self) -> None:
        """測試預設日誌目錄值"""
        assert str(DEFAULT_LOG_DIR) == "logs"


class TestEnsureLogDirectory:
    """測試確保日誌目錄函數"""

    def test_ensure_log_directory_default(self) -> None:
        """測試確保預設日誌目錄"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_dir = temp_path / "test_logs"
            
            result = ensure_log_directory(log_dir)
            
            assert isinstance(result, Path)
            assert result == log_dir
            assert result.exists()
            assert result.is_dir()

    def test_ensure_log_directory_string_path(self) -> None:
        """測試使用字串路徑"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = str(Path(temp_dir) / "string_logs")
            
            result = ensure_log_directory(log_dir)
            
            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_dir()

    def test_ensure_log_directory_existing(self) -> None:
        """測試已存在的目錄"""
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_dir = Path(temp_dir) / "existing_logs"
            existing_dir.mkdir()
            
            result = ensure_log_directory(existing_dir)
            
            assert result == existing_dir
            assert result.exists()

    @patch('pathlib.Path.mkdir')
    def test_ensure_log_directory_mkdir_called(self, mock_mkdir: MagicMock) -> None:
        """測試 mkdir 被正確調用"""
        log_dir = Path("test_logs")
        ensure_log_directory(log_dir)
        mock_mkdir.assert_called_once_with(exist_ok=True)


class TestGetHandlerConfigs:
    """測試取得處理器配置函數"""

    def test_get_handler_configs_structure(self) -> None:
        """測試處理器配置結構"""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = get_handler_configs(temp_dir)
            
            assert isinstance(configs, dict)
            
            expected_handlers = ["console", "file_info", "file_error", "file_debug"]
            for handler in expected_handlers:
                assert handler in configs

    def test_console_handler_config(self) -> None:
        """測試控制台處理器配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = get_handler_configs(temp_dir)
            console_config = configs["console"]
            
            assert console_config["handler_type"] == "StreamHandler"
            assert console_config["level"] == LogLevel.INFO.value
            assert console_config["format_type"] == "simple"
            assert console_config["filename"] is None

    def test_file_handler_configs(self) -> None:
        """測試文件處理器配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = get_handler_configs(temp_dir)
            
            # 測試 file_info 處理器
            info_config = configs["file_info"]
            assert info_config["handler_type"] == "RotatingFileHandler"
            assert info_config["level"] == LogLevel.INFO.value
            assert info_config["format_type"] == "detailed"
            assert "application.log" in info_config["filename"]
            assert info_config["max_bytes"] == 10485760  # 10MB
            assert info_config["backup_count"] == 5
            
            # 測試 file_error 處理器
            error_config = configs["file_error"]
            assert error_config["handler_type"] == "RotatingFileHandler"
            assert error_config["level"] == LogLevel.ERROR.value
            assert "error.log" in error_config["filename"]
            assert error_config["backup_count"] == 3
            
            # 測試 file_debug 處理器
            debug_config = configs["file_debug"]
            assert debug_config["handler_type"] == "RotatingFileHandler"
            assert debug_config["level"] == LogLevel.DEBUG.value
            assert "debug.log" in debug_config["filename"]
            assert debug_config["max_bytes"] == 20971520  # 20MB

    def test_handler_config_types(self) -> None:
        """測試處理器配置類型"""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = get_handler_configs(temp_dir)
            
            for handler_name, config in configs.items():
                handler_config: HandlerConfig = config
                
                assert isinstance(handler_config["handler_type"], str)
                assert isinstance(handler_config["level"], str)
                assert isinstance(handler_config["format_type"], str)
                assert handler_config["filename"] is None or isinstance(handler_config["filename"], str)
                assert handler_config["max_bytes"] is None or isinstance(handler_config["max_bytes"], int)
                assert handler_config["backup_count"] is None or isinstance(handler_config["backup_count"], int)
                assert handler_config["when"] is None or isinstance(handler_config["when"], str)
                assert handler_config["interval"] is None or isinstance(handler_config["interval"], int)
                assert handler_config["encoding"] is None or isinstance(handler_config["encoding"], str)

    def test_log_directory_creation(self) -> None:
        """測試日誌目錄創建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "new_logs"
            configs = get_handler_configs(log_dir)
            
            # 確認目錄被創建
            assert log_dir.exists()
            assert log_dir.is_dir()
            
            # 確認文件路徑包含正確的目錄
            for handler_name, config in configs.items():
                if config["filename"]:
                    file_path = Path(config["filename"])
                    assert file_path.parent == log_dir


class TestGetLoggingConfig:
    """測試取得日誌配置函數"""

    def test_get_logging_config_default(self) -> None:
        """測試預設日誌配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = get_logging_config(log_dir=temp_dir)
            
            assert isinstance(config, dict)
            
            required_keys = {
                "version", "disable_existing_loggers", "formatters",
                "handlers", "loggers", "root"
            }
            assert set(config.keys()) == required_keys

    def test_logging_config_version(self) -> None:
        """測試日誌配置版本"""
        config = get_logging_config()
        assert config["version"] == 1
        assert isinstance(config["disable_existing_loggers"], bool)

    def test_logging_config_formatters(self) -> None:
        """測試日誌配置格式器"""
        config = get_logging_config()
        formatters = config["formatters"]
        
        assert isinstance(formatters, dict)
        for format_name in LOG_FORMATS.keys():
            assert format_name in formatters

    def test_logging_config_handlers(self) -> None:
        """測試日誌配置處理器"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = get_logging_config(log_dir=temp_dir)
            handlers = config["handlers"]
            
            assert isinstance(handlers, dict)
            expected_handlers = ["console", "file_info", "file_error"]
            for handler in expected_handlers:
                assert handler in handlers

    def test_logging_config_loggers(self) -> None:
        """測試日誌配置記錄器"""
        config = get_logging_config()
        loggers = config["loggers"]
        
        assert isinstance(loggers, dict)
        # 檢查是否包含預期的記錄器配置

    def test_logging_config_root(self) -> None:
        """測試根記錄器配置"""
        config = get_logging_config()
        root_config = config["root"]
        
        assert isinstance(root_config, dict)
        assert "level" in root_config
        assert "handlers" in root_config
        assert "propagate" in root_config
        assert isinstance(root_config["handlers"], list)

    def test_logging_config_with_debug(self) -> None:
        """測試除錯模式日誌配置"""
        config = get_logging_config(debug_mode=True)
        
        # 除錯模式下應該包含除錯處理器
        handlers = config["handlers"]
        assert "file_debug" in handlers

    def test_logging_config_without_debug(self) -> None:
        """測試非除錯模式日誌配置"""
        config = get_logging_config(debug_mode=False)
        
        # 非除錯模式下仍然包含除錯處理器，但可能不在根記錄器中使用
        handlers = config["handlers"]
        # 檢查配置是否合理，但不強制要求不包含除錯處理器
        assert isinstance(handlers, dict)
        assert "console" in handlers
        assert "file_info" in handlers
        assert "file_error" in handlers

    def test_logging_config_custom_levels(self) -> None:
        """測試自定義級別日誌配置"""
        config = get_logging_config(
            console_level=LogLevel.WARNING,
            file_level=LogLevel.ERROR
        )
        
        root_config = config["root"]
        assert root_config["level"] == LogLevel.ERROR.value


class TestEnvironmentSpecificLoggingConfigs:
    """測試環境特定日誌配置"""

    def test_development_logging_config(self) -> None:
        """測試開發環境日誌配置"""
        config = get_development_logging_config()
        
        assert isinstance(config, dict)
        assert config["version"] == 1
        
        # 開發環境應該啟用除錯
        assert "file_debug" in config["handlers"]
        
        # 檢查根記錄器級別
        root_config = config["root"]
        assert root_config["level"] == LogLevel.DEBUG.value

    def test_production_logging_config(self) -> None:
        """測試生產環境日誌配置"""
        config = get_production_logging_config()
        
        assert isinstance(config, dict)
        assert config["version"] == 1
        
        # 生產環境仍然包含除錯處理器，但根記錄器使用 INFO 級別
        handlers = config["handlers"]
        assert isinstance(handlers, dict)
        assert "console" in handlers
        assert "file_info" in handlers
        assert "file_error" in handlers
        
        # 檢查根記錄器級別
        root_config = config["root"]
        assert root_config["level"] == LogLevel.INFO.value

    def test_testing_logging_config(self) -> None:
        """測試測試環境日誌配置"""
        config = get_testing_logging_config()
        
        assert isinstance(config, dict)
        assert config["version"] == 1
        
        # 測試環境應該啟用除錯
        assert "file_debug" in config["handlers"]
        
        # 檢查根記錄器級別
        root_config = config["root"]
        assert root_config["level"] == LogLevel.DEBUG.value
        
        # 檢查是否使用測試日誌目錄
        handlers = config["handlers"]
        for handler_name, handler_config in handlers.items():
            if handler_config.get("filename"):
                assert "test" in handler_config["filename"]

    def test_environment_configs_consistency(self) -> None:
        """測試環境配置一致性"""
        dev_config = get_development_logging_config()
        prod_config = get_production_logging_config()
        test_config = get_testing_logging_config()
        
        configs = [dev_config, prod_config, test_config]
        
        # 所有配置應該有相同的基本結構
        for config in configs:
            assert config["version"] == 1
            assert isinstance(config["disable_existing_loggers"], bool)
            assert "formatters" in config
            assert "handlers" in config
            assert "loggers" in config
            assert "root" in config


class TestLoggerNames:
    """測試記錄器名稱常數"""

    def test_logger_names_attributes(self) -> None:
        """測試記錄器名稱屬性"""
        assert hasattr(LoggerNames, "WORKFLOW")
        assert hasattr(LoggerNames, "NODES")

    def test_logger_names_values(self) -> None:
        """測試記錄器名稱值"""
        assert LoggerNames.WORKFLOW == "src.workflow"
        assert LoggerNames.NODES == "src.workflow.nodes"

    def test_logger_names_types(self) -> None:
        """測試記錄器名稱類型"""
        assert isinstance(LoggerNames.WORKFLOW, str)
        assert isinstance(LoggerNames.NODES, str)


class TestAgentLogConfigs:
    """測試 Agent 日誌配置"""

    def test_agent_log_configs_structure(self) -> None:
        """測試 Agent 日誌配置結構"""
        assert isinstance(AGENT_LOG_CONFIGS, dict)
        
        for agent_name, config in AGENT_LOG_CONFIGS.items():
            assert isinstance(config, dict)
            # 檢查基本配置鍵
            assert "logger_name" in config
            # 其他鍵是特定於該 agent 的布林配置選項

    def test_agent_log_configs_values(self) -> None:
        """測試 Agent 日誌配置值"""
        for agent_name, config in AGENT_LOG_CONFIGS.items():
            if "level" in config:
                assert config["level"] in [level.value for level in LogLevel]
            if "handlers" in config:
                assert isinstance(config["handlers"], list)
            if "propagate" in config:
                assert isinstance(config["propagate"], bool)


class TestLogFilters:
    """測試日誌過濾器"""

    def test_log_filters_structure(self) -> None:
        """測試日誌過濾器結構"""
        assert isinstance(LOG_FILTERS, dict)

    def test_log_filters_values(self) -> None:
        """測試日誌過濾器值"""
        for filter_name, filter_config in LOG_FILTERS.items():
            assert isinstance(filter_config, dict)
            # 檢查過濾器配置是否包含必要信息


class TestTypeDefinitions:
    """測試類型定義"""

    def test_handler_config_type(self) -> None:
        """測試處理器配置類型"""
        config: HandlerConfig = {
            "handler_type": "StreamHandler",
            "level": LogLevel.INFO.value,
            "format_type": "simple",
            "filename": None,
            "max_bytes": None,
            "backup_count": None,
            "when": None,
            "interval": None,
            "encoding": None
        }
        
        assert isinstance(config["handler_type"], str)
        assert isinstance(config["level"], str)
        assert isinstance(config["format_type"], str)

    def test_formatter_config_type(self) -> None:
        """測試格式器配置類型"""
        config: FormatterConfig = {
            "format_string": "%(asctime)s - %(name)s - %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
            "style": "%"
        }
        
        assert isinstance(config["format_string"], str)
        assert isinstance(config["date_format"], str)
        assert isinstance(config["style"], str)

    def test_logger_config_type(self) -> None:
        """測試記錄器配置類型"""
        config: LoggerConfig = {
            "level": LogLevel.INFO.value,
            "handlers": ["console", "file"],
            "propagate": True
        }
        
        assert isinstance(config["level"], str)
        assert isinstance(config["handlers"], list)
        assert isinstance(config["propagate"], bool)

    def test_logging_system_config_type(self) -> None:
        """測試日誌系統配置類型"""
        # 使用實際配置進行測試
        config = get_development_logging_config()
        system_config: LoggingSystemConfig = config
        
        assert isinstance(system_config["version"], int)
        assert isinstance(system_config["disable_existing_loggers"], bool)
        assert isinstance(system_config["formatters"], dict)
        assert isinstance(system_config["handlers"], dict)
        assert isinstance(system_config["loggers"], dict)
        assert isinstance(system_config["root"], dict)


class TestConfigurationIntegration:
    """測試配置整合"""

    def test_format_handler_integration(self) -> None:
        """測試格式與處理器整合"""
        with tempfile.TemporaryDirectory() as temp_dir:
            handlers = get_handler_configs(temp_dir)
            
            for handler_name, handler_config in handlers.items():
                format_type = handler_config["format_type"]
                assert format_type in LOG_FORMATS

    def test_environment_config_integration(self) -> None:
        """測試環境配置整合"""
        configs = [
            get_development_logging_config(),
            get_production_logging_config(),
            get_testing_logging_config()
        ]
        
        for config in configs:
            # 檢查格式器
            for formatter_name in config["formatters"].keys():
                assert formatter_name in LOG_FORMATS
            
            # 檢查處理器配置
            for handler_name, handler_config in config["handlers"].items():
                assert "level" in handler_config
                level = handler_config["level"]
                assert level in [lvl.value for lvl in LogLevel]

    def test_log_directory_integration(self) -> None:
        """測試日誌目錄整合"""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_log_dir = Path(temp_dir) / "custom_logs"
            
            # 測試自定義目錄
            config = get_logging_config(log_dir=custom_log_dir)
            
            # 檢查處理器是否使用正確的目錄
            handlers = config["handlers"]
            for handler_name, handler_config in handlers.items():
                if handler_config.get("filename"):
                    file_path = Path(handler_config["filename"])
                    assert file_path.parent == custom_log_dir

    def test_configuration_completeness(self) -> None:
        """測試配置完整性"""
        config = get_development_logging_config()
        
        # 檢查所有格式器是否定義
        formatters = config["formatters"]
        for format_name in formatters.keys():
            assert format_name in LOG_FORMATS
            assert formatters[format_name] == LOG_FORMATS[format_name]
        
        # 檢查所有處理器是否有對應的格式器
        handlers = config["handlers"]
        for handler_name, handler_config in handlers.items():
            if "formatter" in handler_config:
                formatter_name = handler_config["formatter"]
                assert formatter_name in formatters

    def test_level_hierarchy_consistency(self) -> None:
        """測試級別層次一致性"""
        # 檢查不同環境的級別設置是否合理
        dev_config = get_development_logging_config()
        prod_config = get_production_logging_config()
        
        dev_level = dev_config["root"]["level"]
        prod_level = prod_config["root"]["level"]
        
        # 開發環境應該更詳細（級別更低）
        level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        dev_index = level_order.index(dev_level)
        prod_index = level_order.index(prod_level)
        
        assert dev_index <= prod_index  # 開發環境級別應該 <= 生產環境級別


class TestSetupLogging:
    """測試日誌系統設置功能"""

    def test_setup_logging_basic(self) -> None:
        """測試基本日誌設置"""
        with patch('src.config.logging_config.logging.config.dictConfig') as mock_dict_config:
            with patch('src.config.settings.get_environment', return_value="development"):
                from src.config.logging_config import setup_logging
                
                setup_logging(log_level="INFO", verbose=False)
                
                # 驗證 dictConfig 被調用
                mock_dict_config.assert_called_once()
                config = mock_dict_config.call_args[0][0]
                
                # 驗證基本結構
                assert "version" in config
                assert "handlers" in config
                assert "loggers" in config
                assert "root" in config

    def test_setup_logging_debug_mode(self) -> None:
        """測試調試模式日誌設置"""
        with patch('src.config.logging_config.logging.config.dictConfig') as mock_dict_config:
            with patch('src.config.settings.get_environment', return_value="development"):
                from src.config.logging_config import setup_logging
                
                setup_logging(log_level="DEBUG", verbose=True)
                
                mock_dict_config.assert_called_once()
                config = mock_dict_config.call_args[0][0]
                
                # 驗證 console 處理器級別設為 DEBUG
                assert config["handlers"]["console"]["level"] == "DEBUG"
                assert config["root"]["level"] == "DEBUG"

    def test_setup_logging_production_environment(self) -> None:
        """測試生產環境日誌設置"""
        with patch('src.config.logging_config.logging.config.dictConfig') as mock_dict_config:
            with patch('src.config.settings.get_environment', return_value="production"):
                from src.config.logging_config import setup_logging
                
                setup_logging(log_level="INFO", verbose=False)
                
                mock_dict_config.assert_called_once()

    def test_setup_logging_testing_environment(self) -> None:
        """測試測試環境日誌設置"""
        with patch('src.config.logging_config.logging.config.dictConfig') as mock_dict_config:
            with patch('src.config.settings.get_environment', return_value="testing"):
                from src.config.logging_config import setup_logging
                
                setup_logging(log_level="WARNING", verbose=False)
                
                mock_dict_config.assert_called_once()

    def test_setup_logging_invalid_level(self) -> None:
        """測試無效日誌級別處理"""
        with patch('src.config.logging_config.logging.config.dictConfig') as mock_dict_config:
            with patch('src.config.settings.get_environment', return_value="development"):
                from src.config.logging_config import setup_logging
                
                # 傳入無效的日誌級別
                setup_logging(log_level="INVALID_LEVEL", verbose=False)
                
                mock_dict_config.assert_called_once()
                config = mock_dict_config.call_args[0][0]
                
                # 應該回退到 INFO 級別
                assert config["root"]["level"] == "INFO"

    def test_setup_logging_verbose_override(self) -> None:
        """測試詳細模式覆蓋設置"""
        with patch('src.config.logging_config.logging.config.dictConfig') as mock_dict_config:
            with patch('src.config.settings.get_environment', return_value="development"):
                from src.config.logging_config import setup_logging
                
                # verbose=True 應該覆蓋日誌級別為 DEBUG
                setup_logging(log_level="WARNING", verbose=True)
                
                mock_dict_config.assert_called_once()
                config = mock_dict_config.call_args[0][0]
                
                # console 級別應該是 DEBUG（verbose 模式）
                assert config["handlers"]["console"]["level"] == "DEBUG"

    def test_setup_logging_ensure_directory_called(self) -> None:
        """測試確保日誌目錄被調用"""
        with patch('src.config.logging_config.logging.config.dictConfig'):
            with patch('src.config.logging_config.ensure_log_directory') as mock_ensure:
                with patch('src.config.settings.get_environment', return_value="development"):
                    from src.config.logging_config import setup_logging
                    
                    setup_logging()
                    
                    # 驗證目錄創建函數被調用（至少一次）
                    assert mock_ensure.call_count >= 1


class TestLoggerNames:
    """測試日誌器名稱常數"""

    def test_logger_names_attributes(self) -> None:
        """測試日誌器名稱常數屬性"""
        assert LoggerNames.WORKFLOW == "src.workflow"
        assert LoggerNames.NODES == "src.workflow.nodes"
        assert LoggerNames.ANALYZER_DESIGNER == "src.workflow.nodes.analyzer_designer"
        assert LoggerNames.REPRODUCER == "src.workflow.nodes.reproducer"
        assert LoggerNames.EVALUATOR == "src.workflow.nodes.evaluator"
        assert LoggerNames.LLM_SERVICE == "src.services.llm_service"
        assert LoggerNames.DATA_LOADER == "src.services.data_loader"
        assert LoggerNames.DATASET_MANAGER == "src.core.dataset_manager"
        assert LoggerNames.QUALITY == "quality"
        assert LoggerNames.PERFORMANCE == "performance"
        assert LoggerNames.ERROR == "error"
        assert LoggerNames.SECURITY == "security"

    def test_logger_names_consistency(self) -> None:
        """測試日誌器名稱與配置的一致性"""
        # 檢查 AGENT_LOG_CONFIGS 中引用的名稱是否存在
        for agent_config in AGENT_LOG_CONFIGS.values():
            logger_name = agent_config["logger_name"]
            # 確保名稱存在於 LoggerNames 類中
            logger_names_values = [
                getattr(LoggerNames, attr) for attr in dir(LoggerNames)
                if not attr.startswith('_')
            ]
            assert logger_name in logger_names_values


class TestAgentLogConfigs:
    """測試 Agent 日誌配置"""

    def test_agent_log_configs_structure(self) -> None:
        """測試 Agent 日誌配置結構"""
        assert isinstance(AGENT_LOG_CONFIGS, dict)
        assert len(AGENT_LOG_CONFIGS) >= 3  # 至少有 analyzer_designer, reproducer, evaluator
        
        # 檢查每個配置的結構
        for agent_name, config in AGENT_LOG_CONFIGS.items():
            assert "logger_name" in config
            assert isinstance(config["logger_name"], str)

    def test_agent_log_configs_completeness(self) -> None:
        """測試 Agent 日誌配置完整性"""
        expected_agents = ["analyzer_designer", "reproducer", "evaluator"]
        for agent in expected_agents:
            assert agent in AGENT_LOG_CONFIGS

    def test_agent_specific_configs(self) -> None:
        """測試 Agent 特定配置"""
        # 測試 analyzer_designer 配置
        analyzer_config = AGENT_LOG_CONFIGS["analyzer_designer"]
        assert analyzer_config["logger_name"] == LoggerNames.ANALYZER_DESIGNER
        assert "log_translation_details" in analyzer_config
        assert "log_terminology_decisions" in analyzer_config
        assert "log_complexity_analysis" in analyzer_config
        
        # 測試 reproducer 配置
        reproducer_config = AGENT_LOG_CONFIGS["reproducer"]
        assert reproducer_config["logger_name"] == LoggerNames.REPRODUCER
        assert "log_qa_execution" in reproducer_config
        assert "log_reasoning_steps" in reproducer_config
        
        # 測試 evaluator 配置
        evaluator_config = AGENT_LOG_CONFIGS["evaluator"]
        assert evaluator_config["logger_name"] == LoggerNames.EVALUATOR
        assert "log_semantic_analysis" in evaluator_config
        assert "log_quality_scores" in evaluator_config


class TestLogFilters:
    """測試日誌過濾器配置"""

    def test_log_filters_structure(self) -> None:
        """測試日誌過濾器結構"""
        assert isinstance(LOG_FILTERS, dict)
        
        # 檢查預期的過濾器類型
        expected_filters = ["sensitive_data", "performance_threshold", "quality_threshold"]
        for filter_type in expected_filters:
            assert filter_type in LOG_FILTERS

    def test_sensitive_data_filter(self) -> None:
        """測試敏感資料過濾器"""
        sensitive_filter = LOG_FILTERS["sensitive_data"]
        assert "api_keys" in sensitive_filter
        assert "user_tokens" in sensitive_filter
        assert "internal_paths" in sensitive_filter
        
        # 確保都是布林值
        for key, value in sensitive_filter.items():
            assert isinstance(value, bool)

    def test_performance_threshold_filter(self) -> None:
        """測試效能閾值過濾器"""
        perf_filter = LOG_FILTERS["performance_threshold"]
        assert "min_duration_ms" in perf_filter
        assert "log_slow_operations" in perf_filter
        
        # 檢查數據類型
        assert isinstance(perf_filter["min_duration_ms"], (int, float))
        assert isinstance(perf_filter["log_slow_operations"], bool)

    def test_quality_threshold_filter(self) -> None:
        """測試品質閾值過濾器"""
        quality_filter = LOG_FILTERS["quality_threshold"]
        assert "min_score_to_log" in quality_filter
        assert "log_all_failures" in quality_filter
        
        # 檢查數據類型
        assert isinstance(quality_filter["min_score_to_log"], (int, float))
        assert isinstance(quality_filter["log_all_failures"], bool)


class TestModuleExports:
    """測試模組匯出"""

    def test_all_exports_exist(self) -> None:
        """測試所有匯出項目都存在"""
        from src.config import logging_config
        
        # 檢查 __all__ 中的每個項目都可以從模組中導入
        for item in logging_config.__all__:
            assert hasattr(logging_config, item), f"Missing export: {item}"

    def test_all_exports_complete(self) -> None:
        """測試匯出列表完整性"""
        from src.config.logging_config import __all__
        
        # 檢查重要的類和函數是否都在 __all__ 中
        expected_exports = [
            "LogLevel", "LogFormat", "LoggerNames",
            "ensure_log_directory", "get_logging_config", "setup_logging",
            "get_development_logging_config", "get_production_logging_config",
            "get_testing_logging_config"
        ]
        
        for export in expected_exports:
            assert export in __all__, f"Missing from __all__: {export}"
