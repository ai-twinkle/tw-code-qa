"""
LLM 配置管理
LLM Configuration Management

定義 LLM 服務的配置參數和環境設定
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..constants.llm import LLMProvider, LLMModel


@dataclass
class LLMConfig:
    """
    LLM 配置類
    
    管理不同 LLM Provider 的配置參數
    """
    provider: LLMProvider
    model: LLMModel
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    @classmethod
    def from_environment(cls, provider: Optional[LLMProvider] = None) -> "LLMConfig":
        """
        從環境變數創建配置
        
        Args:
            provider: 指定的 LLM Provider，如果未指定則使用預設
            
        Returns:
            LLM 配置實例
        """
        # 使用指定的 provider 或從環境變數獲取，預設為 OPENAI
        if provider is None:
            provider_str = os.getenv("LLM_PROVIDER", "openai")
            provider = LLMProvider(provider_str)
        
        # 根據 provider 選擇預設模型
        model_mapping = {
            LLMProvider.OPENAI: LLMModel.GPT_4O,
            LLMProvider.ANTHROPIC: LLMModel.CLAUDE_4_SONNET,
            LLMProvider.GOOGLE: LLMModel.GEMINI_2_5_FLASH,
            LLMProvider.OLLAMA: LLMModel.LLAMA3_1
        }
        
        default_model = model_mapping.get(provider, LLMModel.GPT_4O)
        model_str = os.getenv("LLM_MODEL", default_model.value)
        model = LLMModel(model_str)
        
        # 獲取 API Key
        api_key_env_mapping = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY", 
            LLMProvider.GOOGLE: "GOOGLE_API_KEY",
            LLMProvider.OLLAMA: "OLLAMA_API_KEY"  # 如果需要
        }
        
        api_key_env = api_key_env_mapping.get(provider, "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env, "")

        # 獲取 Base URL（主要用於 Ollama）
        base_url = None
        if provider == LLMProvider.OLLAMA:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            retry_attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        轉換為字典格式
        
        Returns:
            配置字典
        """
        return {
            "provider": self.provider.value,
            "model": self.model.value,
            "api_key": "***" if self.api_key else "",  # 隱藏 API Key
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts
        }
    
    def validate(self) -> bool:
        """
        驗證配置是否有效
        
        Returns:
            配置是否有效
        """
        # 檢查 API Key
        if not self.api_key:
            return False
        
        # 檢查參數範圍
        if not (0 <= self.temperature <= 2):
            return False
        
        if self.max_tokens <= 0:
            return False
        
        if self.timeout_seconds <= 0:
            return False
        
        if self.retry_attempts < 0:
            return False
        
        return True


@dataclass
class LoggingConfig:
    """
    日誌配置類
    """
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    enable_console_logging: bool = True
    enable_file_logging: bool = False
    
    @classmethod
    def from_environment(cls) -> "LoggingConfig":
        """
        從環境變數創建日誌配置
        
        Returns:
            日誌配置實例
        """
        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            log_file=os.getenv("LOG_FILE"),
            enable_console_logging=os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true",
            enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "false").lower() == "true"
        )


__all__ = [
    "LLMConfig",
    "LoggingConfig"
]
