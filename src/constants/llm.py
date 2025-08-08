"""
LLM-related constants and configurations
"""

from enum import Enum
from typing import List, Dict


class LLMProvider(Enum):
    """LLM 提供者"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"


class LLMModel(Enum):
    """LLM 模型列舉"""
    # OpenAI Models (按照設計文檔要求)
    GPT_4O = "gpt-4o"
    GPT_4_1 = "gpt-4.1"
    
    # Anthropic Models
    CLAUDE_4_SONNET = "claude-4-sonnet"
    
    # Google Models
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    
    # Ollama Models
    LLAMA3_1 = "llama3.1:8b"
    QWEN2_5 = "qwen2.5:7b"
    DEEPSEEK_CODER = "deepseek-coder:6.7b"


# 各提供者支援的模型映射
PROVIDER_MODELS: Dict[LLMProvider, List[LLMModel]] = {
    LLMProvider.OPENAI: [LLMModel.GPT_4O, LLMModel.GPT_4_1],
    LLMProvider.ANTHROPIC: [LLMModel.CLAUDE_4_SONNET],
    LLMProvider.GOOGLE: [LLMModel.GEMINI_2_5_FLASH],
    LLMProvider.OLLAMA: [LLMModel.LLAMA3_1, LLMModel.QWEN2_5, LLMModel.DEEPSEEK_CODER]
}

# 預設模型設定 (按照設計文檔)
DEFAULT_MODELS: Dict[str, str] = {
    "analyzer_designer": "gpt-4o",
    "reproducer": "claude-4-sonnet",
    "evaluator": "gpt-4o"
}

# API 環境變數
ENV_VARS: Dict[str, str] = {
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY", 
    "google_api_key": "GOOGLE_API_KEY",
    "ollama_base_url": "OLLAMA_BASE_URL"
}


__all__ = [
    "LLMProvider",
    "LLMModel",
    "PROVIDER_MODELS",
    "DEFAULT_MODELS",
    "ENV_VARS"
]
