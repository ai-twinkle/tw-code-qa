"""
LLM service configurations
LLM 服務配置
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from typing_extensions import TypedDict


class LLMProvider(Enum):
    """LLM 服務提供者"""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    LOCAL = "local"


class ModelTier(Enum):
    """模型等級"""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ADVANCED = "advanced"


class RetryConfig(TypedDict):
    """重試配置類型"""
    max_retries: int
    initial_delay: float
    max_delay: float
    exponential_base: float
    jitter: bool


class RateLimitConfig(TypedDict):
    """速率限制配置類型"""
    requests_per_minute: int
    tokens_per_minute: int
    concurrent_requests: int


class ModelConfig(TypedDict):
    """模型配置類型"""
    provider: str
    model_name: str
    api_key_name: str
    base_url: Optional[str]
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    timeout_seconds: int
    retry_config: RetryConfig
    rate_limit: RateLimitConfig


class AgentModelConfig(TypedDict):
    """Agent 模型配置類型"""
    primary_model: str
    fallback_model: Optional[str]
    temperature: float
    max_tokens: int
    custom_prompt_template: Optional[str]


# LLM 服務預設配置
DEFAULT_LLM_CONFIGS: Dict[str, ModelConfig] = {
    "gpt-4o": {
        "provider": LLMProvider.OPENAI.value,
        "model_name": "gpt-4o",
        "api_key_name": "OPENAI_API_KEY",
        "base_url": None,
        "temperature": 0.1,
        "max_tokens": 16384,  # Updated max tokens
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout_seconds": 120,
        "retry_config": {
            "max_retries": 3,
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True,
        },
        "rate_limit": {
            "requests_per_minute": 60,
            "tokens_per_minute": 150000,
            "concurrent_requests": 5,
        },
    },
    "gpt-4.1": {
        "provider": LLMProvider.OPENAI.value,
        "model_name": "gpt-4.1",
        "api_key_name": "OPENAI_API_KEY",
        "base_url": None,
        "temperature": 0.1,
        "max_tokens": 65536,  # Updated max tokens
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout_seconds": 120,
        "retry_config": {
            "max_retries": 3,
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True,
        },
        "rate_limit": {
            "requests_per_minute": 50,
            "tokens_per_minute": 120000,
            "concurrent_requests": 4,
        },
    },
    "claude-4-sonnet": {
        "provider": LLMProvider.ANTHROPIC.value,
        "model_name": "claude-4-sonnet",
        "api_key_name": "ANTHROPIC_API_KEY",
        "base_url": None,
        "temperature": 0.1,
        "max_tokens": 8192,  # Updated max tokens
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout_seconds": 120,
        "retry_config": {
            "max_retries": 3,
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True,
        },
        "rate_limit": {
            "requests_per_minute": 50,
            "tokens_per_minute": 100000,
            "concurrent_requests": 3,
        },
    },
    "gemini-2.5-flash": {
        "provider": LLMProvider.GOOGLE.value,
        "model_name": "gemini-2.5-flash",
        "api_key_name": "GOOGLE_API_KEY",
        "base_url": None,
        "temperature": 0.1,
        "max_tokens": 8192,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout_seconds": 120,
        "retry_config": {
            "max_retries": 3,
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True,
        },
        "rate_limit": {
            "requests_per_minute": 100,
            "tokens_per_minute": 200000,
            "concurrent_requests": 5,
        },
    },
}

# Agent 特定配置
AGENT_MODEL_CONFIGS: Dict[str, AgentModelConfig] = {
    "analyzer_designer": {
        "primary_model": "gpt-4o",  # As per design spec
        "fallback_model": "claude-4-sonnet",
        "temperature": 0.1,
        "max_tokens": 8192,  # Increased for detailed analysis
        "custom_prompt_template": None,
    },
    "reproducer": {
        "primary_model": "claude-4-sonnet",  # As per design spec
        "fallback_model": "gpt-4o",
        "temperature": 0.0,  # 更低的溫度確保一致性
        "max_tokens": 4096,
        "custom_prompt_template": None,
    },
    "evaluator": {
        "primary_model": "gpt-4o",  # As per design spec
        "fallback_model": "claude-4-sonnet",
        "temperature": 0.1,
        "max_tokens": 4096,
        "custom_prompt_template": None,
    },
}

# 成本追蹤配置
COST_TRACKING_CONFIG = {
    "enabled": True,
    "track_by_agent": True,
    "track_by_model": True,
    "daily_budget_usd": 100.0,
    "weekly_budget_usd": 500.0,
    "monthly_budget_usd": 2000.0,
    "alert_threshold": 0.8,  # 80% 預算時發出警告
}

# 模型價格配置 (每 1K tokens 的價格，USD) - Verified from web sources
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o": {
        "input_price_per_1k": 0.0025,   # $2.50 per 1M tokens (verified)
        "output_price_per_1k": 0.010,   # $10.00 per 1M tokens (verified)
    },
    "gpt-4.1": {
        "input_price_per_1k": 0.002,    # $2.00 per 1M tokens (verified)
        "output_price_per_1k": 0.008,   # $8.00 per 1M tokens (verified)
    },
    "claude-4-sonnet": {
        "input_price_per_1k": 0.003,    # $3.00 per 1M tokens (verified Anthropic)
        "output_price_per_1k": 0.015,   # $15.00 per 1M tokens (verified Anthropic)
    },
    "gemini-2.5-flash": {
        "input_price_per_1k": 0.0003,   # $0.30 per 1M tokens (verified Google)
        "output_price_per_1k": 0.0025,  # $2.50 per 1M tokens (verified Google)
    },
}

# 品質閾值配置
QUALITY_THRESHOLDS = {
    "minimum_semantic_score": 7.0,
    "maximum_retry_attempts": 3,
    "translation_consistency_threshold": 0.8,
    "qa_accuracy_threshold": 0.85,
}

def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """取得指定模型的配置"""
    return DEFAULT_LLM_CONFIGS.get(model_name)


def _load_agent_models_from_json() -> Dict[str, AgentModelConfig]:
    """從 agent_models.json 檔案載入 Agent 模型配置"""
    try:
        # 取得 agent_models.json 檔案路徑
        config_dir = Path(__file__).parent
        agent_models_file = config_dir / "agent_models.json"
        
        if not agent_models_file.exists():
            print(f"Warning: agent_models.json not found at {agent_models_file}")
            return {}
            
        with open(agent_models_file, 'r', encoding='utf-8') as f:
            agent_models_data = json.load(f)
        
        # 轉換 JSON 格式到 AgentModelConfig 格式
        agent_configs = {}
        for agent_name, agent_data in agent_models_data.get("agents", {}).items():
            agent_configs[agent_name] = {
                "primary_model": agent_data.get("primary_model", "gpt-4o"),
                "fallback_model": agent_data.get("fallback_models", [None])[0] if agent_data.get("fallback_models") else None,
                "temperature": agent_data.get("model_parameters", {}).get("temperature", 0.1),
                "max_tokens": agent_data.get("model_parameters", {}).get("max_tokens", 4096),
                "custom_prompt_template": None,
            }
        
        print(f"Loaded agent configurations from agent_models.json: {list(agent_configs.keys())}")
        return agent_configs
        
    except Exception as e:
        print(f"Error loading agent_models.json: {e}")
        return {}


def get_agent_config(agent_name: str) -> Optional[AgentModelConfig]:
    """取得指定 Agent 的模型配置"""
    # 首先嘗試從 JSON 檔案載入
    json_configs = _load_agent_models_from_json()
    if json_configs and agent_name in json_configs:
        return json_configs[agent_name]
    
    # 如果 JSON 檔案無法載入或不包含該 Agent，則使用預設配置
    return AGENT_MODEL_CONFIGS.get(agent_name)


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """計算 API 呼叫成本"""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1000) * pricing["input_price_per_1k"]
    output_cost = (output_tokens / 1000) * pricing["output_price_per_1k"]
    
    return input_cost + output_cost


def get_llm_config(model_name: str = "gpt-4o") -> ModelConfig:
    """取得 LLM 配置
    
    Args:
        model_name: 模型名稱，預設為 gpt-4o
        
    Returns:
        ModelConfig: LLM 配置
    """
    config = get_model_config(model_name)
    if config is None:
        # 返回預設配置
        return DEFAULT_LLM_CONFIGS["gpt-4o"]
    return config


__all__ = [
    "LLMProvider",
    "ModelTier",
    "RetryConfig",
    "RateLimitConfig", 
    "ModelConfig",
    "AgentModelConfig",
    "DEFAULT_LLM_CONFIGS",
    "AGENT_MODEL_CONFIGS",
    "COST_TRACKING_CONFIG",
    "MODEL_PRICING",
    "QUALITY_THRESHOLDS",
    "get_model_config",
    "get_agent_config",
    "calculate_cost",
    "get_llm_config",
]
