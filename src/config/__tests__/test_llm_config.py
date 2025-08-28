"""
LLM configuration tests
LLM 配置測試模組

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.config.llm_config import (
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
)


class TestLLMProviderEnum:
    """測試 LLM 提供者枚舉"""

    def test_llm_provider_values(self) -> None:
        """測試 LLM 提供者值"""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.AZURE_OPENAI.value == "azure_openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.GOOGLE.value == "google"
        assert LLMProvider.MISTRAL.value == "mistral"
        assert LLMProvider.LOCAL.value == "local"

    def test_llm_provider_members(self) -> None:
        """測試 LLM 提供者成員"""
        providers = list(LLMProvider)
        assert len(providers) == 6
        expected_providers = [
            LLMProvider.OPENAI,
            LLMProvider.AZURE_OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.GOOGLE,
            LLMProvider.MISTRAL,
            LLMProvider.LOCAL
        ]
        for provider in expected_providers:
            assert provider in providers


class TestModelTierEnum:
    """測試模型等級枚舉"""

    def test_model_tier_values(self) -> None:
        """測試模型等級值"""
        assert ModelTier.BASIC.value == "basic"
        assert ModelTier.STANDARD.value == "standard"
        assert ModelTier.PREMIUM.value == "premium"
        assert ModelTier.ADVANCED.value == "advanced"

    def test_model_tier_members(self) -> None:
        """測試模型等級成員"""
        tiers = list(ModelTier)
        assert len(tiers) == 4
        expected_tiers = [
            ModelTier.BASIC,
            ModelTier.STANDARD,
            ModelTier.PREMIUM,
            ModelTier.ADVANCED
        ]
        for tier in expected_tiers:
            assert tier in tiers


class TestDefaultLLMConfigs:
    """測試預設 LLM 配置"""

    def test_default_llm_configs_structure(self) -> None:
        """測試預設 LLM 配置結構"""
        assert isinstance(DEFAULT_LLM_CONFIGS, dict)
        assert len(DEFAULT_LLM_CONFIGS) > 0
        
        # 檢查是否包含基本模型
        expected_models = ["gpt-4o", "gpt-4.1"]
        for model in expected_models:
            assert model in DEFAULT_LLM_CONFIGS

    def test_model_config_structure(self) -> None:
        """測試模型配置結構"""
        for model_name, config in DEFAULT_LLM_CONFIGS.items():
            assert isinstance(config, dict)
            
            # 檢查必要的鍵
            required_keys = {
                "provider", "model_name", "api_key_name", "base_url",
                "temperature", "max_tokens", "top_p", "frequency_penalty",
                "presence_penalty", "timeout_seconds", "retry_config", "rate_limit"
            }
            assert set(config.keys()) == required_keys

    def test_gpt_4o_config(self) -> None:
        """測試 GPT-4o 配置"""
        config = DEFAULT_LLM_CONFIGS["gpt-4o"]
        
        assert config["provider"] == LLMProvider.OPENAI.value
        assert config["model_name"] == "gpt-4o"
        assert config["api_key_name"] == "OPENAI_API_KEY"
        assert config["base_url"] is None
        assert config["temperature"] == 0.1
        assert config["max_tokens"] == 16384
        assert config["top_p"] == 0.95
        assert config["frequency_penalty"] == 0.0
        assert config["presence_penalty"] == 0.0
        assert config["timeout_seconds"] == 120

    def test_retry_config_structure(self) -> None:
        """測試重試配置結構"""
        for model_name, config in DEFAULT_LLM_CONFIGS.items():
            retry_config = config["retry_config"]
            
            required_keys = {
                "max_retries", "initial_delay", "max_delay",
                "exponential_base", "jitter"
            }
            assert set(retry_config.keys()) == required_keys
            
            assert isinstance(retry_config["max_retries"], int)
            assert isinstance(retry_config["initial_delay"], (int, float))
            assert isinstance(retry_config["max_delay"], (int, float))
            assert isinstance(retry_config["exponential_base"], (int, float))
            assert isinstance(retry_config["jitter"], bool)

    def test_rate_limit_config_structure(self) -> None:
        """測試速率限制配置結構"""
        for model_name, config in DEFAULT_LLM_CONFIGS.items():
            rate_limit = config["rate_limit"]
            
            required_keys = {
                "requests_per_minute", "tokens_per_minute", "concurrent_requests"
            }
            assert set(rate_limit.keys()) == required_keys
            
            assert isinstance(rate_limit["requests_per_minute"], int)
            assert isinstance(rate_limit["tokens_per_minute"], int)
            assert isinstance(rate_limit["concurrent_requests"], int)
            
            # 檢查合理的值範圍
            assert rate_limit["requests_per_minute"] > 0
            assert rate_limit["tokens_per_minute"] > 0
            assert rate_limit["concurrent_requests"] > 0


class TestAgentModelConfigs:
    """測試 Agent 模型配置"""

    def test_agent_model_configs_structure(self) -> None:
        """測試 Agent 模型配置結構"""
        assert isinstance(AGENT_MODEL_CONFIGS, dict)
        
        for agent_name, config in AGENT_MODEL_CONFIGS.items():
            assert isinstance(config, dict)
            
            # 檢查必要的鍵
            required_keys = {
                "primary_model", "fallback_model", "temperature",
                "max_tokens", "custom_prompt_template"
            }
            assert set(config.keys()) == required_keys

    def test_agent_config_values(self) -> None:
        """測試 Agent 配置值"""
        for agent_name, config in AGENT_MODEL_CONFIGS.items():
            assert isinstance(config["primary_model"], str)
            assert config["fallback_model"] is None or isinstance(config["fallback_model"], str)
            assert isinstance(config["temperature"], (int, float))
            assert isinstance(config["max_tokens"], int)
            assert config["custom_prompt_template"] is None or isinstance(config["custom_prompt_template"], str)
            
            # 檢查值的合理性
            assert 0.0 <= config["temperature"] <= 2.0
            assert config["max_tokens"] > 0


class TestCostTracking:
    """測試成本追蹤"""

    def test_cost_tracking_config_structure(self) -> None:
        """測試成本追蹤配置結構"""
        assert isinstance(COST_TRACKING_CONFIG, dict)
        
        required_keys = {
            "enabled", "track_by_agent", "track_by_model", 
            "daily_budget_usd", "weekly_budget_usd",
            "monthly_budget_usd", "alert_threshold"
        }
        assert set(COST_TRACKING_CONFIG.keys()) == required_keys

    def test_cost_tracking_config_values(self) -> None:
        """測試成本追蹤配置值"""
        config = COST_TRACKING_CONFIG
        
        assert isinstance(config["enabled"], bool)
        assert isinstance(config["track_by_agent"], bool)
        assert isinstance(config["track_by_model"], bool)
        assert isinstance(config["daily_budget_usd"], (int, float))
        assert isinstance(config["weekly_budget_usd"], (int, float))
        assert isinstance(config["monthly_budget_usd"], (int, float))
        assert isinstance(config["alert_threshold"], (int, float))
        
        # 檢查預算值的合理性
        assert config["daily_budget_usd"] > 0
        assert config["weekly_budget_usd"] > 0
        assert config["monthly_budget_usd"] > 0
        assert 0.0 <= config["alert_threshold"] <= 1.0


class TestModelPricing:
    """測試模型價格"""

    def test_model_pricing_structure(self) -> None:
        """測試模型價格結構"""
        assert isinstance(MODEL_PRICING, dict)
        assert len(MODEL_PRICING) > 0
        
        # 檢查是否包含主要模型
        expected_models = ["gpt-4o", "gpt-4.1", "claude-4-sonnet", "gemini-2.5-flash"]
        for model in expected_models:
            assert model in MODEL_PRICING

    def test_pricing_config_structure(self) -> None:
        """測試價格配置結構"""
        for model_name, pricing in MODEL_PRICING.items():
            assert isinstance(pricing, dict)
            
            required_keys = {"input_price_per_1k", "output_price_per_1k"}
            assert set(pricing.keys()) == required_keys
            
            assert isinstance(pricing["input_price_per_1k"], (int, float))
            assert isinstance(pricing["output_price_per_1k"], (int, float))
            
            # 檢查價格為正數
            assert pricing["input_price_per_1k"] > 0
            assert pricing["output_price_per_1k"] > 0

    def test_gpt_4o_pricing(self) -> None:
        """測試 GPT-4o 價格"""
        pricing = MODEL_PRICING["gpt-4o"]
        assert pricing["input_price_per_1k"] == 0.0025
        assert pricing["output_price_per_1k"] == 0.010

    def test_claude_pricing(self) -> None:
        """測試 Claude 價格"""
        pricing = MODEL_PRICING["claude-4-sonnet"]
        assert pricing["input_price_per_1k"] == 0.003
        assert pricing["output_price_per_1k"] == 0.015


class TestQualityThresholds:
    """測試品質閾值"""

    def test_quality_thresholds_structure(self) -> None:
        """測試品質閾值結構"""
        assert isinstance(QUALITY_THRESHOLDS, dict)
        
        expected_keys = {
            "minimum_semantic_score", "maximum_retry_attempts",
            "translation_consistency_threshold", "qa_accuracy_threshold"
        }
        assert set(QUALITY_THRESHOLDS.keys()) == expected_keys

    def test_quality_thresholds_values(self) -> None:
        """測試品質閾值值"""
        thresholds = QUALITY_THRESHOLDS
        
        assert isinstance(thresholds["minimum_semantic_score"], (int, float))
        assert isinstance(thresholds["maximum_retry_attempts"], int)
        assert isinstance(thresholds["translation_consistency_threshold"], (int, float))
        assert isinstance(thresholds["qa_accuracy_threshold"], (int, float))
        
        # 檢查值的合理性
        assert thresholds["minimum_semantic_score"] >= 0
        assert thresholds["maximum_retry_attempts"] > 0
        assert 0.0 <= thresholds["translation_consistency_threshold"] <= 1.0
        assert 0.0 <= thresholds["qa_accuracy_threshold"] <= 1.0


class TestConfigurationFunctions:
    """測試配置函數"""

    def test_get_model_config_existing(self) -> None:
        """測試取得存在的模型配置"""
        # 測試已知存在的模型
        config = get_model_config("gpt-4o")
        assert config is not None
        assert isinstance(config, dict)
        assert config == DEFAULT_LLM_CONFIGS["gpt-4o"]

    def test_get_model_config_non_existing(self) -> None:
        """測試取得不存在的模型配置"""
        config = get_model_config("non_existent_model")
        assert config is None

    def test_get_model_config_all_default_models(self) -> None:
        """測試取得所有預設模型配置"""
        for model_name in DEFAULT_LLM_CONFIGS.keys():
            config = get_model_config(model_name)
            assert config is not None
            assert config == DEFAULT_LLM_CONFIGS[model_name]

    def test_get_agent_config_existing(self) -> None:
        """測試取得存在的 Agent 配置"""
        # 如果有 Agent 配置，測試第一個
        if AGENT_MODEL_CONFIGS:
            agent_name = list(AGENT_MODEL_CONFIGS.keys())[0]
            config = get_agent_config(agent_name)
            assert config is not None
            assert isinstance(config, dict)
            assert config == AGENT_MODEL_CONFIGS[agent_name]

    def test_get_agent_config_non_existing(self) -> None:
        """測試取得不存在的 Agent 配置"""
        config = get_agent_config("non_existent_agent")
        assert config is None

    def test_get_agent_config_all_default_agents(self) -> None:
        """測試取得所有預設 Agent 配置"""
        for agent_name in AGENT_MODEL_CONFIGS.keys():
            config = get_agent_config(agent_name)
            assert config is not None
            assert config == AGENT_MODEL_CONFIGS[agent_name]


class TestCostCalculation:
    """測試成本計算"""

    def test_calculate_cost_existing_model(self) -> None:
        """測試計算存在模型的成本"""
        # 測試 GPT-4o
        cost = calculate_cost("gpt-4o", 1000, 500)
        
        # 預期成本：(1000/1000 * 0.0025) + (500/1000 * 0.010) = 0.0025 + 0.005 = 0.0075
        expected_cost = 0.0075
        assert abs(cost - expected_cost) < 0.0001

    def test_calculate_cost_non_existing_model(self) -> None:
        """測試計算不存在模型的成本"""
        cost = calculate_cost("non_existent_model", 1000, 500)
        assert cost == 0.0

    def test_calculate_cost_zero_tokens(self) -> None:
        """測試零 token 的成本計算"""
        cost = calculate_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_calculate_cost_input_only(self) -> None:
        """測試僅輸入 token 的成本計算"""
        cost = calculate_cost("gpt-4o", 1000, 0)
        
        # 預期成本：1000/1000 * 0.0025 = 0.0025
        expected_cost = 0.0025
        assert abs(cost - expected_cost) < 0.0001

    def test_calculate_cost_output_only(self) -> None:
        """測試僅輸出 token 的成本計算"""
        cost = calculate_cost("gpt-4o", 0, 500)
        
        # 預期成本：500/1000 * 0.010 = 0.005
        expected_cost = 0.005
        assert abs(cost - expected_cost) < 0.0001

    def test_calculate_cost_large_numbers(self) -> None:
        """測試大數量 token 的成本計算"""
        cost = calculate_cost("gpt-4o", 100000, 50000)
        
        # 預期成本：(100000/1000 * 0.0025) + (50000/1000 * 0.010) = 0.25 + 0.5 = 0.75
        expected_cost = 0.75
        assert abs(cost - expected_cost) < 0.0001

    def test_calculate_cost_all_pricing_models(self) -> None:
        """測試所有有價格的模型成本計算"""
        for model_name in MODEL_PRICING.keys():
            cost = calculate_cost(model_name, 1000, 1000)
            assert cost > 0
            assert isinstance(cost, float)

    def test_calculate_cost_formula_accuracy(self) -> None:
        """測試成本計算公式準確性"""
        model_name = "gpt-4.1"
        input_tokens = 2500
        output_tokens = 1500
        
        pricing = MODEL_PRICING[model_name]
        expected_input_cost = (input_tokens / 1000) * pricing["input_price_per_1k"]
        expected_output_cost = (output_tokens / 1000) * pricing["output_price_per_1k"]
        expected_total_cost = expected_input_cost + expected_output_cost
        
        actual_cost = calculate_cost(model_name, input_tokens, output_tokens)
        assert abs(actual_cost - expected_total_cost) < 0.0001


class TestTypeDefinitions:
    """測試類型定義"""

    def test_retry_config_type(self) -> None:
        """測試重試配置類型"""
        config: RetryConfig = {
            "max_retries": 3,
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True
        }
        
        assert isinstance(config["max_retries"], int)
        assert isinstance(config["initial_delay"], (int, float))
        assert isinstance(config["max_delay"], (int, float))
        assert isinstance(config["exponential_base"], (int, float))
        assert isinstance(config["jitter"], bool)

    def test_rate_limit_config_type(self) -> None:
        """測試速率限制配置類型"""
        config: RateLimitConfig = {
            "requests_per_minute": 60,
            "tokens_per_minute": 150000,
            "concurrent_requests": 5
        }
        
        assert isinstance(config["requests_per_minute"], int)
        assert isinstance(config["tokens_per_minute"], int)
        assert isinstance(config["concurrent_requests"], int)

    def test_model_config_type(self) -> None:
        """測試模型配置類型"""
        # 使用預設配置作為測試對象
        for model_name, config in DEFAULT_LLM_CONFIGS.items():
            model_config: ModelConfig = config
            
            assert isinstance(model_config["provider"], str)
            assert isinstance(model_config["model_name"], str)
            assert isinstance(model_config["api_key_name"], str)
            assert model_config["base_url"] is None or isinstance(model_config["base_url"], str)
            assert isinstance(model_config["temperature"], (int, float))
            assert isinstance(model_config["max_tokens"], int)
            assert isinstance(model_config["top_p"], (int, float))
            assert isinstance(model_config["frequency_penalty"], (int, float))
            assert isinstance(model_config["presence_penalty"], (int, float))
            assert isinstance(model_config["timeout_seconds"], int)
            assert isinstance(model_config["retry_config"], dict)
            assert isinstance(model_config["rate_limit"], dict)

    def test_agent_model_config_type(self) -> None:
        """測試 Agent 模型配置類型"""
        for agent_name, config in AGENT_MODEL_CONFIGS.items():
            agent_config: AgentModelConfig = config
            
            assert isinstance(agent_config["primary_model"], str)
            assert agent_config["fallback_model"] is None or isinstance(agent_config["fallback_model"], str)
            assert isinstance(agent_config["temperature"], (int, float))
            assert isinstance(agent_config["max_tokens"], int)
            assert agent_config["custom_prompt_template"] is None or isinstance(agent_config["custom_prompt_template"], str)


class TestConfigurationIntegration:
    """測試配置整合"""

    def test_model_config_consistency(self) -> None:
        """測試模型配置一致性"""
        # 檢查所有模型配置是否有一致的結構
        for model_name, config in DEFAULT_LLM_CONFIGS.items():
            # 檢查是否有對應的價格配置
            if model_name in MODEL_PRICING:
                pricing = MODEL_PRICING[model_name]
                assert "input_price_per_1k" in pricing
                assert "output_price_per_1k" in pricing
            
            # 檢查配置的合理性
            assert 0.0 <= config["temperature"] <= 2.0
            assert config["max_tokens"] > 0
            assert 0.0 <= config["top_p"] <= 1.0
            assert config["timeout_seconds"] > 0

    def test_function_integration(self) -> None:
        """測試函數整合"""
        # 測試 get_model_config 和 calculate_cost 的整合
        for model_name in DEFAULT_LLM_CONFIGS.keys():
            config = get_model_config(model_name)
            assert config is not None
            
            # 如果模型有價格配置，計算成本應該返回正數
            if model_name in MODEL_PRICING:
                cost = calculate_cost(model_name, 1000, 1000)
                assert cost > 0

    def test_configuration_completeness(self) -> None:
        """測試配置完整性"""
        # 確保所有模型配置都有必要的字段
        for model_name, config in DEFAULT_LLM_CONFIGS.items():
            assert config["provider"] in [p.value for p in LLMProvider]
            assert isinstance(config["retry_config"], dict)
            assert isinstance(config["rate_limit"], dict)
            
        # 確保所有價格配置都有必要的字段
        for model_name, pricing in MODEL_PRICING.items():
            assert "input_price_per_1k" in pricing
            assert "output_price_per_1k" in pricing
