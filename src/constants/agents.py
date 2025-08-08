"""
Agent-related constants and configurations
"""

# Agent role definitions
AGENT_ROLES = {
    "analyzer_designer": {
        "name": "分析設計者",
        "description": "負責語義分析與序列式翻譯執行",
        "primary_tasks": [
            "語義複雜度分析",
            "Question 和 Answer 的序列式翻譯",
            "程式碼區塊識別與保護", 
            "基於 Prompt 的術語一致性管理"
        ],
    },
    "reproducer": {
        "name": "再現者", 
        "description": "負責 QA 推理與盲測驗證",
        "primary_tasks": [
            "原始英文問題的 QA 推理執行",
            "翻譯中文問題的 QA 推理執行",
            "執行結果的結構化記錄",
            "推理過程的詳細追蹤"
        ],
    },
    "evaluator": {
        "name": "評估者",
        "description": "負責語義一致性分析與翻譯指導",
        "primary_tasks": [
            "雙語 QA 結果語義一致性分析",
            "翻譯品質偏差檢測",
            "翻譯改進建議生成",
            "重試機制管理",
            "品質閾值控制"
        ],
    },
}

# Default model assignments for each agent
AGENT_MODELS = {
    "cloud_config": {
        "analyzer_designer": "openai/gpt-4o",
        "reproducer": "anthropic/claude-4-sonnet", 
        "evaluator": "openai/gpt-4o"
    },
    "alternative_cloud": {
        "analyzer_designer": "google/gemini-2.5-flash",
        "reproducer": "anthropic/claude-4-sonnet",
        "evaluator": "google/gemini-2.5-flash"
    },
    "local_config": {
        "analyzer_designer": "ollama/deepseek-coder:6.7b",
        "reproducer": "ollama/llama3.1:8b", 
        "evaluator": "ollama/qwen2.5:7b"
    }
}

# Agent processing timeouts (seconds)
AGENT_TIMEOUTS = {
    "analyzer_designer": 180,  # 3 minutes for translation
    "reproducer": 240,         # 4 minutes for QA execution  
    "evaluator": 120,          # 2 minutes for evaluation
}

# Agent memory requirements (MB)
AGENT_MEMORY_REQUIREMENTS = {
    "analyzer_designer": 512,
    "reproducer": 1024,
    "evaluator": 256,
}

# Workflow node priorities
NODE_PRIORITIES = {
    "analyzer_designer_node": 1,
    "reproducer_node": 2, 
    "evaluator_node": 3,
}

# Agent capability flags
AGENT_CAPABILITIES = {
    "analyzer_designer": {
        "supports_code_analysis": True,
        "supports_multilingual": True,
        "supports_context_awareness": True,
        "supports_retry_optimization": True,
    },
    "reproducer": {
        "supports_blind_testing": True,
        "supports_qa_reasoning": True,
        "supports_parallel_execution": False,
        "supports_detailed_logging": True,
    },
    "evaluator": {
        "supports_semantic_comparison": True,
        "supports_quality_scoring": True,
        "supports_feedback_generation": True,
        "supports_threshold_management": True,
    },
}

# Prompt template types for each agent
PROMPT_TEMPLATES = {
    "analyzer_designer": [
        "translation_with_context",
        "code_block_protection", 
        "terminology_consistency",
        "retry_improvement",
    ],
    "reproducer": [
        "qa_execution_english",
        "qa_execution_chinese",
        "reasoning_documentation",
        "blind_test_setup",
    ],
    "evaluator": [
        "semantic_consistency_check",
        "quality_assessment",
        "improvement_suggestions",
        "threshold_validation",
    ],
}
