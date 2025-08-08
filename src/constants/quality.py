"""
Quality control constants and thresholds
"""

# Quality assessment thresholds
QUALITY_THRESHOLDS = {
    "auto_pass_threshold": 7.0,      # 自動通過閾值
    "auto_retry_threshold": 7.0,     # 自動重試閾值 (< 7.0)
    "max_retry_attempts": 3,         # 最大重試次數
    "batch_failure_rate_limit": 0.15, # 批次失敗率上限 (15%)
}

# Semantic consistency scoring standards (0-10 scale)
SEMANTIC_SCORING_STANDARDS = {
    "excellent": {
        "range": (9, 10),
        "description": "完全語義一致，翻譯保持原意且表達自然"
    },
    "good": {
        "range": (7, 8), 
        "description": "高度一致，僅有輕微語言風格差異"
    },
    "acceptable": {
        "range": (5, 6),
        "description": "基本一致，存在可接受的語義偏差"
    },
    "needs_improvement": {
        "range": (3, 4),
        "description": "部分偏差，需要改進翻譯"
    },
    "poor": {
        "range": (0, 2),
        "description": "嚴重偏差，翻譯錯誤或語義遺失"
    }
}

# Code integrity scoring standards (0-10 scale)
CODE_INTEGRITY_STANDARDS = {
    "perfect": {
        "range": (10, 10),
        "description": "程式碼語法完全正確，結構完整"
    },
    "excellent": {
        "range": (8, 9),
        "description": "語法正確，可能有極少數格式問題"
    },
    "good": {
        "range": (6, 7), 
        "description": "語法基本正確，有少量非關鍵錯誤"
    },
    "acceptable": {
        "range": (4, 5),
        "description": "有明顯語法錯誤但邏輯可理解"
    },
    "poor": {
        "range": (0, 3),
        "description": "嚴重語法錯誤或程式碼結構損壞"
    }
}

# Translation naturalness scoring standards (0-10 scale)
TRANSLATION_NATURALNESS_STANDARDS = {
    "excellent": {
        "range": (9, 10),
        "description": "翻譯自然流暢，符合繁體中文表達習慣"
    },
    "good": {
        "range": (7, 8),
        "description": "翻譯通順，偶有用詞不夠自然"
    },
    "acceptable": {
        "range": (5, 6),
        "description": "翻譯可理解，但有明顯翻譯痕跡"
    },
    "needs_improvement": {
        "range": (3, 4),
        "description": "翻譯生硬，影響理解"
    },
    "poor": {
        "range": (0, 2),
        "description": "翻譯錯誤或無法理解"
    }
}

# Overall quality score calculation weights
QUALITY_SCORE_WEIGHTS = {
    "semantic_consistency": 0.5,
    "code_integrity": 0.3,
    "translation_naturalness": 0.2,
}

# Retry policies by error type
RETRY_POLICIES = {
    "api_errors": {
        "max_retries": 5,
        "backoff_strategy": "exponential",  # 1s, 2s, 4s, 8s, 16s
        "base_delay": 1.0,
        "max_delay": 60.0,
    },
    "quality_issues": {
        "max_retries": 3,
        "backoff_strategy": "immediate",
        "base_delay": 0.0,
        "max_delay": 0.0,
    },
    "syntax_errors": {
        "max_retries": 2,
        "backoff_strategy": "immediate", 
        "base_delay": 0.0,
        "max_delay": 0.0,
    },
    "timeout_errors": {
        "max_retries": 3,
        "backoff_strategy": "linear",  # 5s, 10s, 15s
        "base_delay": 5.0,
        "max_delay": 30.0,
    },
}

# Quality check configurations
QUALITY_CHECKS = {
    "code_blocks": {
        "syntax_validation": True,
        "structure_integrity": True,
        "comment_translation": True,
        "api_name_protection": True,
    },
    "semantic_consistency": {
        "llm_evaluation": True,
        "key_concept_comparison": True,
        "logic_step_consistency": True,
        "answer_completeness": True,
    },
    "translation_quality": {
        "terminology_consistency": True,
        "naturalness_assessment": True,
        "context_preservation": True,
        "cultural_adaptation": True,
    },
}

# Error classification and handling
ERROR_HANDLING = {
    "critical_errors": [
        "authentication_failure",
        "model_unavailable", 
        "system_resource_exhausted",
    ],
    "recoverable_errors": [
        "network_timeout",
        "rate_limit_exceeded",
        "temporary_api_error",
    ],
    "quality_errors": [
        "low_semantic_consistency",
        "code_syntax_error",
        "poor_translation_quality",
    ],
}

# Performance monitoring thresholds
PERFORMANCE_THRESHOLDS = {
    "processing_time_warning_seconds": 300,  # 5 minutes
    "processing_time_critical_seconds": 600, # 10 minutes
    "memory_usage_warning_mb": 2048,         # 2GB
    "memory_usage_critical_mb": 4096,        # 4GB
    "success_rate_warning": 0.85,            # 85%
    "success_rate_critical": 0.70,           # 70%
}
