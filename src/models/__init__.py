"""
資料模型模組
Data Models Module

本模組包含系統中所有核心資料結構的定義
"""

from .dataset import *
from .quality import *

__all__ = [
    # 資料集相關明確類型
    "RecordMetadata",
    "ProcessingConfig",
    # 資料集相關模型
    "OriginalRecord",
    "TranslationResult", 
    "QAExecutionResult",
    "ProcessedRecord",
    "DatasetMetadata",
    
    # 品質評估模型
    "QualityAssessment",
    "ErrorRecord",
    "QualityReport",
    "BatchQualityReport",
    
    # 枚舉類型
    "ProcessingStatus",
    "ComplexityLevel", 
    "ErrorType",
    "Language",
    "ProcessingStage",
    "CharacterEncoding",
]
