"""
資料集相關模型定義
Dataset Related Models

定義系統中處理資料集相關的所有資料結構
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TypedDict


# 明確定義元數據結構
class RecordMetadata(TypedDict, total=False):
    """
    記錄元數據結構
    
    包含記錄的基本元數據和從 flags 動態合併的欄位
    """
    # 基本必需欄位
    tag: str                    # 記錄標籤/類別
    source_index: int          # 原始資料集中的索引
    
    # 可選的 flags 欄位（會從原始記錄的 flags 動態合併）
    refusal: bool              # 是否為拒絕回答
    unsolicited: bool          # 是否為主動提供的內容
    nsfw: bool                 # 是否包含不當內容
    pii: bool                  # 是否包含個人識別資訊
    disclaimer: bool           # 是否需要免責聲明
    
    # 其他可能的動態欄位（支援從 flags 合併任意鍵值對）
    # 注意：TypedDict 的 total=False 允許所有欄位都是可選的


# 明確定義處理配置結構
class ProcessingConfig(TypedDict, total=False):
    """處理配置結構"""
    max_retries: int
    timeout_seconds: int
    enable_caching: bool
    batch_size: int
    parallel_workers: int
    output_format: str
    quality_threshold: float


class ProcessingStatus(Enum):
    """處理狀態枚舉"""
    PENDING = "pending"             # 等待處理
    PROCESSING = "processing"       # 處理中
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"              # 處理失敗
    RETRY_NEEDED = "retry_needed"   # 需要重試
    SKIPPED = "skipped"            # 已跳過


class ComplexityLevel(Enum):
    """複雜度級別枚舉"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class Language(Enum):
    """語言類型枚舉"""
    ENGLISH = "en"
    TRADITIONAL_CHINESE = "zh-tw"


class ProcessingStage(Enum):
    """處理階段枚舉"""
    DATA_LOADING = "data_loading"
    TRANSLATION = "translation"
    QA_EXECUTION = "qa_execution"
    EVALUATION = "evaluation"
    QUALITY_CHECK = "quality_check"
    OUTPUT_GENERATION = "output_generation"
    COMPLETED = "completed"


class CharacterEncoding(Enum):
    """字符編碼枚舉"""
    UTF8 = "utf-8"
    UTF16 = "utf-16"
    UTF32 = "utf-32"
    ASCII = "ascii"


@dataclass
class OriginalRecord:
    """原始資料記錄"""
    id: str
    question: str
    answer: str
    source_dataset: str
    metadata: RecordMetadata = field(default_factory=dict)
    complexity_level: Optional[ComplexityLevel] = None
    
    def __post_init__(self) -> None:
        """資料後處理驗證"""
        if not self.question or not self.answer:
            raise ValueError("Question and answer cannot be empty")


@dataclass 
class TranslationResult:
    """翻譯結果"""
    original_record_id: str
    translated_question: str
    translated_answer: str
    translation_strategy: str
    terminology_notes: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self) -> None:
        """驗證翻譯結果"""
        if not self.translated_question or not self.translated_answer:
            raise ValueError("Translated question and answer cannot be empty")


@dataclass
class QAExecutionResult:
    """QA 執行結果"""
    record_id: str
    language: Language
    input_question: str
    generated_answer: str
    execution_time: float
    reasoning_steps: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProcessedRecord:
    """處理完成的記錄"""
    original_record: OriginalRecord
    translation_result: TranslationResult
    original_qa_result: QAExecutionResult
    translated_qa_result: QAExecutionResult
    processing_status: ProcessingStatus
    final_quality_score: float
    processing_time: float
    retry_count: int = 0
    

@dataclass
class DatasetMetadata:
    """資料集元資料"""
    name: str
    version: str
    description: str
    total_records: int
    processed_records: int
    source_language: Language
    target_language: Language
    creation_timestamp: float = field(default_factory=time.time)
    processing_config: ProcessingConfig = field(default_factory=dict)


__all__ = [
    # 明確類型定義
    "RecordMetadata",
    "ProcessingConfig",
    # 枚舉類型
    "ProcessingStatus",
    "ComplexityLevel",
    "Language",
    "ProcessingStage",
    "CharacterEncoding",
    # 資料類別
    "OriginalRecord",
    "TranslationResult",
    "QAExecutionResult",
    "ProcessedRecord",
    "DatasetMetadata"
]
