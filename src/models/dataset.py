"""
資料集相關模型定義
Dataset Related Models

定義系統中處理資料集相關的所有資料結構
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from enum import Enum
import time


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
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
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
    processing_config: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)


__all__ = [
    "ProcessingStatus",
    "ComplexityLevel", 
    "Language",
    "ProcessingStage",
    "CharacterEncoding",
    "OriginalRecord",
    "TranslationResult",
    "QAExecutionResult", 
    "ProcessedRecord",
    "DatasetMetadata",
]
