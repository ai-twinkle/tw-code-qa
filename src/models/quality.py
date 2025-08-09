"""
品質評估相關模型定義
Quality Assessment Related Models

定義系統中品質評估和錯誤處理相關的所有資料結構
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import time


class ErrorType(Enum):
    """錯誤類型枚舉"""
    API_CONNECTION = "api_connection"
    TRANSLATION_QUALITY = "translation_quality"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    OTHER = "other"


@dataclass
class ErrorRecord:
    """錯誤記錄"""
    error_type: ErrorType
    error_message: str
    timestamp: float
    retry_attempt: int
    agent_name: str
    recovery_action: str
    context_data: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """驗證錯誤記錄"""
        if not self.error_message:
            raise ValueError("Error message cannot be empty")


@dataclass
class QualityMetric:
    """品質指標"""
    metric_name: str
    score: float
    max_score: float
    description: str
    
    def __post_init__(self) -> None:
        """驗證品質指標"""
        if self.score < 0 or self.score > self.max_score:
            raise ValueError(f"Score {self.score} must be between 0 and {self.max_score}")
    
    def get_percentage(self) -> float:
        """獲取百分比分數"""
        return (self.score / self.max_score) * 100.0


@dataclass
class QualityAssessment:
    """品質評估結果"""
    record_id: str
    semantic_consistency_score: float  # 語義一致性分數 (0-10)
    code_integrity_score: float       # 程式碼完整性分數 (0-10)
    translation_naturalness_score: float  # 翻譯自然度分數 (0-10)
    overall_quality_score: float      # 綜合品質分數
    
    # 詳細評估
    semantic_analysis: str
    code_analysis: str
    naturalness_analysis: str
    
    # 改進建議
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # 元資料
    evaluator_model: str = ""
    evaluation_timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self) -> None:
        """計算綜合品質分數"""
        if self.overall_quality_score == 0:  # 如果未手動設定
            self.overall_quality_score = (
                self.semantic_consistency_score * 0.5 + 
                self.code_integrity_score * 0.3 + 
                self.translation_naturalness_score * 0.2
            )
    
    def is_acceptable_quality(self) -> bool:
        """判斷品質是否可接受"""
        return self.overall_quality_score >= 7.0
    
    def needs_retry(self) -> bool:
        """判斷是否需要重試"""
        return self.overall_quality_score < 7.0


@dataclass 
class QualityReport:
    """單一記錄品質報告"""
    record_id: str
    quality_assessment: QualityAssessment
    processing_time: float
    retry_count: int
    final_status: str  # "passed", "failed", "needs_manual_review"
    error_history: List[ErrorRecord] = field(default_factory=list)
    
    def add_error(self, error: ErrorRecord) -> None:
        """添加錯誤記錄"""
        self.error_history.append(error)


@dataclass
class BatchQualityReport:
    """批次品質報告"""
    batch_id: str
    total_records: int
    processed_records: int
    passed_records: int
    failed_records: int
    retry_records: int
    
    # 品質統計
    average_quality_score: float
    min_quality_score: float
    max_quality_score: float
    
    # 處理統計
    total_processing_time: float
    average_processing_time: float
    total_retries: int
    
    # 錯誤統計
    error_summary: Dict[ErrorType, int] = field(default_factory=dict)
    
    # 詳細報告
    individual_reports: List[QualityReport] = field(default_factory=list)
    
    # 元資料
    batch_start_time: float = field(default_factory=time.time)
    batch_end_time: Optional[float] = None
    
    def calculate_statistics(self) -> None:
        """計算批次統計資訊"""
        if not self.individual_reports:
            return
            
        quality_scores = [
            report.quality_assessment.overall_quality_score 
            for report in self.individual_reports
        ]
        
        self.average_quality_score = sum(quality_scores) / len(quality_scores)
        self.min_quality_score = min(quality_scores)
        self.max_quality_score = max(quality_scores)
        
        processing_times = [report.processing_time for report in self.individual_reports]
        self.average_processing_time = sum(processing_times) / len(processing_times)
        
        # 計算錯誤統計
        error_counts: Dict[ErrorType, int] = {}
        total_retries = 0
        
        for report in self.individual_reports:
            total_retries += report.retry_count
            for error in report.error_history:
                error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
        
        self.total_retries = total_retries
        self.error_summary = error_counts
    
    def get_success_rate(self) -> float:
        """獲取成功率"""
        if self.total_records == 0:
            return 0.0
        return self.passed_records / self.total_records
    
    def get_failure_rate(self) -> float:
        """獲取失敗率"""
        if self.total_records == 0:
            return 0.0
        return self.failed_records / self.total_records


__all__ = [
    "ErrorType",
    "ErrorRecord", 
    "QualityMetric",
    "QualityAssessment",
    "QualityReport",
    "BatchQualityReport",
]
