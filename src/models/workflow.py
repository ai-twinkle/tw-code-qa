"""
工作流狀態類型定義
Workflow State Type Definitions

定義 LangGraph 工作流中使用的狀態結構
根據系統設計文檔第677行要求實現
"""

from typing import List, Optional, Dict
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages

from .dataset import (
    OriginalRecord,
    TranslationResult,
    QAExecutionResult,
    ProcessingStatus
)
from .quality import (
    QualityAssessment,
    ErrorRecord
)


# 明確定義性能指標的結構
class PerformanceMetrics(TypedDict):
    """性能指標結構"""
    execution_time_ms: float       # 執行時間（毫秒）
    memory_usage_mb: float         # 記憶體使用量（MB）
    token_count: int               # Token 數量
    success_rate: float            # 成功率
    error_count: int               # 錯誤次數


# 明確定義重試策略配置
class RetryPolicy(TypedDict):
    """重試策略配置"""
    max_attempts: int              # 最大嘗試次數
    base_delay_seconds: float      # 基礎延遲時間（秒）
    max_delay_seconds: float       # 最大延遲時間（秒）
    exponential_backoff: bool      # 是否使用指數退避
    retry_on_timeout: bool         # 超時時是否重試


# 明確定義監控配置
class MonitoringConfig(TypedDict):
    """監控配置"""
    enable_logging: bool           # 是否啟用日誌
    enable_metrics: bool           # 是否啟用指標收集
    log_level: str                 # 日誌級別
    metrics_interval_seconds: int  # 指標收集間隔（秒）
    alert_on_failure: bool         # 失敗時是否發送警報


class WorkflowState(TypedDict):
    """
    LangGraph 工作流狀態定義
    
    這是 Multi-Agent 系統的核心狀態結構，在三個 Agent 之間傳遞：
    - AnalyzerDesigner: 分析設計者 Agent
    - Reproducer: 再現者 Agent  
    - Evaluator: 評估者 Agent
    """
    current_record: OriginalRecord                              # 當前處理的記錄
    translation_result: Optional[TranslationResult]             # 翻譯結果
    original_qa_result: Optional[QAExecutionResult]             # 原始英文 QA 結果
    translated_qa_result: Optional[QAExecutionResult]           # 翻譯中文 QA 結果
    quality_assessment: Optional[QualityAssessment]             # 品質評估結果
    retry_count: int                                           # 重試次數
    processing_status: ProcessingStatus                        # 處理狀態
    error_history: Annotated[List[ErrorRecord], add_messages]  # 錯誤歷史（使用 LangGraph 消息機制）
    improvement_suggestions: List[str]                         # 改進建議列表


class BatchWorkflowState(TypedDict):
    """
    批次處理工作流狀態
    
    用於管理多筆記錄的批次處理狀態
    """
    batch_id: str                           # 批次ID
    total_records: int                      # 總記錄數
    processed_records: int                  # 已處理記錄數
    successful_records: int                 # 成功處理記錄數
    failed_records: int                     # 失敗記錄數
    current_batch_status: ProcessingStatus  # 當前批次狀態
    individual_states: List[WorkflowState]  # 個別記錄的工作流狀態
    batch_start_time: float                 # 批次開始時間
    estimated_completion_time: float        # 預估完成時間


class NodeExecutionContext(TypedDict):
    """
    節點執行上下文
    
    提供給每個 Agent 節點的執行環境資訊
    """
    node_name: str                    # 節點名稱
    execution_id: str                 # 執行ID
    llm_provider: str                 # 使用的 LLM Provider
    llm_model: str                    # 使用的 LLM 模型
    max_retries: int                  # 最大重試次數
    timeout_seconds: int              # 超時時間（秒）
    execution_timestamp: float        # 執行時間戳


class NodeExecutionResult(TypedDict):
    """
    節點執行結果
    
    每個 Agent 節點執行後的標準化結果
    """
    node_name: str                    # 執行的節點名稱
    execution_success: bool           # 執行是否成功
    execution_time: float             # 執行時間（秒）
    updated_state: WorkflowState      # 更新後的工作流狀態
    error_message: Optional[str]      # 錯誤訊息（如果有）
    performance_metrics: PerformanceMetrics  # 性能指標


class GraphConfiguration(TypedDict):
    """
    圖形配置
    
    LangGraph 工作流的配置參數
    """
    enable_checkpointing: bool        # 是否啟用檢查點
    max_execution_time: int           # 最大執行時間（秒）
    parallel_execution: bool          # 是否支援並行執行
    retry_policy: RetryPolicy         # 重試策略配置
    monitoring_config: MonitoringConfig  # 監控配置


# 工作流狀態轉換映射
WORKFLOW_STATE_TRANSITIONS = {
    ProcessingStatus.PENDING: [ProcessingStatus.PROCESSING],
    ProcessingStatus.PROCESSING: [
        ProcessingStatus.COMPLETED,
        ProcessingStatus.FAILED,
        ProcessingStatus.RETRY_NEEDED
    ],
    ProcessingStatus.RETRY_NEEDED: [
        ProcessingStatus.PROCESSING,
        ProcessingStatus.FAILED,
        ProcessingStatus.SKIPPED
    ],
    ProcessingStatus.FAILED: [ProcessingStatus.SKIPPED],
    ProcessingStatus.COMPLETED: [],
    ProcessingStatus.SKIPPED: []
}


# 定義工作流狀態更新值的明確類型
class StateUpdateValue(TypedDict, total=False):
    """工作流狀態更新值"""
    current_record: OriginalRecord
    translation_result: Optional[TranslationResult]
    original_qa_result: Optional[QAExecutionResult]
    translated_qa_result: Optional[QAExecutionResult]
    quality_assessment: Optional[QualityAssessment]
    retry_count: int
    processing_status: ProcessingStatus
    error_history: List[ErrorRecord]
    improvement_suggestions: List[str]

__all__ = [
    "PerformanceMetrics",
    "RetryPolicy", 
    "MonitoringConfig",
    "WorkflowState",
    "BatchWorkflowState", 
    "NodeExecutionContext",
    "NodeExecutionResult",
    "GraphConfiguration",
    "StateUpdateValue",
    "WORKFLOW_STATE_TRANSITIONS",
]
