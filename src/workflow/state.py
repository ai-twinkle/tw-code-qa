"""
LangGraph 工作流狀態管理
LangGraph Workflow State Management

實現 Multi-Agent 系統的核心狀態結構，在三個 Agent 之間傳遞：
- AnalyzerDesigner: 分析設計者 Agent
- Reproducer: 再現者 Agent  
- Evaluator: 評估者 Agent
"""

from typing import List, Optional

from langgraph.graph import add_messages
from typing_extensions import TypedDict, Annotated

from ..models.dataset import (
    OriginalRecord,
    TranslationResult,
    QAExecutionResult,
    ProcessingStatus
)
from ..models.quality import (
    QualityAssessment,
    ErrorRecord
)


class WorkflowState(TypedDict):
    """
    LangGraph 工作流狀態定義
    
    這是 Multi-Agent 系統的核心狀態結構，在三個 Agent 之間傳遞：
    - AnalyzerDesigner: 分析設計者 Agent
    - Reproducer: 再現者 Agent  
    - Evaluator: 評估者 Agent
    """
    # 當前處理的原始記錄
    current_record: OriginalRecord                              
    
    # Agent 處理結果
    translation_result: Optional[TranslationResult]             # 翻譯結果
    original_qa_result: Optional[QAExecutionResult]             # 原始英文 QA 結果
    translated_qa_result: Optional[QAExecutionResult]           # 翻譯中文 QA 結果
    quality_assessment: Optional[QualityAssessment]             # 品質評估結果
    
    # 處理狀態管理
    retry_count: int                                           # 重試次數
    processing_status: ProcessingStatus                        # 處理狀態
    
    # 錯誤與改進追蹤 (使用 LangGraph 消息機制)
    error_history: Annotated[List[ErrorRecord], add_messages]  # 錯誤歷史
    improvement_suggestions: List[str]                         # 改進建議列表


class BatchProcessingState(TypedDict):
    """
    批次處理狀態
    
    用於管理多筆記錄的批次處理
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


class AgentExecutionContext(TypedDict):
    """
    Agent 執行上下文
    
    提供給每個 Agent 節點的執行環境資訊
    """
    agent_name: str                   # Agent 名稱 ("analyzer_designer", "reproducer", "evaluator")
    execution_id: str                 # 執行ID
    llm_provider: str                 # 使用的 LLM Provider
    llm_model: str                    # 使用的 LLM 模型
    max_retries: int                  # 最大重試次數
    timeout_seconds: int              # 超時時間（秒）
    execution_timestamp: float        # 執行時間戳


# 工作流狀態更新值的類型定義 (用於狀態部分更新)
class StateUpdateValue(TypedDict, total=False):
    """
    工作流狀態更新值
    
    用於 Node 節點進行部分狀態更新時的類型安全
    """
    current_record: OriginalRecord
    translation_result: Optional[TranslationResult]
    original_qa_result: Optional[QAExecutionResult]
    translated_qa_result: Optional[QAExecutionResult]
    quality_assessment: Optional[QualityAssessment]
    retry_count: int
    processing_status: ProcessingStatus
    error_history: List[ErrorRecord]
    improvement_suggestions: List[str]


# 狀態轉換規則定義
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


def create_initial_state(record: OriginalRecord) -> WorkflowState:
    """
    創建初始工作流狀態
    
    Args:
        record: 要處理的原始記錄
        
    Returns:
        初始化的工作流狀態
    """
    return WorkflowState(
        current_record=record,
        translation_result=None,
        original_qa_result=None,
        translated_qa_result=None,
        quality_assessment=None,
        retry_count=0,
        processing_status=ProcessingStatus.PENDING,
        error_history=[],
        improvement_suggestions=[]
    )


def is_state_transition_valid(
    current_status: ProcessingStatus, 
    new_status: ProcessingStatus
) -> bool:
    """
    檢查狀態轉換是否有效
    
    Args:
        current_status: 當前狀態
        new_status: 新狀態
        
    Returns:
        是否為有效的狀態轉換
    """
    valid_transitions = WORKFLOW_STATE_TRANSITIONS.get(current_status, [])
    return new_status in valid_transitions


def update_state_safely(
    state: WorkflowState, 
    updates: StateUpdateValue
) -> WorkflowState:
    """
    安全地更新工作流狀態
    
    Args:
        state: 當前狀態
        updates: 要更新的值
        
    Returns:
        更新後的狀態
        
    Raises:
        ValueError: 當狀態轉換無效時
    """
    # 檢查狀態轉換的有效性
    if 'processing_status' in updates:
        new_status = updates['processing_status']
        if not is_state_transition_valid(state['processing_status'], new_status):
            raise ValueError(
                f"Invalid state transition from {state['processing_status']} to {new_status}"
            )
    
    # 執行更新
    state.update(updates)
    return state


__all__ = [
    "WorkflowState",
    "BatchProcessingState", 
    "AgentExecutionContext",
    "StateUpdateValue",
    "WORKFLOW_STATE_TRANSITIONS",
    "create_initial_state",
    "is_state_transition_valid",
    "update_state_safely",
]
