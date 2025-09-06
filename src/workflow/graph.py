"""
LangGraph 工作流程圖定義
LangGraph Workflow Graph Definition

定義 Multi-Agent 系統的工作流程圖，連接三個 Agent：
- AnalyzerDesigner: 分析設計者 Agent  
- Reproducer: 再現者 Agent
- Evaluator: 評估者 Agent

實現重試邏輯和狀態轉換管理
"""

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from .nodes.analyzer_designer_node import analyzer_designer_node
from .nodes.evaluator_node import evaluator_node
from .nodes.reproducer_node import reproducer_node
from .state import WorkflowState
from ..models.dataset import (
    ProcessingStatus
)

logger = logging.getLogger(__name__)


def route_after_evaluation(state: WorkflowState) -> Literal["analyzer_designer", "end"]:
    """
    評估後的路由邏輯
    
    根據評估結果決定是重試還是結束：
    - 如果需要重試且未達重試上限：返回 analyzer_designer 重新翻譯
    - 否則：結束工作流
    
    Args:
        state: 當前工作流狀態
        
    Returns:
        下一個節點名稱或 "end"
    """
    try:
        processing_status = state.get("processing_status", ProcessingStatus.PENDING)
        current_record = state.get("current_record")
        record_id = current_record.id if current_record else "unknown"
        
        if processing_status == ProcessingStatus.RETRY_NEEDED:
            logger.info(f"Record {record_id} needs retry")
            return "analyzer_designer"
        else:
            logger.info(f"Record {record_id} completed with status: {processing_status}")
            return "end"
            
    except Exception as e:
        logger.error(f"Routing error: {e}")
        return "end"  # 出錯時結束流程


def create_workflow_graph() -> StateGraph:
    """
    創建 LangGraph 工作流程圖
    
    Returns:
        配置好的 StateGraph
    """
    logger.info("Creating Multi-Agent workflow graph")
    
    # 創建狀態圖
    graph = StateGraph(WorkflowState)
    
    # 添加節點
    graph.add_node("analyzer_designer", analyzer_designer_node)
    graph.add_node("reproducer", reproducer_node)
    graph.add_node("evaluator", evaluator_node)
    
    # 定義工作流邊
    # 開始 -> 分析設計者
    graph.add_edge(START, "analyzer_designer")
    
    # 分析設計者 -> 再現者
    graph.add_edge("analyzer_designer", "reproducer")
    
    # 再現者 -> 評估者
    graph.add_edge("reproducer", "evaluator")
    
    # 評估者的條件邊：根據評估結果決定重試或結束
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "analyzer_designer": "analyzer_designer",  # 重試：回到分析設計者
            "end": END  # 結束：工作流完成
        }
    )
    
    logger.info("Multi-Agent workflow graph created successfully")
    return graph


def create_workflow_with_checkpointing() -> CompiledStateGraph:
    """
    創建帶檢查點的工作流程圖
    
    Returns:
        編譯後的 StateGraph，支援檢查點和記憶體
    """
    logger.info("Creating workflow with checkpointing support")
    
    # 創建基礎圖
    graph = create_workflow_graph()
    
    # 創建記憶體檢查點
    memory = MemorySaver()
    
    # 編譯圖並啟用檢查點
    compiled_graph = graph.compile(checkpointer=memory)
    
    logger.info("Workflow with checkpointing created successfully")
    return compiled_graph


def create_basic_workflow() -> CompiledStateGraph:
    """
    創建基礎工作流程圖（不帶檢查點）
    
    Returns:
        編譯後的 StateGraph
    """
    logger.info("Creating basic workflow")
    
    # 創建基礎圖
    graph = create_workflow_graph()
    
    # 編譯圖
    compiled_graph = graph.compile()
    
    logger.info("Basic workflow created successfully")
    return compiled_graph


class WorkflowManager:
    """工作流管理器"""
    
    def __init__(self, enable_checkpointing: bool = True):
        """
        初始化工作流管理器
        
        Args:
            enable_checkpointing: 是否啟用檢查點功能
        """
        self.enable_checkpointing = enable_checkpointing
        self.logger = logging.getLogger(__name__ + ".WorkflowManager")
        
        # 創建工作流
        if enable_checkpointing:
            self.workflow = create_workflow_with_checkpointing()
        else:
            self.workflow = create_basic_workflow()
    
    def process_record(self, state: WorkflowState, config: dict = None) -> WorkflowState:
        """
        處理單一記錄
        
        Args:
            state: 初始工作流狀態
            config: 工作流配置
            
        Returns:
            最終工作流狀態
        """
        try:
            current_record = state.get("current_record")
            record_id = current_record.id if current_record else "unknown"
            self.logger.info(f"Processing record {record_id}")
            
            # 執行工作流
            if config:
                # 使用配置執行
                final_state = self.workflow.invoke(state, config=config)
            else:
                # 無配置執行
                final_state = self.workflow.invoke(state)
            
            self.logger.info(f"Record {record_id} processing completed with status: {final_state.get('processing_status')}")
            return final_state
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            # 更新狀態為失敗
            state["processing_status"] = ProcessingStatus.FAILED
            return state
    
    def stream_record_processing(self, state: WorkflowState, config: dict = None):
        """
        流式處理記錄（用於監控進度）
        
        Args:
            state: 初始工作流狀態
            config: 工作流配置
            
        Yields:
            中間狀態更新
        """
        try:
            current_record = state.get("current_record")
            record_id = current_record.id if current_record else "unknown"
            self.logger.info(f"Starting stream processing for record {record_id}")
            
            # 流式執行工作流
            if config and self.enable_checkpointing:
                for event in self.workflow.stream(state, config=config):
                    yield event
            else:
                for event in self.workflow.stream(state):
                    yield event
                    
        except Exception as e:
            self.logger.error(f"Stream processing failed: {e}")
            yield {"error": str(e)}
    
    def get_workflow_graph_visualization(self) -> bytes:
        """
        獲取工作流圖的視覺化
        
        Returns:
            PNG 圖片的二進制數據
        """
        try:
            return self.workflow.get_graph().draw_mermaid_png()
        except Exception as e:
            self.logger.error(f"Failed to generate workflow visualization: {e}")
            return b""


# 預設工作流實例
default_workflow_manager = WorkflowManager(enable_checkpointing=True)


__all__ = [
    "WorkflowManager",
    "create_workflow_graph",
    "create_workflow_with_checkpointing", 
    "create_basic_workflow",
    "route_after_evaluation",
    "default_workflow_manager",
]
