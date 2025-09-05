"""
Workflow package initialization
工作流程套件初始化
"""

from .state import (
    WorkflowState,
    BatchProcessingState,
    AgentExecutionContext,
    StateUpdateValue,
    create_initial_state,
    update_state_safely
)

from .graph import (
    WorkflowManager,
    create_workflow_graph,
    create_workflow_with_checkpointing,
    default_workflow_manager
)

__all__ = [
    # State management
    "WorkflowState",
    "BatchProcessingState", 
    "AgentExecutionContext",
    "StateUpdateValue",
    "create_initial_state",
    "update_state_safely",
    
    # Graph management
    "WorkflowManager",
    "create_workflow_graph",
    "create_workflow_with_checkpointing",
    "default_workflow_manager",
]
