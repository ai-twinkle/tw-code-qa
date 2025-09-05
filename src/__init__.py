"""
Traditional Chinese Code-QA Dataset Conversion System
Multi-Agent Architecture for Dataset Translation and Quality Assurance
"""

__version__ = "0.1.0"
__author__ = "AI Twinkle"
__description__ = "Traditional Chinese Code-QA Dataset Conversion System using Multi-Agent Architecture"

from .config import (
    get_settings,
    get_llm_config,
    get_logging_config,
)

from .models import (
    DatasetMetadata,
    QualityAssessment,
    QAExecutionResult,
)

from .services import (
    LLMService,
    DataLoaderFactory,
)

from .workflow import (
    WorkflowState,
    WorkflowManager,
)

from .core import (
    DatasetManager,
)

from .utils import (
    DataFormatConverter,
)

__all__ = [
    # Configuration
    "get_settings",
    "get_llm_config", 
    "get_logging_config",
    
    # Models
    "DatasetMetadata",
    "QualityAssessment",
    "QAExecutionResult",
    
    # Services
    "LLMService",
    "DataLoaderFactory",
    
    # Workflow
    "WorkflowState",
    "WorkflowManager",
    
    # Core
    "DatasetManager",
    
    # Utils
    "DataFormatConverter",
]
