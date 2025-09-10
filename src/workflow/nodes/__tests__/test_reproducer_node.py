"""
再現者節點測試模組
Test module for Reproducer Node

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import sys
import time
import importlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.workflow.nodes.reproducer_node import ReproducerAgent, reproducer_node as reproducer_function
from src.models.dataset import (
    OriginalRecord, 
    TranslationResult, 
    QAExecutionResult, 
    Language, 
    ProcessingStatus
)
from src.models.quality import ErrorType
from src.constants.llm import LLMProvider, LLMModel


# Get the actual module for patching purposes
reproducer_module = importlib.import_module('src.workflow.nodes.reproducer_node')


@pytest.fixture
def mock_llm_service():
    """模擬 LLM 服務"""
    mock = Mock()
    mock.invoke.return_value = AIMessage(content="Detailed programming answer with step-by-step reasoning.")
    return mock


@pytest.fixture
def mock_agent_config():
    """模擬 agent 配置"""
    return {
        "primary_model": "claude-4-sonnet",
        "fallback_model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 4096
    }


@pytest.fixture
def sample_original_record():
    """創建測試用原始記錄"""
    return OriginalRecord(
        id="test_001",
        question="How do you create a Python class?",
        answer="To create a Python class, use the `class` keyword:\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```",
        source_dataset="test_dataset",
        metadata={"tag": "python", "source_index": 1}
    )


@pytest.fixture
def sample_translation_result():
    """創建測試用翻譯結果"""
    return TranslationResult(
        original_record_id="test_001",
        translated_question="如何創建 Python 類別？",
        translated_answer="要創建 Python 類別，使用 `class` 關鍵字：\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```",
        translation_strategy="sequential_context_aware",
        terminology_notes=["Complexity: Medium"]
    )


@pytest.fixture
def sample_workflow_state(sample_original_record, sample_translation_result):
    """創建測試用工作流狀態"""
    return {
        "current_record": sample_original_record,
        "translation_result": sample_translation_result,
        "processing_status": ProcessingStatus.PROCESSING,
        "retry_count": 0,
        "error_history": []
    }


class TestReproducerAgent:
    """再現者 Agent 測試類"""
    
    @patch('src.services.llm_service.is_production', return_value=False)
    def test_init(self, mock_is_production):
        """測試 ReproducerAgent 初始化"""
        agent = ReproducerAgent()
        
        # Verify the agent is initialized with correct config values
        assert agent.primary_model == "claude-4-sonnet"
        assert agent.temperature == 0.0
        assert agent.max_tokens == 2048  # Updated to match agent_models.json
        assert agent.fallback_model == "gpt-4o"
        assert agent.llm_service is not None
    
    @patch.object(reproducer_module, 'get_agent_config')
    def test_init_no_config(self, mock_get_config):
        """測試初始化時配置不存在"""
        mock_get_config.return_value = None
        
        with pytest.raises(ValueError, match="Failed to get reproducer configuration"):
            ReproducerAgent()
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_execute_qa_reasoning_english(self, mock_create_llm, mock_get_config,
                                        mock_agent_config, mock_llm_service):
        """測試英文 QA 推理執行"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(
            content="1. Understanding: The question asks about Python class creation.\n"
                   "2. Reasoning: Classes are defined using the 'class' keyword.\n"
                   "3. Answer: Here's how to create a Python class..."
        )
        
        agent = ReproducerAgent()
        
        result = agent.execute_qa_reasoning(
            "How do you create a Python class?",
            Language.ENGLISH,
            "Focus on comprehensive solution"
        )
        
        assert isinstance(result, QAExecutionResult)
        assert result.language == Language.ENGLISH
        assert result.input_question == "How do you create a Python class?"
        assert len(result.reasoning_steps) > 0
        assert result.confidence_score == 0.8
        mock_llm_service.invoke.assert_called_once()
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_execute_qa_reasoning_chinese(self, mock_create_llm, mock_get_config,
                                        mock_agent_config, mock_llm_service):
        """測試中文 QA 推理執行"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(
            content="一、問題理解：此問題詢問如何創建 Python 類別。\n"
                   "二、推理過程：類別使用 'class' 關鍵字定義。\n"
                   "三、最終答案：以下是創建 Python 類別的方法..."
        )
        
        agent = ReproducerAgent()
        
        result = agent.execute_qa_reasoning(
            "如何創建 Python 類別？",
            Language.TRADITIONAL_CHINESE,
            "請專注於全面的解決方案"
        )
        
        assert isinstance(result, QAExecutionResult)
        assert result.language == Language.TRADITIONAL_CHINESE
        assert result.input_question == "如何創建 Python 類別？"
        assert len(result.reasoning_steps) > 0
        mock_llm_service.invoke.assert_called_once()
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_execute_qa_reasoning_exception(self, mock_create_llm, mock_get_config,
                                          mock_agent_config, mock_llm_service):
        """測試 QA 推理異常處理"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("QA reasoning failed")
        
        agent = ReproducerAgent()
        
        with pytest.raises(Exception, match="QA reasoning failed"):
            agent.execute_qa_reasoning("test question", Language.ENGLISH)
    
    @patch('src.services.llm_service.is_production', return_value=False)
    def test_extract_reasoning_steps(self, mock_is_production):
        """測試推理步驟提取"""
        # 測試編號步驟提取
        answer_with_numbers = """
        Here's the solution:
        1. Understanding the problem
        2. Analyzing requirements
        3. Implementing the solution
        4. Testing the code
        
        Additional details...
        """
        
        agent = ReproducerAgent()
        steps = agent._extract_reasoning_steps(answer_with_numbers)
        
        assert len(steps) == 4
        assert "1. Understanding the problem" in steps
        assert "4. Testing the code" in steps
    
    @patch('src.services.llm_service.is_production', return_value=False)
    def test_extract_reasoning_steps_chinese(self, mock_is_production):
        """測試中文推理步驟提取"""
        answer_chinese = """
        解決方案如下：
        一、理解問題
        二、分析需求
        三、實作解決方案
        四、測試程式碼
        
        其他詳細內容...
        """
        
        agent = ReproducerAgent()
        steps = agent._extract_reasoning_steps(answer_chinese)
        
        assert len(steps) >= 4
        assert any("一、理解問題" in step for step in steps)
    
    @patch('src.services.llm_service.is_production', return_value=False)
    def test_extract_reasoning_steps_keywords(self, mock_is_production):
        """測試關鍵字步驟提取"""
        answer_with_keywords = """
        首先，我們需要理解問題。
        然後，分析具體需求。
        接下來，實作解決方案。
        最後，測試我們的程式碼。
        """
        
        agent = ReproducerAgent()
        steps = agent._extract_reasoning_steps(answer_with_keywords)
        
        assert len(steps) >= 4
        assert any("首先" in step for step in steps)
        assert any("最後" in step for step in steps)
    
    @patch('src.services.llm_service.is_production', return_value=False)
    def test_extract_reasoning_steps_fallback(self, mock_is_production):
        """測試推理步驟提取回退機制"""
        answer_no_structure = """
        This is a simple answer without clear structure.
        It doesn't have numbered steps or clear keywords.
        Just continuous text explaining the solution.
        """
        
        agent = ReproducerAgent()
        steps = agent._extract_reasoning_steps(answer_no_structure)
        
        # 應該回退到段落分割
        assert len(steps) > 0
        assert len(steps) <= 5  # 最多5段
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_execute_comparative_qa(self, mock_create_llm, mock_get_config,
                                  mock_agent_config, mock_llm_service):
        """測試比較性 QA 執行"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # 設置兩次不同的回應
        responses = [
            AIMessage(content="English QA response with detailed explanation."),
            AIMessage(content="中文 QA 回應，包含詳細說明。")
        ]
        mock_llm_service.invoke.side_effect = responses
        
        agent = ReproducerAgent()
        
        original_qa, translated_qa = agent.execute_comparative_qa(
            "How do you create a Python class?",
            "如何創建 Python 類別？",
            "test_001"
        )
        
        assert isinstance(original_qa, QAExecutionResult)
        assert isinstance(translated_qa, QAExecutionResult)
        assert original_qa.record_id == "test_001"
        assert translated_qa.record_id == "test_001"
        assert original_qa.language == Language.ENGLISH
        assert translated_qa.language == Language.TRADITIONAL_CHINESE
        assert mock_llm_service.invoke.call_count == 2
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_execute_comparative_qa_exception(self, mock_create_llm, mock_get_config,
                                            mock_agent_config, mock_llm_service):
        """測試比較性 QA 異常處理"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("Comparative QA failed")
        
        agent = ReproducerAgent()
        
        with pytest.raises(Exception, match="Comparative QA failed"):
            agent.execute_comparative_qa(
                "test question", "測試問題", "test_001"
            )


class TestReproducerNode:
    """再現者節點測試類"""
    
    @patch.object(reproducer_module, 'ReproducerAgent')
    def test_reproducer_node_success(self, mock_agent_class, sample_workflow_state):
        """測試再現者節點成功執行"""
        # 設置 mock agent
        mock_agent = Mock()
        
        original_qa = QAExecutionResult(
            record_id="test_001",
            language=Language.ENGLISH,
            input_question="How do you create a Python class?",
            generated_answer="To create a Python class, use the class keyword...",
            execution_time=1.5,
            reasoning_steps=["Step 1", "Step 2"]
        )
        
        translated_qa = QAExecutionResult(
            record_id="test_001",
            language=Language.TRADITIONAL_CHINESE,
            input_question="如何創建 Python 類別？",
            generated_answer="要創建 Python 類別，使用 class 關鍵字...",
            execution_time=1.8,
            reasoning_steps=["步驟 1", "步驟 2"]
        )
        
        mock_agent.execute_comparative_qa.return_value = (original_qa, translated_qa)
        mock_agent_class.return_value = mock_agent
        
        result_state = reproducer_function(sample_workflow_state)
        
        assert result_state["original_qa_result"] == original_qa
        assert result_state["translated_qa_result"] == translated_qa
        assert result_state["processing_status"] == ProcessingStatus.PROCESSING
        mock_agent.execute_comparative_qa.assert_called_once_with(
            "How do you create a Python class?",
            "如何創建 Python 類別？",
            "test_001"
        )
    
    def test_reproducer_node_no_current_record(self):
        """測試再現者節點沒有當前記錄"""
        state = {
            "current_record": None,
            "processing_status": ProcessingStatus.PROCESSING,
            "retry_count": 0,
            "error_history": []
        }
        
        result_state = reproducer_function(state)
        
        assert result_state["processing_status"] == ProcessingStatus.FAILED
        assert len(result_state["error_history"]) > 0
    
    def test_reproducer_node_no_translation_result(self, sample_workflow_state):
        """測試再現者節點沒有翻譯結果"""
        sample_workflow_state["translation_result"] = None
        
        result_state = reproducer_function(sample_workflow_state)
        
        assert result_state["processing_status"] == ProcessingStatus.FAILED
        assert len(result_state["error_history"]) > 0
    
    def test_reproducer_node_existing_results(self, sample_workflow_state):
        """測試再現者節點已有 QA 結果"""
        # 添加已存在的 QA 結果
        existing_original = QAExecutionResult(
            record_id="test_001",
            language=Language.ENGLISH,
            input_question="existing question",
            generated_answer="existing answer",
            execution_time=1.0
        )
        
        existing_translated = QAExecutionResult(
            record_id="test_001",
            language=Language.TRADITIONAL_CHINESE,
            input_question="現有問題",
            generated_answer="現有答案",
            execution_time=1.0
        )
        
        sample_workflow_state["original_qa_result"] = existing_original
        sample_workflow_state["translated_qa_result"] = existing_translated
        sample_workflow_state["processing_status"] = ProcessingStatus.COMPLETED
        
        result_state = reproducer_function(sample_workflow_state)
        
        # 應該跳過執行
        assert result_state["original_qa_result"] == existing_original
        assert result_state["translated_qa_result"] == existing_translated
        assert result_state == sample_workflow_state
    
    def test_reproducer_node_retry_needed(self, sample_workflow_state):
        """測試再現者節點重試情況"""
        # 設置重試狀態
        sample_workflow_state["processing_status"] = ProcessingStatus.RETRY_NEEDED
        
        with patch.object(reproducer_module, 'ReproducerAgent') as mock_agent_class:
            mock_agent = Mock()
            
            new_original = QAExecutionResult(
                record_id="test_001",
                language=Language.ENGLISH,
                input_question="How do you create a Python class?",
                generated_answer="Retry answer for original",
                execution_time=1.2
            )
            
            new_translated = QAExecutionResult(
                record_id="test_001",
                language=Language.TRADITIONAL_CHINESE,
                input_question="如何創建 Python 類別？",
                generated_answer="重試的翻譯答案",
                execution_time=1.3
            )
            
            mock_agent.execute_comparative_qa.return_value = (new_original, new_translated)
            mock_agent_class.return_value = mock_agent
            
            result_state = reproducer_function(sample_workflow_state)
            
            assert result_state["original_qa_result"] == new_original
            assert result_state["translated_qa_result"] == new_translated
            assert result_state["processing_status"] == ProcessingStatus.PROCESSING
    
    @patch.object(reproducer_module, 'ReproducerAgent')
    def test_reproducer_node_exception(self, mock_agent_class, sample_workflow_state):
        """測試再現者節點異常處理"""
        mock_agent = Mock()
        mock_agent.execute_comparative_qa.side_effect = Exception("QA execution failed")
        mock_agent_class.return_value = mock_agent
        
        result_state = reproducer_function(sample_workflow_state)
        
        assert result_state["processing_status"] == ProcessingStatus.FAILED
        assert len(result_state["error_history"]) > 0
        error_record = result_state["error_history"][0]
        assert error_record.error_type == ErrorType.OTHER
        assert "QA execution failed" in error_record.error_message
        assert error_record.agent_name == "reproducer"


class TestReproducerAgentIntegration:
    """再現者 Agent 整合測試"""
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_full_workflow_simulation(self, mock_create_llm, mock_get_config, 
                                    mock_agent_config, mock_llm_service):
        """測試完整工作流模擬"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # 模擬完整的 QA 流程
        responses = [
            AIMessage(content="1. Understand the question\n2. Provide class syntax\n3. Show example"),
            AIMessage(content="一、理解問題\n二、提供類別語法\n三、顯示範例")
        ]
        mock_llm_service.invoke.side_effect = responses
        
        agent = ReproducerAgent()
        
        # 執行比較性 QA
        original_qa, translated_qa = agent.execute_comparative_qa(
            "How do you create a Python class?",
            "如何創建 Python 類別？",
            "integration_test_001"
        )
        
        # 驗證結果
        assert original_qa.record_id == "integration_test_001"
        assert translated_qa.record_id == "integration_test_001"
        assert original_qa.language != translated_qa.language
        assert len(original_qa.reasoning_steps) > 0
        assert len(translated_qa.reasoning_steps) > 0
        assert original_qa.execution_time >= 0  # Allow 0 in mocked tests
        assert translated_qa.execution_time >= 0  # Allow 0 in mocked tests
