"""
分析設計者節點測試模組
Test module for Analyzer Designer Node

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import sys
import importlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.workflow.nodes.analyzer_designer_node import (
    AnalyzerDesignerAgent, 
    analyzer_designer_node as analyzer_designer_function
)
from src.models.dataset import OriginalRecord, TranslationResult, ProcessingStatus
from src.models.quality import ErrorType
from src.constants.llm import LLMProvider, LLMModel


# Get the actual module for patching purposes
analyzer_designer_module = importlib.import_module('src.workflow.nodes.analyzer_designer_node')


@pytest.fixture
def mock_llm_service():
    """模擬 LLM 服務"""
    mock = Mock()
    mock.invoke.return_value = AIMessage(content="模擬 LLM 回應")
    return mock


@pytest.fixture
def mock_agent_config():
    """模擬 agent 配置 - 從 agent_models.json 動態載入"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "agent_models.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    agent_config = config_data["agents"]["analyzer_designer"]
    return {
        "primary_model": agent_config["primary_model"],
        "fallback_model": agent_config["fallback_models"][0] if agent_config["fallback_models"] else None,
        "temperature": agent_config["model_parameters"]["temperature"],
        "max_tokens": agent_config["model_parameters"]["max_tokens"]
    }


@pytest.fixture
def sample_dataset_record():
    """創建測試用資料集記錄 - 別名為 sample_original_record"""
    return OriginalRecord(
        id="test_001",
        question="How do you create a Python class?",
        answer="To create a Python class, use the `class` keyword:\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```\nThis creates a basic class structure.",
        source_dataset="test_dataset",
        metadata={"tag": "python", "source_index": 1}
    )


@pytest.fixture
def sample_original_record():
    """創建測試用資料集記錄"""
    return OriginalRecord(
        id="test_001",
        question="How do you create a Python class?",
        answer="To create a Python class, use the `class` keyword:\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```\nThis creates a basic class structure.",
        source_dataset="test_dataset",
        metadata={"tag": "python", "source_index": 1}
    )


@pytest.fixture
def sample_semantic_context():
    """創建測試用語義複雜度上下文"""
    return {
        "complexity": "Medium",
        "programming_languages": ["Python"],
        "key_concepts": ["class", "object-oriented programming"],
        "code_block_count": 1,
        "translation_challenges": ["technical_terms", "code_protection"]
    }


@pytest.fixture
def sample_workflow_state(sample_original_record):
    """創建測試用工作流狀態"""
    return {
        "current_record": sample_original_record,
        "processing_status": ProcessingStatus.PENDING,
        "retry_count": 0,
        "error_history": []
    }


class TestAnalyzerDesignerAgent:
    """分析設計者 Agent 測試類"""
    
    def test_init(self, mock_agent_config):
        """測試 AnalyzerDesignerAgent 初始化"""
        with patch('src.config.llm_config.get_agent_config') as mock_get_config:
            # Mock the config to return data from agent_models.json
            mock_get_config.return_value = mock_agent_config
            
            with patch('src.services.llm_service.LLMFactory.create_llm') as mock_create_llm:
                mock_create_llm.return_value = Mock()
                
                agent = AnalyzerDesignerAgent()
                
                # Verify the agent is initialized with correct config values
                assert agent.primary_model == mock_agent_config["primary_model"]
                assert agent.temperature == mock_agent_config["temperature"]
                assert agent.max_tokens == mock_agent_config["max_tokens"]
                assert agent.fallback_model == mock_agent_config["fallback_model"]
                assert agent.llm_service is not None
    
    @patch.object(analyzer_designer_module, 'get_agent_config')
    def test_init_no_config(self, mock_get_config):
        """測試初始化時配置不存在"""
        mock_get_config.return_value = None
        
        with pytest.raises(ValueError, match="Failed to get analyzer_designer configuration"):
            AnalyzerDesignerAgent()
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_initialize_llm_service(self, mock_create_llm, mock_get_config, mock_llm_service, mock_agent_config):
        """測試初始化 LLM 服務"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        agent = AnalyzerDesignerAgent()
        
        # Determine expected provider and model based on primary_model
        if mock_agent_config["primary_model"] == "gpt-4o":
            expected_provider = LLMProvider.OPENAI
            expected_model = LLMModel.GPT_4O
        elif mock_agent_config["primary_model"] == "gemini-2.5-flash":
            expected_provider = LLMProvider.GOOGLE
            expected_model = LLMModel.GEMINI_2_5_FLASH
        else:
            # For other models, we'll need to add more cases
            expected_provider = LLMProvider.OPENAI
            expected_model = LLMModel.GPT_4O
        
        mock_create_llm.assert_called_once_with(expected_provider, expected_model)
    
    @patch.object(analyzer_designer_module, 'get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_initialize_llm_service_claude(self, mock_create_llm, mock_get_config, mock_llm_service):
        """測試初始化 Claude LLM 服務"""
        config = {"primary_model": "claude-4-sonnet", "temperature": 0.2, "max_tokens": 4000}
        mock_get_config.return_value = config
        mock_create_llm.return_value = mock_llm_service
        
        agent = AnalyzerDesignerAgent()
        
        assert agent.primary_model == "claude-4-sonnet", f"Agent got wrong model: {agent.primary_model}"
        
        mock_create_llm.assert_called_once_with(LLMProvider.ANTHROPIC, LLMModel.CLAUDE_4_SONNET)
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_analyze_semantic_complexity(self, mock_create_llm, mock_get_config, 
                                       mock_agent_config, mock_llm_service):
        """測試語義複雜度分析"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        agent = AnalyzerDesignerAgent()
        
        question = "How do you create a Python class?"
        answer = "To create a Python class:\n```python\nclass MyClass:\n    pass\n```"
        
        context = agent.analyze_semantic_complexity(question, answer)
        
        # 驗證返回的上下文結構
        assert isinstance(context, dict)
        assert "complexity" in context
        assert "programming_languages" in context
        assert "key_concepts" in context
        assert "code_block_count" in context
        assert "translation_challenges" in context
        
        # 驗證程式碼區塊計數
        assert context["code_block_count"] == 1  # 一個 ```python``` 區塊
        
        mock_llm_service.invoke.assert_called_once()
        
        # 檢查 prompt 內容
        call_args = mock_llm_service.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert "語義複雜度" in call_args[0]["content"]
        assert question in call_args[0]["content"]
        assert answer in call_args[0]["content"]
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_analyze_semantic_complexity_no_code_blocks(self, mock_create_llm, mock_get_config, 
                                                       mock_agent_config, mock_llm_service):
        """測試語義複雜度分析 - 沒有程式碼區塊"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        agent = AnalyzerDesignerAgent()
        
        question = "What is Python?"
        answer = "Python is a programming language."
        
        context = agent.analyze_semantic_complexity(question, answer)
        
        assert context["code_block_count"] == 0  # 沒有程式碼區塊
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_analyze_semantic_complexity_exception(self, mock_create_llm, mock_get_config, 
                                                 mock_agent_config, mock_llm_service):
        """測試語義複雜度分析 - 異常情況"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("LLM 服務錯誤")
        
        agent = AnalyzerDesignerAgent()
        
        question = "How do you create a Python class?"
        answer = "To create a Python class..."
        
        context = agent.analyze_semantic_complexity(question, answer)
        
        # 驗證回退到預設值
        assert context["complexity"] == "Unknown"
        assert context["programming_languages"] == ["Python"]  # 應該檢測到 Python（從問題文本中）
        assert context["key_concepts"] == ["object_oriented"]  # 應該檢測到面向對象概念（從 "class" 關鍵詞）
        assert context["translation_challenges"] == []
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_translate_question(self, mock_create_llm, mock_get_config, 
                              mock_agent_config, mock_llm_service, sample_semantic_context):
        """測試問題翻譯"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(content="如何創建 Python 類別？")
        
        agent = AnalyzerDesignerAgent()
        
        question = "How do you create a Python class?"
        translated = agent.translate_question(question, sample_semantic_context)
        
        assert translated == "如何創建 Python 類別？"
        mock_llm_service.invoke.assert_called_once()
        
        # 檢查 prompt 內容
        call_args = mock_llm_service.invoke.call_args[0][0]
        assert question in call_args[0]["content"]
        assert "繁體中文" in call_args[0]["content"]
        assert sample_semantic_context["complexity"] in call_args[0]["content"]
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_translate_question_exception(self, mock_create_llm, mock_get_config, 
                                        mock_agent_config, mock_llm_service, sample_semantic_context):
        """測試問題翻譯 - 異常情況"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("翻譯失敗")
        
        agent = AnalyzerDesignerAgent()
        
        question = "How do you create a Python class?"
        
        with pytest.raises(Exception, match="翻譯失敗"):
            agent.translate_question(question, sample_semantic_context)
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_translate_answer(self, mock_create_llm, mock_get_config, 
                            mock_agent_config, mock_llm_service, sample_semantic_context):
        """測試答案翻譯"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(content="要創建 Python 類別，使用 `class` 關鍵字：\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```")
        
        agent = AnalyzerDesignerAgent()
        
        question = "How do you create a Python class?"
        answer = "To create a Python class, use the `class` keyword:\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```"
        translated_question = "如何創建 Python 類別？"
        
        translated = agent.translate_answer(question, answer, translated_question, sample_semantic_context)
        
        assert "要創建 Python 類別" in translated
        assert "```python" in translated  # 確保程式碼區塊被保留
        mock_llm_service.invoke.assert_called_once()
        
        # 檢查 prompt 內容
        call_args = mock_llm_service.invoke.call_args[0][0]
        assert question in call_args[0]["content"]
        assert answer in call_args[0]["content"]
        assert translated_question in call_args[0]["content"]
        assert "嚴格保護程式碼片段" in call_args[0]["content"]
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_translate_answer_exception(self, mock_create_llm, mock_get_config, 
                                      mock_agent_config, mock_llm_service, sample_semantic_context):
        """測試答案翻譯 - 異常情況"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("答案翻譯失敗")
        
        agent = AnalyzerDesignerAgent()
        
        question = "How do you create a Python class?"
        answer = "To create a Python class..."
        translated_question = "如何創建 Python 類別？"
        
        with pytest.raises(Exception, match="答案翻譯失敗"):
            agent.translate_answer(question, answer, translated_question, sample_semantic_context)
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_perform_translation_success(self, mock_create_llm, mock_get_config, 
                                       mock_agent_config, mock_llm_service, sample_dataset_record):
        """測試執行完整翻譯流程 - 成功"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # 設置 LLM 回應序列
        mock_responses = [
            AIMessage(content="語義複雜度分析回應"),  # analyze_semantic_complexity
            AIMessage(content="如何創建 Python 類別？"),  # translate_question
            AIMessage(content="要創建 Python 類別，使用 `class` 關鍵字：\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```")  # translate_answer
        ]
        mock_llm_service.invoke.side_effect = mock_responses
        
        agent = AnalyzerDesignerAgent()
        
        result = agent.perform_translation(
            sample_dataset_record.id,
            sample_dataset_record.question,
            sample_dataset_record.answer
        )
        
        # 驗證結果
        assert isinstance(result, TranslationResult)
        assert result.original_record_id == sample_dataset_record.id
        assert result.translated_question == "如何創建 Python 類別？"
        assert "要創建 Python 類別" in result.translated_answer
        assert result.translation_strategy == "sequential_context_aware"
        assert len(result.terminology_notes) > 0
        
        # 驗證調用次數
        assert mock_llm_service.invoke.call_count == 3
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_perform_translation_exception(self, mock_create_llm, mock_get_config, 
                                         mock_agent_config, mock_llm_service, sample_dataset_record):
        """測試執行翻譯流程 - 異常情況"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("翻譯流程失敗")
        
        agent = AnalyzerDesignerAgent()
        
        with pytest.raises(Exception, match="翻譯流程失敗"):
            agent.perform_translation(
                sample_dataset_record.id,
                sample_dataset_record.question,
                sample_dataset_record.answer
            )


class TestAnalyzerDesignerNode:
    """分析設計者節點函數測試類"""
    
    def test_analyzer_designer_node_success(self, sample_workflow_state):
        """測試分析設計者節點成功執行"""
        with patch.object(analyzer_designer_module, 'AnalyzerDesignerAgent') as mock_agent_class:
            # 設置 mock agent
            mock_agent = Mock()
            translation_result = TranslationResult(
                original_record_id="test_001",
                translated_question="如何創建 Python 類別？",
                translated_answer="要創建 Python 類別，使用 `class` 關鍵字...",
                translation_strategy="sequential_context_aware",
                terminology_notes=["Complexity: Medium", "Languages: Python", "Code blocks: 1"]
            )
            mock_agent.perform_translation.return_value = translation_result
            mock_agent_class.return_value = mock_agent
            
            # 執行節點
            result = analyzer_designer_function(sample_workflow_state)
            
            # 驗證結果
            assert result["translation_result"] == translation_result
            assert result["processing_status"] == ProcessingStatus.PROCESSING
            mock_agent.perform_translation.assert_called_once_with(
                "test_001",
                "How do you create a Python class?",
                sample_workflow_state["current_record"].answer
            )
    
    @patch.object(analyzer_designer_module, 'AnalyzerDesignerAgent')
    def test_analyzer_designer_node_existing_translation(self, mock_agent_class, sample_workflow_state):
        """測試分析設計者節點 - 已存在翻譯結果"""
        # 設置已存在的翻譯結果
        existing_translation = TranslationResult(
            original_record_id="test_001",
            translated_question="已存在的翻譯",
            translated_answer="已存在的翻譯答案",
            translation_strategy="existing",
            terminology_notes=[]
        )
        sample_workflow_state["translation_result"] = existing_translation
        sample_workflow_state["processing_status"] = ProcessingStatus.PROCESSING
        
        # 執行節點
        result = analyzer_designer_function(sample_workflow_state)
        
        # 驗證結果 - 應該跳過翻譯
        assert result["translation_result"] == existing_translation
        mock_agent_class.assert_not_called()
    
    @patch.object(analyzer_designer_module, 'AnalyzerDesignerAgent')
    def test_analyzer_designer_node_retry_needed(self, mock_agent_class, sample_workflow_state):
        """測試分析設計者節點 - 需要重試"""
        # 設置重試狀態
        sample_workflow_state["processing_status"] = ProcessingStatus.RETRY_NEEDED
        sample_workflow_state["improvement_suggestions"] = ["舊的改進建議"]
        
        # 設置 mock agent
        mock_agent = Mock()
        translation_result = TranslationResult(
            original_record_id="test_001",
            translated_question="重新翻譯的問題",
            translated_answer="重新翻譯的答案",
            translation_strategy="sequential_context_aware",
            terminology_notes=[]
        )
        mock_agent.perform_translation.return_value = translation_result
        mock_agent_class.return_value = mock_agent
        
        # 執行節點
        result = analyzer_designer_function(sample_workflow_state)
        
        # 驗證結果
        assert result["translation_result"] == translation_result
        assert result["processing_status"] == ProcessingStatus.PROCESSING
        assert result["improvement_suggestions"] == []  # 重試時清空改進建議
    
    def test_analyzer_designer_node_no_current_record(self):
        """測試分析設計者節點 - 沒有當前記錄"""
        incomplete_state = {
            "current_record": None,
            "error_history": []
        }
        
        result = analyzer_designer_function(incomplete_state)
        
        assert result["processing_status"] == ProcessingStatus.FAILED
        assert len(result["error_history"]) == 1
        assert result["error_history"][0].error_type == ErrorType.TRANSLATION_QUALITY
        assert "No current record" in result["error_history"][0].error_message
    
    @patch.object(analyzer_designer_module, 'AnalyzerDesignerAgent')
    def test_analyzer_designer_node_agent_exception(self, mock_agent_class, sample_workflow_state):
        """測試分析設計者節點 - Agent 異常"""
        # 設置 mock agent 拋出異常
        mock_agent_class.side_effect = Exception("Agent 初始化失敗")
        
        # 執行節點
        result = analyzer_designer_function(sample_workflow_state)
        
        # 驗證結果
        assert result["processing_status"] == ProcessingStatus.FAILED
        assert len(result["error_history"]) == 1
        assert result["error_history"][0].error_type == ErrorType.TRANSLATION_QUALITY
        assert "Agent 初始化失敗" in result["error_history"][0].error_message
    
    @patch.object(analyzer_designer_module, 'AnalyzerDesignerAgent')
    def test_analyzer_designer_node_translation_exception(self, mock_agent_class, sample_workflow_state):
        """測試分析設計者節點 - 翻譯過程異常"""
        # 設置 mock agent 翻譯失敗
        mock_agent = Mock()
        mock_agent.perform_translation.side_effect = Exception("翻譯過程失敗")
        mock_agent_class.return_value = mock_agent
        
        # 執行節點
        result = analyzer_designer_function(sample_workflow_state)
        
        # 驗證結果
        assert result["processing_status"] == ProcessingStatus.FAILED
        assert len(result["error_history"]) == 1
        assert result["error_history"][0].error_type == ErrorType.TRANSLATION_QUALITY
        assert "翻譯過程失敗" in result["error_history"][0].error_message


class TestSemanticComplexityContext:
    """語義複雜度上下文測試類"""
    
    def test_semantic_complexity_context_structure(self, sample_semantic_context):
        """測試語義複雜度上下文結構"""
        context = sample_semantic_context
        
        # 驗證必需字段
        assert "complexity" in context
        assert "programming_languages" in context
        assert "key_concepts" in context
        assert "code_block_count" in context
        assert "translation_challenges" in context
        
        # 驗證數據類型
        assert isinstance(context["complexity"], str)
        assert isinstance(context["programming_languages"], list)
        assert isinstance(context["key_concepts"], list)
        assert isinstance(context["code_block_count"], int)
        assert isinstance(context["translation_challenges"], list)
        
        # 驗證值的有效性
        assert context["complexity"] in ["Simple", "Medium", "Complex"]
        assert context["code_block_count"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
