"""
評估者節點測試模組
Test module for Evaluator Node

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

from src.workflow.nodes.evaluator_node import EvaluatorAgent, evaluator_node as evaluator_function
from src.models.dataset import QAExecutionResult, Language, ProcessingStatus, OriginalRecord
from src.models.quality import QualityAssessment, ErrorType
from src.constants.llm import LLMProvider, LLMModel


# Get the actual module for patching purposes
evaluator_module = importlib.import_module('src.workflow.nodes.evaluator_node')


@pytest.fixture
def mock_llm_service():
    """模擬 LLM 服務"""
    mock = Mock()
    mock.invoke.return_value = AIMessage(content="8.5")
    return mock


@pytest.fixture
def mock_agent_config():
    """模擬 agent 配置"""
    return {
        "primary_model": "gpt-4o",
        "fallback_model": "claude-4-sonnet",
        "temperature": 0.1,
        "max_tokens": 4096
    }


@pytest.fixture
def sample_qa_results():
    """創建測試用 QA 結果"""
    original_qa = QAExecutionResult(
        record_id="test_001",
        input_question="How do you create a Python class?",
        generated_answer="To create a Python class, use the `class` keyword:\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```",
        language=Language.ENGLISH,
        timestamp=time.time(),
        execution_time=1.5,
        reasoning_steps=["Identify class syntax", "Show example"]
    )
    
    translated_qa = QAExecutionResult(
        record_id="test_001",
        input_question="如何創建 Python 類別？",
        generated_answer="要創建 Python 類別，使用 `class` 關鍵字：\n```python\nclass MyClass:\n    def __init__(self):\n        pass\n```",
        language=Language.TRADITIONAL_CHINESE,
        timestamp=time.time(),
        execution_time=1.8,
        reasoning_steps=["識別類別語法", "顯示範例"]
    )
    
    return original_qa, translated_qa


@pytest.fixture
def sample_workflow_state(sample_qa_results):
    """創建測試用工作流狀態"""
    original_qa, translated_qa = sample_qa_results
    
    record = OriginalRecord(
        id="test_001",
        question="How do you create a Python class?",
        answer="To create a Python class, use the `class` keyword...",
        source_dataset="test_dataset",
        metadata={"tag": "python", "source_index": 1}
    )
    
    translation_result = Mock()
    translation_result.translated_question = "如何創建 Python 類別？"
    translation_result.translated_answer = "要創建 Python 類別，使用 `class` 關鍵字..."
    
    return {
        "current_record": record,
        "translation_result": translation_result,
        "original_qa_result": original_qa,
        "translated_qa_result": translated_qa,
        "processing_status": ProcessingStatus.PROCESSING,
        "retry_count": 0,
        "error_history": []
    }


class TestEvaluatorAgent:
    """評估者 Agent 測試類"""
    
    @patch('src.services.llm_service.is_production', return_value=False)
    def test_init(self, mock_is_production):
        """測試 EvaluatorAgent 初始化"""
        agent = EvaluatorAgent()
        
        # Verify the agent is initialized with correct config values
        assert agent.primary_model == "gpt-4o"
        assert agent.temperature == 0.1
        assert agent.max_tokens == 4096
        assert agent.fallback_model == "claude-4-sonnet"
        assert agent.llm_service is not None
    
    @patch.object(evaluator_module, 'get_agent_config')
    def test_init_no_config(self, mock_get_config):
        """測試初始化時配置不存在"""
        mock_get_config.return_value = None
        
        with pytest.raises(ValueError, match="Failed to get evaluator configuration"):
            EvaluatorAgent()
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_initialize_llm_service_openai(self, mock_create_llm, mock_get_config, mock_llm_service):
        """測試初始化 OpenAI LLM 服務"""
        config = {"primary_model": "gpt-4o", "temperature": 0.1, "max_tokens": 4000}
        mock_get_config.return_value = config
        mock_create_llm.return_value = mock_llm_service
        
        agent = EvaluatorAgent()
        
        mock_create_llm.assert_called_once_with(LLMProvider.OPENAI, LLMModel.GPT_4O)
    
    @patch.object(evaluator_module, 'get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_initialize_llm_service_claude(self, mock_create_llm, mock_get_config, mock_llm_service):
        """測試初始化 Claude LLM 服務"""
        config = {"primary_model": "claude-4-sonnet", "temperature": 0.1, "max_tokens": 4000}
        mock_get_config.return_value = config
        mock_create_llm.return_value = mock_llm_service
        
        agent = EvaluatorAgent()
        
        mock_create_llm.assert_called_once_with(LLMProvider.ANTHROPIC, LLMModel.CLAUDE_4_SONNET)
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_semantic_consistency(self, mock_create_llm, mock_get_config, 
                                         mock_agent_config, mock_llm_service, sample_qa_results):
        """測試語義一致性評估"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(content="8.5")
        
        agent = EvaluatorAgent()
        original_qa, translated_qa = sample_qa_results
        
        score = agent.evaluate_semantic_consistency(original_qa, translated_qa)
        
        assert score == 8.5
        mock_llm_service.invoke.assert_called_once()
        
        # 檢查 prompt 內容
        call_args = mock_llm_service.invoke.call_args[0][0]
        assert len(call_args) == 1
        assert "語義一致性" in call_args[0]["content"]
        assert original_qa.input_question in call_args[0]["content"]
        assert translated_qa.input_question in call_args[0]["content"]
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_semantic_consistency_invalid_response(self, mock_create_llm, mock_get_config, 
                                                          mock_agent_config, mock_llm_service, sample_qa_results):
        """測試語義一致性評估 - 無效回應"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(content="無法解析的回應")
        
        agent = EvaluatorAgent()
        original_qa, translated_qa = sample_qa_results
        
        score = agent.evaluate_semantic_consistency(original_qa, translated_qa)
        
        assert score == 5.0  # 預設分數
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_semantic_consistency_out_of_range(self, mock_create_llm, mock_get_config, 
                                                       mock_agent_config, mock_llm_service, sample_qa_results):
        """測試語義一致性評估 - 分數超出範圍"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(content="15.5")
        
        agent = EvaluatorAgent()
        original_qa, translated_qa = sample_qa_results
        
        score = agent.evaluate_semantic_consistency(original_qa, translated_qa)
        
        assert score == 10.0  # 限制在最大值
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_semantic_consistency_exception(self, mock_create_llm, mock_get_config, 
                                                   mock_agent_config, mock_llm_service, sample_qa_results):
        """測試語義一致性評估 - 異常處理"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("LLM service error")
        
        agent = EvaluatorAgent()
        original_qa, translated_qa = sample_qa_results
        
        score = agent.evaluate_semantic_consistency(original_qa, translated_qa)
        
        assert score == 5.0  # 異常時預設分數

    @patch.object(evaluator_module, 'get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_initialize_llm_service_google(self, mock_create_llm, mock_get_config, mock_llm_service):
        """測試初始化 Google Gemini LLM 服務"""
        config = {"primary_model": "gemini-2.5-flash", "temperature": 0.1, "max_tokens": 4000}
        mock_get_config.return_value = config
        mock_create_llm.return_value = mock_llm_service
        
        agent = EvaluatorAgent()
        
        mock_create_llm.assert_called_once_with(LLMProvider.GOOGLE, LLMModel.GEMINI_2_5_FLASH)

    @patch.object(evaluator_module, 'get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_initialize_llm_service_unknown_model(self, mock_create_llm, mock_get_config, mock_llm_service):
        """測試初始化未知模型 - 應該預設為 OpenAI"""
        config = {"primary_model": "unknown-model", "temperature": 0.1, "max_tokens": 4000}
        mock_get_config.return_value = config
        mock_create_llm.return_value = mock_llm_service
        
        # 這個測試應該拋出異常，因為 LLMModel enum 不接受未知的模型
        with pytest.raises(ValueError):
            EvaluatorAgent()

    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_initialize_llm_service_exception(self, mock_create_llm, mock_get_config):
        """測試 LLM 服務初始化異常"""
        config = {"primary_model": "gpt-4o", "temperature": 0.1, "max_tokens": 4000}
        mock_get_config.return_value = config
        mock_create_llm.side_effect = Exception("Failed to create LLM")
        
        with pytest.raises(Exception, match="Failed to create LLM"):
            EvaluatorAgent()
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_semantic_consistency_exception(self, mock_create_llm, mock_get_config, 
                                                   mock_agent_config, mock_llm_service, sample_qa_results):
        """測試語義一致性評估 - 異常情況"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("LLM 服務錯誤")
        
        agent = EvaluatorAgent()
        original_qa, translated_qa = sample_qa_results
        
        score = agent.evaluate_semantic_consistency(original_qa, translated_qa)
        
        assert score == 5.0  # 預設分數
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_code_integrity_consistent_blocks(self, mock_create_llm, mock_get_config, 
                                                      mock_agent_config, mock_llm_service):
        """測試程式碼完整性評估 - 一致的程式碼區塊"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # Mock LLM response to return expected score
        from langchain_core.messages import AIMessage
        mock_llm_service.invoke.return_value = AIMessage(content="8.0")
        
        agent = EvaluatorAgent()
        
        original = "Here's the code:\n```python\nprint('hello')\n```"
        translated = "這是程式碼：\n```python\nprint('hello')\n```"
        
        score = agent.evaluate_code_integrity(original, translated)
        
        assert score == 8.0  # 包含程式碼區塊
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_code_integrity_no_code(self, mock_create_llm, mock_get_config, 
                                           mock_agent_config, mock_llm_service):
        """測試程式碼完整性評估 - 沒有程式碼"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # Mock LLM response to return expected score
        from langchain_core.messages import AIMessage
        mock_llm_service.invoke.return_value = AIMessage(content="10.0")
        
        agent = EvaluatorAgent()
        
        original = "This is just text without code"
        translated = "這只是沒有程式碼的文字"
        
        score = agent.evaluate_code_integrity(original, translated)
        
        assert score == 10.0  # 沒有程式碼，視為完整
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_code_integrity_inconsistent_blocks(self, mock_create_llm, mock_get_config, 
                                                        mock_agent_config, mock_llm_service):
        """測試程式碼完整性評估 - 不一致的程式碼區塊"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # Mock LLM response to return expected score
        from langchain_core.messages import AIMessage
        mock_llm_service.invoke.return_value = AIMessage(content="5.0")
        
        agent = EvaluatorAgent()
        
        original = "Code: ```python\nprint('hello')\n```"
        translated = "程式碼：print('hello')"  # 缺少程式碼區塊標記
        
        score = agent.evaluate_code_integrity(original, translated)
        
        assert score == 5.0  # 程式碼區塊數量不一致
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_code_integrity_exception(self, mock_create_llm, mock_get_config, 
                                             mock_agent_config, mock_llm_service):
        """測試程式碼完整性評估 - 異常處理"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # Mock the LLM service to raise an exception
        mock_llm_service.invoke.side_effect = Exception("Mock LLM error")
        
        agent = EvaluatorAgent()
        
        # 創造會導致異常的情況（使用None值會引發AttributeError）
        score = agent.evaluate_code_integrity(None, "test")
            
        assert score == 7.0  # 異常時預設分數
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_translation_naturalness(self, mock_create_llm, mock_get_config, 
                                             mock_agent_config, mock_llm_service):
        """測試翻譯自然度評估"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(content="9.0")
        
        agent = EvaluatorAgent()
        
        translated_question = "如何創建 Python 類別？"
        translated_answer = "要創建 Python 類別，使用 class 關鍵字..."
        
        score = agent.evaluate_translation_naturalness(translated_question, translated_answer)
        
        assert score == 9.0
        mock_llm_service.invoke.assert_called_once()
        
        # 檢查 prompt 內容
        call_args = mock_llm_service.invoke.call_args[0][0]
        assert "自然度和流暢度" in call_args[0]["content"]
        assert translated_question in call_args[0]["content"]
        assert translated_answer in call_args[0]["content"]
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_evaluate_translation_naturalness_exception(self, mock_create_llm, mock_get_config, 
                                                       mock_agent_config, mock_llm_service):
        """測試翻譯自然度評估 - 異常處理"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.side_effect = Exception("LLM service error")
        
        agent = EvaluatorAgent()
        
        score = agent.evaluate_translation_naturalness("測試問題", "測試答案")
        
        assert score == 7.0  # 異常時預設分數
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_generate_improvement_suggestions_low_semantic(self, mock_create_llm, mock_get_config, 
                                                          mock_agent_config, mock_llm_service, sample_qa_results):
        """測試生成改進建議 - 低語義一致性分數"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # Mock LLM response to return suggestions
        from langchain_core.messages import AIMessage
        mock_llm_service.invoke.return_value = AIMessage(content="""
        - 改善語義一致性：確保翻譯保持原文的核心技術意圖
        - 檢查技術術語翻譯的準確性
        - 確保邏輯結構保持一致
        - 提升技術概念的表達清晰度
        """)
        
        agent = EvaluatorAgent()
        
        original_qa, translated_qa = sample_qa_results
        quality_scores = {
            "semantic_consistency": 6.0,
            "code_integrity": 8.0,
            "translation_naturalness": 9.0
        }
        
        suggestions = agent.generate_improvement_suggestions(
            original_qa, translated_qa, quality_scores
        )
        
        assert len(suggestions) == 4  # 實際建議數量
        assert any("語義一致性" in suggestion for suggestion in suggestions)
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_generate_improvement_suggestions_low_code_integrity(self, mock_create_llm, mock_get_config, 
                                                               mock_agent_config, mock_llm_service, sample_qa_results):
        """測試生成改進建議 - 低程式碼完整性分數"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # Mock LLM response to return suggestions
        from langchain_core.messages import AIMessage
        mock_llm_service.invoke.return_value = AIMessage(content="""
        - 保護程式碼完整性：確保程式碼區塊不被誤譯或遺失
        - 檢查程式碼格式的一致性
        - 確保程式語法結構保持正確
        """)
        
        agent = EvaluatorAgent()
        
        original_qa, translated_qa = sample_qa_results
        quality_scores = {
            "semantic_consistency": 9.0,
            "code_integrity": 6.0,
            "translation_naturalness": 9.0
        }
        
        suggestions = agent.generate_improvement_suggestions(
            original_qa, translated_qa, quality_scores
        )
        
        assert any("程式碼完整性" in suggestion for suggestion in suggestions)
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_generate_improvement_suggestions_low_naturalness(self, mock_create_llm, mock_get_config, 
                                                            mock_agent_config, mock_llm_service, sample_qa_results):
        """測試生成改進建議 - 低翻譯自然度分數"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        # Mock LLM response to return suggestions
        from langchain_core.messages import AIMessage
        mock_llm_service.invoke.return_value = AIMessage(content="""
        - 提升翻譯自然度：使用更符合繁體中文習慣的表達方式
        - 改善語言流暢性
        - 優化術語使用
        """)
        
        agent = EvaluatorAgent()
        
        original_qa, translated_qa = sample_qa_results
        quality_scores = {
            "semantic_consistency": 9.0,
            "code_integrity": 9.0,
            "translation_naturalness": 6.0
        }
        
        suggestions = agent.generate_improvement_suggestions(
            original_qa, translated_qa, quality_scores
        )
        
        assert any("翻譯自然度" in suggestion for suggestion in suggestions)
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_perform_quality_assessment(self, mock_create_llm, mock_get_config, 
                                       mock_agent_config, mock_llm_service, sample_qa_results):
        """測試執行品質評估"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        mock_llm_service.invoke.return_value = AIMessage(content="8.0")
        
        agent = EvaluatorAgent()
        original_qa, translated_qa = sample_qa_results
        
        quality_assessment = agent.perform_quality_assessment(
            original_qa, translated_qa,
            "如何創建 Python 類別？",
            "要創建 Python 類別...",
            "test_001"
        )
        
        assert isinstance(quality_assessment, QualityAssessment)
        assert quality_assessment.record_id == "test_001"
        assert quality_assessment.semantic_consistency_score == 8.0
        assert quality_assessment.code_integrity_score == 8.0
        assert quality_assessment.translation_naturalness_score == 8.0
        assert len(quality_assessment.improvement_suggestions) <= 5
    
    @patch('src.config.llm_config.get_agent_config')
    @patch('src.services.llm_service.LLMFactory.create_llm')
    def test_perform_quality_assessment_exception(self, mock_create_llm, mock_get_config, 
                                                 mock_agent_config, mock_llm_service, sample_qa_results):
        """測試品質評估 - 異常處理"""
        mock_get_config.return_value = mock_agent_config
        mock_create_llm.return_value = mock_llm_service
        
        agent = EvaluatorAgent()
        original_qa, translated_qa = sample_qa_results
        
        # 模擬評估過程中的異常
        with patch.object(agent, 'evaluate_semantic_consistency', side_effect=Exception("Assessment error")):
            with pytest.raises(Exception, match="Assessment error"):
                agent.perform_quality_assessment(
                    original_qa, translated_qa,
                    "如何創建 Python 類別？",
                    "要創建 Python 類別...",
                    "test_001"
                )


class TestEvaluatorNode:
    """評估者節點函數測試類"""
    
    @patch.object(evaluator_module, 'EvaluatorAgent')
    def test_evaluator_node_success(self, mock_agent_class, sample_workflow_state):
        """測試評估者節點成功執行"""
        # 設置 mock agent
        mock_agent = Mock()
        mock_agent.min_semantic_score = 7.0
        mock_agent.max_retry_attempts = 3
        
        # 設置品質評估結果
        quality_assessment = QualityAssessment(
            record_id="test_001",
            semantic_consistency_score=8.5,
            code_integrity_score=9.0,
            translation_naturalness_score=8.0,
            overall_quality_score=8.6,
            semantic_analysis="語義一致性分析：8.5/10",
            code_analysis="程式碼完整性分析：9.0/10",
            naturalness_analysis="翻譯自然度分析：8.0/10",
            improvement_suggestions=["建議1", "建議2"],
            evaluator_model="gpt-4o"
        )
        
        mock_agent.perform_quality_assessment.return_value = quality_assessment
        mock_agent_class.return_value = mock_agent
        
        # 執行節點
        result = evaluator_function(sample_workflow_state)
        
        # 驗證結果
        assert result["quality_assessment"] == quality_assessment
        assert result["processing_status"] == ProcessingStatus.COMPLETED
        assert "retry_count" not in result or result["retry_count"] == 0
    
    @patch.object(evaluator_module, 'EvaluatorAgent')
    def test_evaluator_node_retry_needed(self, mock_agent_class, sample_workflow_state):
        """測試評估者節點需要重試"""
        # 設置 mock agent
        mock_agent = Mock()
        mock_agent.min_semantic_score = 7.0
        mock_agent.max_retry_attempts = 3
        
        # 設置低分品質評估結果
        quality_assessment = QualityAssessment(
            record_id="test_001",
            semantic_consistency_score=5.0,
            code_integrity_score=6.0,
            translation_naturalness_score=5.5,
            overall_quality_score=5.5,
            semantic_analysis="語義一致性分析：5.0/10",
            code_analysis="程式碼完整性分析：6.0/10",
            naturalness_analysis="翻譯自然度分析：5.5/10",
            improvement_suggestions=["需要改善語義一致性", "需要提升翻譯自然度"],
            evaluator_model="gpt-4o"
        )
        
        mock_agent.perform_quality_assessment.return_value = quality_assessment
        mock_agent_class.return_value = mock_agent
        
        # 執行節點
        result = evaluator_function(sample_workflow_state)
        
        # 驗證結果
        assert result["quality_assessment"] == quality_assessment
        assert result["processing_status"] == ProcessingStatus.RETRY_NEEDED
        assert result["retry_count"] == 1
        assert result["improvement_suggestions"] == quality_assessment.improvement_suggestions
    
    @patch.object(evaluator_module, 'EvaluatorAgent')
    def test_evaluator_node_max_retries_reached(self, mock_agent_class, sample_workflow_state):
        """測試評估者節點達到最大重試次數"""
        # 設置已達最大重試次數的狀態
        sample_workflow_state["retry_count"] = 3
        
        # 設置 mock agent
        mock_agent = Mock()
        mock_agent.min_semantic_score = 7.0
        mock_agent.max_retry_attempts = 3
        
        # 設置低分品質評估結果
        quality_assessment = QualityAssessment(
            record_id="test_001",
            semantic_consistency_score=5.0,
            code_integrity_score=6.0,
            translation_naturalness_score=5.5,
            overall_quality_score=5.5,
            semantic_analysis="語義一致性分析：5.0/10",
            code_analysis="程式碼完整性分析：6.0/10",
            naturalness_analysis="翻譯自然度分析：5.5/10",
            improvement_suggestions=["需要改善語義一致性"],
            evaluator_model="gpt-4o"
        )
        
        mock_agent.perform_quality_assessment.return_value = quality_assessment
        mock_agent_class.return_value = mock_agent
        
        # 執行節點
        result = evaluator_function(sample_workflow_state)
        
        # 驗證結果
        assert result["quality_assessment"] == quality_assessment
        assert result["processing_status"] == ProcessingStatus.FAILED
        assert "improvement_suggestions" not in result
    
    def test_evaluator_node_missing_data(self):
        """測試評估者節點缺少必要數據"""
        incomplete_state = {
            "current_record": None,
            "error_history": []
        }
        
        result = evaluator_function(incomplete_state)
        
        assert result["processing_status"] == ProcessingStatus.FAILED
        assert len(result["error_history"]) == 1
        assert result["error_history"][0].error_type == ErrorType.OTHER
        assert "Missing required data" in result["error_history"][0].error_message
    
    @patch.object(evaluator_module, 'EvaluatorAgent')
    def test_evaluator_node_agent_exception(self, mock_agent_class, sample_workflow_state):
        """測試評估者節點 Agent 異常"""
        # 設置 mock agent 拋出異常
        mock_agent_class.side_effect = Exception("Agent 初始化失敗")
        
        # 執行節點
        result = evaluator_function(sample_workflow_state)
        
        # 驗證結果
        assert result["processing_status"] == ProcessingStatus.FAILED
        assert len(result["error_history"]) == 1
        assert result["error_history"][0].error_type == ErrorType.OTHER
        assert "Agent 初始化失敗" in result["error_history"][0].error_message


if __name__ == "__main__":
    pytest.main([__file__])
