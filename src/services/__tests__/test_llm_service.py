"""
LLM 服務測試模組
Test module for LLM Service

根據系統設計文檔的測試覆蓋率要求 (>= 90%)
"""

import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.services.llm_service import LLMService, LLMResponse, LLMFactory
from src.constants.llm import LLMProvider, LLMModel


@pytest.fixture
def mock_chat_model() -> Mock:
    """模擬 ChatModel"""
    mock = Mock()
    mock.invoke.return_value = AIMessage(content="模擬回應內容")
    
    # 為異步方法設置 AsyncMock
    async def mock_ainvoke(messages):
        return AIMessage(content="模擬異步回應內容")
    
    mock.ainvoke = mock_ainvoke
    return mock


@pytest.fixture
def llm_service(mock_chat_model: Mock) -> LLMService:
    """創建測試用 LLMService"""
    with patch('src.services.llm_service.LLMService._create_real_client') as mock_create:
        mock_create.return_value = mock_chat_model
        service = LLMService(
            provider=LLMProvider.OPENAI,
            model=LLMModel.GPT_4O
        )
        return service


class TestLLMService:
    """LLM 服務測試類"""

    def test_init(self) -> None:
        """測試初始化"""
        with patch('src.services.llm_service.LLMService._create_real_client') as mock_create:
            mock_create.return_value = Mock()
            service = LLMService(
                provider=LLMProvider.OPENAI,
                model=LLMModel.GPT_4O
            )
            
            assert service.provider == LLMProvider.OPENAI
            assert service.model == LLMModel.GPT_4O

    def test_invoke_success(self, llm_service: LLMService) -> None:
        """測試同步調用成功"""
        messages = [{"role": "user", "content": "測試問題"}]
        
        result = llm_service.invoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "模擬回應內容"
        assert result.model_name == "gpt-4o"
        assert result.provider == "openai"
        assert result.response_time >= 0  # 允許極快的回應時間
        assert result.timestamp > 0

    def test_invoke_with_human_message(self, llm_service: LLMService) -> None:
        """測試使用 HumanMessage 調用"""
        messages = [HumanMessage(content="測試人類訊息")]
        
        result = llm_service.invoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "模擬回應內容"

    def test_invoke_with_mixed_messages(self, llm_service: LLMService) -> None:
        """測試混合訊息類型"""
        messages = [
            {"role": "user", "content": "用戶訊息"},
            HumanMessage(content="人類訊息")
        ]
        
        result = llm_service.invoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "模擬回應內容"

    @pytest.mark.asyncio
    async def test_ainvoke_success(self, llm_service: LLMService) -> None:
        """測試異步調用成功"""
        messages = [{"role": "user", "content": "異步測試問題"}]
        
        result = await llm_service.ainvoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "模擬異步回應內容"
        assert result.model_name == "gpt-4o"
        assert result.provider == "openai"
        assert result.response_time >= 0  # 允許極快的回應時間
        assert result.timestamp > 0

    @pytest.mark.asyncio
    async def test_ainvoke_with_human_message(self, llm_service: LLMService) -> None:
        """測試異步調用使用 HumanMessage"""
        messages = [HumanMessage(content="異步人類訊息")]
        
        result = await llm_service.ainvoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "模擬異步回應內容"

    def test_invoke_error_handling(self, llm_service: LLMService) -> None:
        """測試錯誤處理"""
        llm_service._client.invoke.side_effect = Exception("模擬錯誤")
        messages = [{"role": "user", "content": "錯誤測試"}]
        
        with pytest.raises(Exception) as exc_info:
            llm_service.invoke(messages)
        
        assert "模擬錯誤" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ainvoke_error_handling(self, llm_service: LLMService) -> None:
        """測試異步錯誤處理"""
        async def mock_ainvoke_error(messages):
            raise Exception("異步模擬錯誤")
        
        llm_service._client.ainvoke = mock_ainvoke_error
        messages = [{"role": "user", "content": "異步錯誤測試"}]
        
        with pytest.raises(Exception) as exc_info:
            await llm_service.ainvoke(messages)
        
        assert "異步模擬錯誤" in str(exc_info.value)

    def test_convert_messages_dict_format(self, llm_service: LLMService) -> None:
        """測試訊息轉換 - 字典格式"""
        messages = [
            {"role": "user", "content": "用戶訊息"},
            {"role": "assistant", "content": "助手訊息"}
        ]
        
        converted = llm_service._convert_messages(messages)
        
        assert len(converted) == 2
        assert all(hasattr(msg, 'content') for msg in converted)

    def test_convert_messages_human_message_format(self, llm_service: LLMService) -> None:
        """測試訊息轉換 - HumanMessage 格式"""
        messages = [HumanMessage(content="人類訊息")]
        
        converted = llm_service._convert_messages(messages)
        
        assert len(converted) == 1
        assert converted[0].content == "人類訊息"

    def test_convert_messages_mixed_format(self, llm_service: LLMService) -> None:
        """測試訊息轉換 - 混合格式"""
        messages = [
            {"role": "user", "content": "字典訊息"},
            HumanMessage(content="物件訊息")
        ]
        
        converted = llm_service._convert_messages(messages)
        
        assert len(converted) == 2
        assert all(hasattr(msg, 'content') for msg in converted)

    @pytest.mark.performance
    def test_invoke_performance(self, llm_service: LLMService) -> None:
        """測試調用性能"""
        messages = [{"role": "user", "content": "性能測試"}]
        
        start_time = time.time()
        result = llm_service.invoke(messages)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # 應該在 5 秒內完成
        assert isinstance(result, LLMResponse)

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ainvoke_performance(self, llm_service: LLMService) -> None:
        """測試異步調用性能"""
        messages = [{"role": "user", "content": "異步性能測試"}]
        
        start_time = time.time()
        result = await llm_service.ainvoke(messages)
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # 應該在 5 秒內完成
        assert isinstance(result, LLMResponse)


class TestLLMFactory:
    """LLM 工廠測試類"""

    def test_create_openai_service(self) -> None:
        """測試創建 OpenAI 服務"""
        with patch('src.services.llm_service.LLMService._create_real_client') as mock_create:
            mock_create.return_value = Mock()
            service = LLMFactory.create_llm(
                provider=LLMProvider.OPENAI,
                model=LLMModel.GPT_4O
            )
            
            assert isinstance(service, LLMService)
            assert service.provider == LLMProvider.OPENAI
            assert service.model == LLMModel.GPT_4O

    def test_create_anthropic_service(self) -> None:
        """測試創建 Anthropic 服務"""
        with patch('src.services.llm_service.LLMService._create_real_client') as mock_create:
            mock_create.return_value = Mock()
            service = LLMFactory.create_llm(
                provider=LLMProvider.ANTHROPIC,
                model=LLMModel.CLAUDE_4_SONNET
            )
            
            assert isinstance(service, LLMService)
            assert service.provider == LLMProvider.ANTHROPIC
            assert service.model == LLMModel.CLAUDE_4_SONNET

    def test_create_google_service(self) -> None:
        """測試創建 Google 服務"""
        with patch('src.services.llm_service.LLMService._create_real_client') as mock_create:
            mock_create.return_value = Mock()
            service = LLMFactory.create_llm(
                provider=LLMProvider.GOOGLE,
                model=LLMModel.GEMINI_2_5_FLASH
            )
            
            assert isinstance(service, LLMService)
            assert service.provider == LLMProvider.GOOGLE
            assert service.model == LLMModel.GEMINI_2_5_FLASH

    def test_create_ollama_service(self) -> None:
        """測試創建 Ollama 服務"""
        with patch('src.services.llm_service.LLMService._create_real_client') as mock_create:
            mock_create.return_value = Mock()
            service = LLMFactory.create_llm(
                provider=LLMProvider.OLLAMA,
                model=LLMModel.LLAMA3_1
            )
            
            assert isinstance(service, LLMService)
            assert service.provider == LLMProvider.OLLAMA
            assert service.model == LLMModel.LLAMA3_1

    @pytest.mark.integration
    def test_create_and_invoke_integration(self) -> None:
        """測試創建服務並調用的整合測試"""
        with patch('src.services.llm_service.LLMService._create_real_client') as mock_create:
            mock_client = Mock()
            mock_client.invoke.return_value = AIMessage(content="整合測試回應")
            mock_create.return_value = mock_client
            
            service = LLMFactory.create_llm(
                provider=LLMProvider.OPENAI,
                model=LLMModel.GPT_4O
            )
            
            messages = [{"role": "user", "content": "整合測試問題"}]
            result = service.invoke(messages)
            
            assert isinstance(result, LLMResponse)
            assert result.content == "整合測試回應"
            assert result.provider == "openai"
            assert result.model_name == "gpt-4o"
