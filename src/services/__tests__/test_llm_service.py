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

from src.services.llm_service import LLMService, LLMResponse, LLMFactory, MissingAPIKeyError
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
    with patch('src.services.llm_service.init_chat_model') as mock_init, \
         patch.object(LLMService, '_check_api_key', return_value=True):
        mock_init.return_value = mock_chat_model
        service = LLMService(
            provider=LLMProvider.OPENAI,
            model=LLMModel.GPT_4O
        )
        return service


class TestLLMService:
    """LLM 服務測試類"""

    def test_init(self) -> None:
        """測試初始化"""
        with patch('src.services.llm_service.init_chat_model') as mock_init, \
             patch.object(LLMService, '_check_api_key', return_value=True):
            mock_init.return_value = Mock()
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

    def test_invoke_empty_response_retry(self, llm_service: LLMService) -> None:
        """測試空回應的重試機制"""
        # 第一次返回空內容，第二次返回有效內容
        llm_service._client.invoke.side_effect = [
            AIMessage(content=""),  # 空回應
            AIMessage(content="有效的回應內容")  # 有效回應
        ]
        
        messages = [{"role": "user", "content": "測試空回應重試"}]
        result = llm_service.invoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "有效的回應內容"
        # 應該調用了兩次
        assert llm_service._client.invoke.call_count == 2
    
    def test_invoke_empty_response_all_retries_fail(self, llm_service: LLMService) -> None:
        """測試所有重試都返回空回應"""
        llm_service._client.invoke.return_value = AIMessage(content="")
        
        messages = [{"role": "user", "content": "測試空回應失敗"}]
        
        with pytest.raises(ValueError) as exc_info:
            llm_service.invoke(messages)
        
        assert "empty response" in str(exc_info.value).lower()
        # 應該調用了 4 次（1 次初始 + 3 次重試）
        assert llm_service._client.invoke.call_count == 4
    
    def test_invoke_retry_on_exception(self, llm_service: LLMService) -> None:
        """測試異常情況下的重試機制"""
        # 前兩次拋出異常，第三次成功
        llm_service._client.invoke.side_effect = [
            Exception("第一次失敗"),
            Exception("第二次失敗"), 
            AIMessage(content="最終成功")
        ]
        
        messages = [{"role": "user", "content": "測試異常重試"}]
        result = llm_service.invoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "最終成功"
        # 應該調用了三次
        assert llm_service._client.invoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_ainvoke_empty_response_retry(self, llm_service: LLMService) -> None:
        """測試異步空回應的重試機制"""
        call_count = 0
        
        async def mock_ainvoke_empty_first(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content="")
            else:
                return AIMessage(content="有效的異步回應內容")
        
        llm_service._client.ainvoke = mock_ainvoke_empty_first
        
        messages = [{"role": "user", "content": "測試異步空回應重試"}]
        result = await llm_service.ainvoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "有效的異步回應內容"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_ainvoke_empty_response_all_retries_fail(self, llm_service: LLMService) -> None:
        """測試異步所有重試都返回空回應"""
        async def mock_ainvoke_empty(messages):
            return AIMessage(content="")
        
        llm_service._client.ainvoke = mock_ainvoke_empty
        
        messages = [{"role": "user", "content": "測試異步空回應失敗"}]
        
        with pytest.raises(ValueError) as exc_info:
            await llm_service.ainvoke(messages)
        
        assert "empty response" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_ainvoke_retry_on_exception(self, llm_service: LLMService) -> None:
        """測試異步異常情況下的重試機制"""
        call_count = 0
        
        async def mock_ainvoke_with_retry(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"異步失敗 {call_count}")
            else:
                return AIMessage(content="異步最終成功")
        
        llm_service._client.ainvoke = mock_ainvoke_with_retry
        
        messages = [{"role": "user", "content": "測試異步異常重試"}]
        result = await llm_service.ainvoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert result.content == "異步最終成功"
        assert call_count == 3

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
        with patch('src.services.llm_service.init_chat_model') as mock_init, \
             patch.object(LLMService, '_check_api_key', return_value=True):
            mock_init.return_value = Mock()
            service = LLMFactory.create_llm(
                provider=LLMProvider.OPENAI,
                model=LLMModel.GPT_4O
            )
            
            assert isinstance(service, LLMService)
            assert service.provider == LLMProvider.OPENAI
            assert service.model == LLMModel.GPT_4O

    def test_create_anthropic_service(self) -> None:
        """測試創建 Anthropic 服務"""
        with patch('src.services.llm_service.init_chat_model') as mock_init, \
             patch.object(LLMService, '_check_api_key', return_value=True):
            mock_init.return_value = Mock()
            service = LLMFactory.create_llm(
                provider=LLMProvider.ANTHROPIC,
                model=LLMModel.CLAUDE_4_SONNET
            )
            
            assert isinstance(service, LLMService)
            assert service.provider == LLMProvider.ANTHROPIC
            assert service.model == LLMModel.CLAUDE_4_SONNET

    def test_create_google_service(self) -> None:
        """測試創建 Google 服務"""
        with patch('src.services.llm_service.init_chat_model') as mock_init, \
             patch.object(LLMService, '_check_api_key', return_value=True):
            mock_init.return_value = Mock()
            service = LLMFactory.create_llm(
                provider=LLMProvider.GOOGLE,
                model=LLMModel.GEMINI_2_5_FLASH
            )
            
            assert isinstance(service, LLMService)
            assert service.provider == LLMProvider.GOOGLE
            assert service.model == LLMModel.GEMINI_2_5_FLASH

    def test_create_ollama_service(self) -> None:
        """測試創建 Ollama 服務"""
        with patch('src.services.llm_service.init_chat_model') as mock_init, \
             patch.object(LLMService, '_check_api_key', return_value=True):
            mock_init.return_value = Mock()
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
        with patch('src.services.llm_service.init_chat_model') as mock_init, \
             patch.object(LLMService, '_check_api_key', return_value=True):
            mock_client = Mock()
            mock_client.invoke.return_value = AIMessage(content="整合測試回應")
            mock_init.return_value = mock_client
            
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


class TestProductionModeAPIKeyValidation:
    """測試生產模式下的 API 密鑰驗證"""

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch.dict('os.environ', {}, clear=True)
    def test_production_mode_missing_api_key_error_details(self, mock_is_production) -> None:
        """測試生產模式下缺少 API 密鑰的詳細錯誤信息"""
        with pytest.raises(MissingAPIKeyError) as exc_info:
            LLMService(LLMProvider.OPENAI, LLMModel.GPT_4O)
        
        error_message = str(exc_info.value)
        assert "在生產環境中必須配置 OPENAI_API_KEY" in error_message
        assert "請執行以下步驟" in error_message
        assert "cp .env.example .env" in error_message
        assert "OPENAI_API_KEY=your_actual_api_key" in error_message

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch.dict('os.environ', {}, clear=True)
    def test_production_mode_missing_anthropic_key_error_details(self, mock_is_production) -> None:
        """測試生產模式下缺少 Anthropic API 密鑰的詳細錯誤信息"""
        with pytest.raises(MissingAPIKeyError) as exc_info:
            LLMService(LLMProvider.ANTHROPIC, LLMModel.CLAUDE_4_SONNET)
        
        error_message = str(exc_info.value)
        assert "在生產環境中必須配置 ANTHROPIC_API_KEY" in error_message

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch.dict('os.environ', {}, clear=True)
    def test_production_mode_missing_google_key_error_details(self, mock_is_production) -> None:
        """測試生產模式下缺少 Google API 密鑰的詳細錯誤信息"""
        with pytest.raises(MissingAPIKeyError) as exc_info:
            LLMService(LLMProvider.GOOGLE, LLMModel.GEMINI_2_5_FLASH)
        
        error_message = str(exc_info.value)
        assert "在生產環境中必須配置 GOOGLE_API_KEY" in error_message

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch('src.services.llm_service.init_chat_model')
    @patch.dict('os.environ', {}, clear=True)
    def test_production_mode_ollama_no_api_key_required(self, mock_init, mock_is_production) -> None:
        """測試生產模式下 Ollama 不需要 API 密鑰"""
        mock_init.return_value = Mock()
        
        # Ollama 不應該拋出 MissingAPIKeyError
        service = LLMService(LLMProvider.OLLAMA, LLMModel.LLAMA3_1)
        assert service.provider == LLMProvider.OLLAMA
        assert service.model == LLMModel.LLAMA3_1

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch.dict('os.environ', {}, clear=True)
    def test_production_mode_missing_anthropic_api_key(self, mock_is_production) -> None:
        """測試生產模式下缺少 Anthropic API 密鑰"""
        with pytest.raises(MissingAPIKeyError) as exc_info:
            LLMService(LLMProvider.ANTHROPIC, LLMModel.CLAUDE_4_SONNET)
        
        error_message = str(exc_info.value)
        assert "在生產環境中必須配置 ANTHROPIC_API_KEY" in error_message

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch.dict('os.environ', {}, clear=True)
    def test_production_mode_missing_google_api_key(self, mock_is_production) -> None:
        """測試生產模式下缺少 Google API 密鑰"""
        with pytest.raises(MissingAPIKeyError) as exc_info:
            LLMService(LLMProvider.GOOGLE, LLMModel.GEMINI_2_5_FLASH)
        
        error_message = str(exc_info.value)
        assert "在生產環境中必須配置 GOOGLE_API_KEY" in error_message

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch('src.services.llm_service.init_chat_model')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    def test_production_mode_with_valid_api_key(self, mock_init, mock_is_production) -> None:
        """測試生產模式下有有效 API 密鑰"""
        mock_init.return_value = Mock()
        
        # 不應該拋出異常
        service = LLMService(LLMProvider.OPENAI, LLMModel.GPT_4O)
        assert service.provider == LLMProvider.OPENAI
        assert service.model == LLMModel.GPT_4O

    @patch('src.services.llm_service.is_production', return_value=True)
    @patch('src.services.llm_service.init_chat_model')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    def test_production_mode_init_chat_model_failure(self, mock_init, mock_is_production) -> None:
        """測試生產模式下 init_chat_model 失敗"""
        mock_init.side_effect = Exception("API 連接失敗")
        
        with pytest.raises(RuntimeError) as exc_info:
            LLMService(LLMProvider.OPENAI, LLMModel.GPT_4O)
        
        error_message = str(exc_info.value)
        assert "LLM 客戶端初始化失敗" in error_message
        assert "API 連接失敗" in error_message
        assert "提供者: openai" in error_message


class TestDevelopmentModeAPIKeyValidation:
    """測試開發模式下的 API 密鑰驗證"""

    @patch('src.services.llm_service.is_production', return_value=False)
    @patch.dict('os.environ', {}, clear=True)
    def test_development_mode_missing_api_key_uses_mock(self, mock_is_production) -> None:
        """測試開發模式下缺少 API 密鑰使用 mock 客戶端"""
        service = LLMService(LLMProvider.OPENAI, LLMModel.GPT_4O)
        
        assert service.provider == LLMProvider.OPENAI
        assert service.model == LLMModel.GPT_4O
        # 檢查是否使用了 mock 客戶端
        assert hasattr(service._client, 'invoke')

    @patch('src.services.llm_service.is_production', return_value=False)
    @patch.dict('os.environ', {}, clear=True)
    def test_development_mode_mock_client_invoke(self, mock_is_production) -> None:
        """測試開發模式下 mock 客戶端的調用"""
        service = LLMService(LLMProvider.OPENAI, LLMModel.GPT_4O)
        
        messages = [{"role": "user", "content": "測試問題"}]
        result = service.invoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert "開發模式的 mock 回應" in result.content
        assert result.provider == "openai"
        assert result.model_name == "gpt-4o"

    @patch('src.services.llm_service.is_production', return_value=False)
    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.asyncio
    async def test_development_mode_mock_client_ainvoke(self, mock_is_production) -> None:
        """測試開發模式下 mock 客戶端的異步調用"""
        service = LLMService(LLMProvider.OPENAI, LLMModel.GPT_4O)
        
        messages = [{"role": "user", "content": "異步測試問題"}]
        result = await service.ainvoke(messages)
        
        assert isinstance(result, LLMResponse)
        assert "開發模式的 mock 回應" in result.content

    @patch('src.services.llm_service.is_production', return_value=False)
    @patch('src.services.llm_service.init_chat_model')
    @patch.dict('os.environ', {}, clear=True)
    def test_development_mode_init_chat_model_failure_uses_mock(self, mock_init, mock_is_production) -> None:
        """測試開發模式下 init_chat_model 失敗使用 mock 客戶端"""
        mock_init.side_effect = Exception("模擬初始化失敗")
        
        # 不應該拋出異常，而是使用 mock 客戶端
        service = LLMService(LLMProvider.OPENAI, LLMModel.GPT_4O)
        assert service.provider == LLMProvider.OPENAI


class TestAPIKeyValidationMethods:
    """測試 API 密鑰驗證相關方法"""

    def test_check_api_key_openai_with_key(self) -> None:
        """測試 OpenAI API 密鑰檢查 - 有密鑰"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            service = LLMService.__new__(LLMService)
            service.provider = LLMProvider.OPENAI
            assert service._check_api_key() is True

    def test_check_api_key_openai_without_key(self) -> None:
        """測試 OpenAI API 密鑰檢查 - 無密鑰"""
        with patch.dict('os.environ', {}, clear=True):
            service = LLMService.__new__(LLMService)
            service.provider = LLMProvider.OPENAI
            assert service._check_api_key() is False

    def test_check_api_key_anthropic_with_key(self) -> None:
        """測試 Anthropic API 密鑰檢查 - 有密鑰"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            service = LLMService.__new__(LLMService)
            service.provider = LLMProvider.ANTHROPIC
            assert service._check_api_key() is True

    def test_check_api_key_google_with_key(self) -> None:
        """測試 Google API 密鑰檢查 - 有密鑰"""
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'}):
            service = LLMService.__new__(LLMService)
            service.provider = LLMProvider.GOOGLE
            assert service._check_api_key() is True

    def test_check_api_key_ollama_always_true(self) -> None:
        """測試 Ollama API 密鑰檢查 - 總是返回 True"""
        with patch.dict('os.environ', {}, clear=True):
            service = LLMService.__new__(LLMService)
            service.provider = LLMProvider.OLLAMA
            assert service._check_api_key() is True

    def test_get_required_api_key_name_openai(self) -> None:
        """測試獲取所需 API 密鑰名稱 - OpenAI"""
        service = LLMService.__new__(LLMService)
        service.provider = LLMProvider.OPENAI
        assert service._get_required_api_key_name() == "OPENAI_API_KEY"

    def test_get_required_api_key_name_anthropic(self) -> None:
        """測試獲取所需 API 密鑰名稱 - Anthropic"""
        service = LLMService.__new__(LLMService)
        service.provider = LLMProvider.ANTHROPIC
        assert service._get_required_api_key_name() == "ANTHROPIC_API_KEY"

    def test_get_required_api_key_name_google(self) -> None:
        """測試獲取所需 API 密鑰名稱 - Google"""
        service = LLMService.__new__(LLMService)
        service.provider = LLMProvider.GOOGLE
        assert service._get_required_api_key_name() == "GOOGLE_API_KEY"

    def test_get_required_api_key_name_ollama(self) -> None:
        """測試獲取所需 API 密鑰名稱 - Ollama"""
        service = LLMService.__new__(LLMService)
        service.provider = LLMProvider.OLLAMA
        assert service._get_required_api_key_name() == "OLLAMA_HOST (可選)"


class TestMockClientBehavior:
    """測試 Mock 客戶端行為"""

    def test_mock_client_semantic_evaluation_response(self) -> None:
        """測試 Mock 客戶端語義評估回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試語義評估請求
        messages = [HumanMessage(content="請評估語義一致性")]
        response = mock_client.invoke(messages)
        assert response.content == "8.5"

    def test_mock_client_translation_response(self) -> None:
        """測試 Mock 客戶端翻譯回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試翻譯請求
        messages = [HumanMessage(content="請翻譯這個程式問題")]
        response = mock_client.invoke(messages)
        assert "翻譯" in response.content

    def test_mock_client_execution_response(self) -> None:
        """測試 Mock 客戶端執行回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試執行請求
        messages = [HumanMessage(content="請執行這個程式問題")]
        response = mock_client.invoke(messages)
        assert "執行" in response.content or "回答" in response.content

    def test_mock_client_default_response(self) -> None:
        """測試 Mock 客戶端預設回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試一般請求
        messages = [HumanMessage(content="一般問題")]
        response = mock_client.invoke(messages)
        assert "開發模式的 mock 回應" in response.content

    @pytest.mark.asyncio
    async def test_mock_client_async_semantic_evaluation_response(self) -> None:
        """測試 Mock 客戶端異步語義評估回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試異步語義評估請求
        messages = [HumanMessage(content="請評估語義一致性")]
        response = await mock_client.ainvoke(messages)
        assert response.content == "8.5"

    @pytest.mark.asyncio
    async def test_mock_client_async_translation_response(self) -> None:
        """測試 Mock 客戶端異步翻譯回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試異步翻譯請求
        messages = [HumanMessage(content="請翻譯這個程式問題")]
        response = await mock_client.ainvoke(messages)
        assert "翻譯" in response.content

    @pytest.mark.asyncio
    async def test_mock_client_async_execution_response(self) -> None:
        """測試 Mock 客戶端異步執行回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試異步執行請求
        messages = [HumanMessage(content="請執行這個程式問題")]
        response = await mock_client.ainvoke(messages)
        assert "執行" in response.content or "回答" in response.content

    @pytest.mark.asyncio
    async def test_mock_client_async_default_response(self) -> None:
        """測試 Mock 客戶端異步預設回應"""
        from src.services.llm_service import LLMService
        
        mock_client = LLMService._create_mock_client()
        
        # 測試異步一般請求
        messages = [HumanMessage(content="一般問題")]
        response = await mock_client.ainvoke(messages)
        assert "開發模式的 mock 回應" in response.content
