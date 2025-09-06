"""
LLM 服務模組 - 使用 LangChain init_chat_model
LLM Service Module - Using LangChain init_chat_model

使用 LangChain 最新的 init_chat_model 方法提供統一的 LLM 介面
"""

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Union

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..constants.llm import LLMModel, LLMProvider


@dataclass
class LLMResponse:
    """LLM 回應結果"""
    content: str
    model_name: str
    provider: str
    response_time: float
    timestamp: float


class LLMService:
    """LLM 服務類別 - 使用 LangChain init_chat_model"""
    
    def __init__(self, provider: LLMProvider, model: LLMModel):
        self.provider = provider
        self.model = model
        self._client = self._initialize_client()
    
    def _initialize_client(self):
        """初始化 LLM 客戶端 - 使用 init_chat_model"""
        try:
            # 檢查 API 密鑰是否設置
            if not self._check_api_key():
                print(f"Warning: No API key found for {self.provider.value}. Using mock client for development.")
                return self._create_mock_client()
            
            # 構建模型標識符，根據 provider 決定格式
            model_identifier = self._get_model_identifier()
            
            # 使用 init_chat_model 初始化客戶端
            client = init_chat_model(
                model=model_identifier,
                temperature=0.7,  # 預設溫度
                max_tokens=None,  # 預設不限制
                timeout=30,       # 30 秒超時
                max_retries=3     # 最多重試 3 次
            )
            
            return client
            
        except Exception as e:
            print(f"Warning: Failed to initialize real LLM client ({e}). Using mock client for development.")
            return self._create_mock_client()
    
    def _get_model_identifier(self) -> str:
        """根據提供者和模型構建模型標識符"""
        if self.provider == LLMProvider.OPENAI:
            return f"{self.model.value}"  # OpenAI 直接使用模型名稱
        elif self.provider == LLMProvider.ANTHROPIC:
            return f"anthropic:{self.model.value}"
        elif self.provider == LLMProvider.GOOGLE:
            return f"google_genai:{self.model.value}"
        elif self.provider == LLMProvider.OLLAMA:
            return f"ollama:{self.model.value}"
        else:
            # 預設直接使用模型名稱
            return self.model.value
    
    def _check_api_key(self) -> bool:
        """檢查對應提供者的 API 密鑰是否設置"""
        if self.provider == LLMProvider.OPENAI:
            return bool(os.getenv("OPENAI_API_KEY"))
        elif self.provider == LLMProvider.ANTHROPIC:
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == LLMProvider.GOOGLE:
            return bool(os.getenv("GOOGLE_API_KEY"))
        elif self.provider == LLMProvider.OLLAMA:
            # Ollama 通常不需要 API key，但需要確認服務可用
            return True
        return False

    @staticmethod
    def _create_mock_client():
        """建立 Mock 客戶端 - 用於開發和測試"""
        class MockChatModel:
            @staticmethod
            def invoke(messages: List[BaseMessage]) -> BaseMessage:
                # 提供更真實的 mock 回應，根據輸入內容返回不同的回應
                content = messages[-1].content if messages else ""
                
                # 如果是語義評估請求
                if "語義一致性" in content or "semantic" in content.lower() or "評估標準" in content or "evaluation" in content.lower():
                    return AIMessage(content="8.5")
                
                # 如果是翻譯請求
                elif "翻譯" in content or "translate" in content.lower():
                    return AIMessage(content="這是一個翻譯後的程式問題示例。")
                
                # 如果是QA執行請求  
                elif "執行" in content or "execute" in content.lower() or "回答" in content:
                    return AIMessage(content="根據題目要求，我需要分析這個程式問題並提供解答。這是一個示例回答。")
                
                return AIMessage(content="這是一個開發模式的 mock 回應，請配置真實的 LLM API 以獲得實際結果。")

            @staticmethod
            async def ainvoke(messages: List[BaseMessage]) -> BaseMessage:
                # 非同步版本，返回相同的邏輯
                content = messages[-1].content if messages else ""
                
                # 如果是語義評估請求
                if "語義一致性" in content or "semantic" in content.lower() or "評估標準" in content or "evaluation" in content.lower():
                    return AIMessage(content="8.5")
                
                # 如果是翻譯請求
                elif "翻譯" in content or "translate" in content.lower():
                    return AIMessage(content="這是一個翻譯後的程式問題示例。")
                elif "執行" in content or "execute" in content.lower() or "回答" in content:
                    return AIMessage(content="根據題目要求，我需要分析這個程式問題並提供解答。這是一個示例回答。")
                
                return AIMessage(content="這是一個開發模式的 mock 回應，請配置真實的 LLM API 以獲得實際結果。")
        
        return MockChatModel()

    @staticmethod
    def _convert_messages(messages: Union[List[Dict[str, str]], List[BaseMessage]]) -> List[BaseMessage]:
        """轉換訊息格式為 LangChain 格式，支持兩種輸入格式"""
        langchain_messages: List[BaseMessage] = []
        
        for msg in messages:
            # 如果已經是 BaseMessage，直接添加
            if isinstance(msg, BaseMessage):
                langchain_messages.append(msg)
            # 如果是字典格式，轉換為 BaseMessage
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:
                    # 預設當作使用者訊息
                    langchain_messages.append(HumanMessage(content=content))
        
        return langchain_messages
    
    def invoke(self, messages: Union[List[Dict[str, str]], List[BaseMessage]]) -> LLMResponse:
        """同步調用 LLM"""
        start_time = time.time()
        
        langchain_messages = self._convert_messages(messages)
        response = self._client.invoke(langchain_messages)
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=str(response.content),
            model_name=self.model.value,
            provider=self.provider.value,
            response_time=response_time,
            timestamp=time.time()
        )
    
    async def ainvoke(self, messages: Union[List[Dict[str, str]], List[BaseMessage]]) -> LLMResponse:
        """非同步調用 LLM"""
        start_time = time.time()
        
        langchain_messages = self._convert_messages(messages)
        response = await self._client.ainvoke(langchain_messages)
        
        response_time = time.time() - start_time
        
        return LLMResponse(
            content=str(response.content),
            model_name=self.model.value,
            provider=self.provider.value,
            response_time=response_time,
            timestamp=time.time()
        )


class LLMFactory:
    """LLM 工廠類別 - 簡化版本使用 LangGraph 工廠"""
    
    @staticmethod
    def create_llm(provider: LLMProvider, model: LLMModel) -> LLMService:
        """建立 LLM 服務實例"""
        return LLMService(provider, model)


__all__ = [
    "LLMResponse", 
    "LLMService",
    "LLMFactory",
]
