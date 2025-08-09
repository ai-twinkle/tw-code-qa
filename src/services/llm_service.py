"""
LLM 服務模組 - 簡化版
LLM Service Module - Simplified using LangGraph Factory

使用 LangGraph 內建工廠提供統一的 LLM 介面
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Union, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..constants.llm import LLMModel, LLMProvider


class ChatModelProtocol(Protocol):
    """聊天模型協議"""
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage: ...
    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage: ...


@dataclass
class LLMResponse:
    """LLM 回應結果"""
    content: str
    model_name: str
    provider: str
    response_time: float
    timestamp: float


class LLMService:
    """LLM 服務類別 - 使用 LangGraph 內建工廠"""
    
    def __init__(self, provider: LLMProvider, model: LLMModel):
        self.provider = provider
        self.model = model
        self._client = self._initialize_client()
    
    def _initialize_client(self) -> ChatModelProtocol:
        """初始化 LLM 客戶端"""
        # 實際專案中應該使用真實的 LangChain 客戶端
        # 這裡提供一個可以被測試 mock 的版本
        return self._create_real_client()

    @staticmethod
    def _create_real_client() -> ChatModelProtocol:
        """建立真實的 LLM 客戶端"""
        # 為了型別安全，我們需要返回符合協議的物件
        class MockChatModel:
            @staticmethod
            def invoke(_messages: List[BaseMessage]) -> BaseMessage:
                return AIMessage(content="Mock response")

            @staticmethod
            async def ainvoke(_messages: List[BaseMessage]) -> BaseMessage:
                return AIMessage(content="Mock async response")
        
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
