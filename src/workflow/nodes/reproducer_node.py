"""
再現者 Agent 節點
Reproducer Agent Node

負責 QA 推理與語義比較（純文字處理）：
1. 對翻譯前（原始英文）的問題進行 QA 推理
2. 對翻譯後（繁體中文）的問題進行 QA 推理  
3. 生成兩個版本的文字回答結果
4. 提供推理過程的詳細記錄
5. 確保在相同邏輯下進行公平比較

注意: 此處的 QA 執行僅指 LLM 文字推理，不涉及程式碼實際執行
"""

import logging
import time
from typing import List

from ..state import WorkflowState, update_state_safely
from ...config.llm_config import get_agent_config
from ...constants.llm import LLMProvider, LLMModel
from ...models.dataset import QAExecutionResult, Language, ProcessingStatus
from ...models.quality import ErrorRecord, ErrorType
from ...services.llm_service import LLMFactory

logger = logging.getLogger(__name__)


class ReproducerAgent:
    """再現者 Agent 類別"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ReproducerAgent")
        
        # 從配置取得模型設定
        agent_config = get_agent_config("reproducer")
        if not agent_config:
            raise ValueError("Failed to get reproducer configuration")
        
        self.primary_model = agent_config["primary_model"]
        self.fallback_model = agent_config.get("fallback_model")
        self.temperature = agent_config["temperature"]  # 0.0 for consistency
        self.max_tokens = agent_config["max_tokens"]
        
        # 初始化 LLM 服務
        self.llm_service = self._initialize_llm_service()
    
    def _initialize_llm_service(self):
        """初始化 LLM 服務"""
        try:
            # 根據模型名稱選擇提供者
            if "gpt" in self.primary_model:
                provider = LLMProvider.OPENAI
            elif "claude" in self.primary_model:
                provider = LLMProvider.ANTHROPIC
            elif "gemini" in self.primary_model:
                provider = LLMProvider.GOOGLE
            else:
                provider = LLMProvider.ANTHROPIC  # 預設使用 Claude
                
            model_enum = LLMModel(self.primary_model)
            return LLMFactory.create_llm(provider, model_enum)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    def execute_qa_reasoning(self, question: str, language: Language, context: str = "") -> QAExecutionResult:
        """
        執行 QA 推理
        
        Args:
            question: 問題內容
            language: 問題語言
            context: 額外上下文（可選）
            
        Returns:
            QA 執行結果
        """
        start_time = time.time()
        
        # 構建 QA 推理提示
        if language == Language.ENGLISH:
            qa_prompt = f"""
            Please provide a detailed answer to the following programming question. 
            Think step by step and provide clear reasoning.

            Question: {question}

            {context}

            Please structure your response with:
            1. Understanding of the question
            2. Step-by-step reasoning
            3. Final answer with code examples if applicable
            4. Key concepts explained

            Answer:
            """
        else:  # Traditional Chinese
            qa_prompt = f"""
            請對以下程式設計問題提供詳細回答。
            請按步驟思考並提供清晰的推理過程。

            問題：{question}

            {context}

            請按以下結構回答：
            1. 對問題的理解
            2. 逐步推理過程
            3. 最終答案（如適用請包含程式碼範例）
            4. 關鍵概念說明

            回答：
            """
        
        try:
            messages = [{"role": "user", "content": qa_prompt}]
            response = self.llm_service.invoke(messages)
            
            execution_time = time.time() - start_time
            
            # 提取推理步驟（簡化版本）
            reasoning_steps = self._extract_reasoning_steps(response.content)
            
            return QAExecutionResult(
                record_id="",  # 將在調用時設定
                language=language,
                input_question=question,
                generated_answer=response.content.strip(),
                execution_time=execution_time,
                reasoning_steps=reasoning_steps,
                confidence_score=0.8  # 預設信心分數
            )
            
        except Exception as e:
            self.logger.error(f"QA reasoning failed for {language.value}: {e}")
            raise
    
    def _extract_reasoning_steps(self, answer: str) -> List[str]:
        """
        從回答中提取推理步驟
        
        Args:
            answer: LLM 生成的回答
            
        Returns:
            推理步驟列表
        """
        try:
            # 簡化的步驟提取邏輯
            steps = []
            lines = answer.split('\n')
            
            for line in lines:
                line = line.strip()
                # 尋找編號步驟
                if (line.startswith(('1.', '2.', '3.', '4.', '5.')) or
                    line.startswith(('一、', '二、', '三、', '四、', '五、')) or
                    line.startswith(('步驟', 'Step', '首先', '然後', '接下來', '最後'))):
                    steps.append(line)
            
            # 如果沒有找到明確步驟，將回答分段
            if not steps:
                paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
                steps = paragraphs[:5]  # 最多取5段
            
            return steps
            
        except Exception as e:
            self.logger.warning(f"Failed to extract reasoning steps: {e}")
            return ["Complete reasoning provided in answer"]
    
    def execute_comparative_qa(self, original_question: str, translated_question: str, record_id: str) -> tuple[QAExecutionResult, QAExecutionResult]:
        """
        執行比較性 QA 推理
        
        Args:
            original_question: 原始英文問題
            translated_question: 翻譯後中文問題
            record_id: 記錄ID
            
        Returns:
            (原始 QA 結果, 翻譯 QA 結果)
        """
        self.logger.info(f"Executing comparative QA for record {record_id}")
        
        try:
            # 執行原始英文問題的 QA 推理
            self.logger.debug("Executing QA for original English question")
            original_qa_result = self.execute_qa_reasoning(
                original_question,
                Language.ENGLISH,
                "Focus on providing comprehensive programming solution with clear explanations."
            )
            original_qa_result.record_id = record_id
            
            # 執行翻譯中文問題的 QA 推理
            self.logger.debug("Executing QA for translated Chinese question")
            translated_qa_result = self.execute_qa_reasoning(
                translated_question,
                Language.TRADITIONAL_CHINESE,
                "請專注於提供全面的程式設計解決方案並給出清楚的說明。"
            )
            translated_qa_result.record_id = record_id
            
            self.logger.info(f"Comparative QA completed for record {record_id}")
            return original_qa_result, translated_qa_result
            
        except Exception as e:
            self.logger.error(f"Comparative QA failed for record {record_id}: {e}")
            raise


def reproducer_node(state: WorkflowState) -> WorkflowState:
    """
    再現者節點 - LangGraph 節點函數
    
    負責執行 QA 推理比較
    
    Args:
        state: 當前工作流狀態
        
    Returns:
        更新後的工作流狀態
    """
    logger = logging.getLogger(__name__ + ".reproducer_node")
    
    try:
        # 檢查前置條件
        current_record = state.get("current_record")
        translation_result = state.get("translation_result")
        
        if not current_record:
            raise ValueError("No current record to process")
        
        if not translation_result:
            raise ValueError("No translation result available")
        
        # 檢查是否已有 QA 結果且不需要重新執行
        if (state.get("original_qa_result") and 
            state.get("translated_qa_result") and
            state["processing_status"] != ProcessingStatus.RETRY_NEEDED):
            logger.info(f"QA results already exist for record {current_record.id}, skipping")
            return state
        
        # 創建再現者 Agent
        agent = ReproducerAgent()
        
        # 執行比較性 QA 推理
        logger.info(f"Starting comparative QA for record {current_record.id}")
        
        original_qa_result, translated_qa_result = agent.execute_comparative_qa(
            current_record.question,
            translation_result.translated_question,
            current_record.id
        )
        
        # 更新狀態
        updates = {
            "original_qa_result": original_qa_result,
            "translated_qa_result": translated_qa_result
        }
        
        # 如果當前狀態是 RETRY_NEEDED，則更新為 PROCESSING
        if state["processing_status"] == ProcessingStatus.RETRY_NEEDED:
            updates["processing_status"] = ProcessingStatus.PROCESSING
        
        return update_state_safely(state, updates)
        
    except Exception as e:
        logger.error(f"Reproducer node failed: {e}")
        
        # 記錄錯誤
        error_record = ErrorRecord(
            error_type=ErrorType.OTHER,
            error_message=str(e),
            timestamp=time.time(),
            retry_attempt=state.get("retry_count", 0),
            agent_name="reproducer",
            recovery_action="logged_error"
        )
        
        # 更新狀態為失敗
        try:
            return update_state_safely(state, {
                "processing_status": ProcessingStatus.FAILED,
                "error_history": state["error_history"] + [error_record]
            })
        except Exception:
            # 如果狀態更新失敗，至少更新錯誤歷史
            state["error_history"].append(error_record)
            state["processing_status"] = ProcessingStatus.FAILED
            return state


__all__ = [
    "ReproducerAgent",
    "reproducer_node",
]
