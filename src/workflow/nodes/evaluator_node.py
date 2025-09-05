"""
評估者 Agent 節點
Evaluator Agent Node

負責語義一致性檢查與翻譯指導：
1. 檢查翻譯後 LLM 復現是否保持與翻譯前近似的語義輸出
2. 比較翻譯前後 QA 執行結果的語義一致性
3. 評估兩個回答的語義偏差程度
4. 分析翻譯過程中可能的語義遺失或曲解
5. 標記語義分數過低的問題記錄
6. 翻譯改進指導：當發現語義偏差時，直接向 AnalyzerDesigner Agent 提供具體的翻譯改進建議
7. 管理最大重試次數限制，避免無限循環
"""

import logging
import time
from typing import List, Dict

from ..state import WorkflowState, update_state_safely
from ...config.llm_config import get_agent_config, QUALITY_THRESHOLDS
from ...constants.llm import LLMProvider, LLMModel
from ...models.dataset import QAExecutionResult, ProcessingStatus
from ...models.quality import QualityAssessment, ErrorRecord, ErrorType
from ...services.llm_service import LLMFactory

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """評估者 Agent 類別"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".EvaluatorAgent")
        
        # 從配置取得模型設定
        agent_config = get_agent_config("evaluator")
        if not agent_config:
            raise ValueError("Failed to get evaluator configuration")
        
        self.primary_model = agent_config["primary_model"]
        self.fallback_model = agent_config.get("fallback_model")
        self.temperature = agent_config["temperature"]
        self.max_tokens = agent_config["max_tokens"]
        
        # 品質閾值
        self.min_semantic_score = QUALITY_THRESHOLDS["minimum_semantic_score"]
        self.max_retry_attempts = QUALITY_THRESHOLDS["maximum_retry_attempts"]
        
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
                provider = LLMProvider.OPENAI  # 預設
                
            model_enum = LLMModel(self.primary_model)
            return LLMFactory.create_llm(provider, model_enum)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    def evaluate_semantic_consistency(self, original_qa: QAExecutionResult, translated_qa: QAExecutionResult) -> float:
        """
        評估語義一致性
        
        Args:
            original_qa: 原始英文 QA 結果
            translated_qa: 翻譯中文 QA 結果
            
        Returns:
            語義一致性分數 (0-10)
        """
        evaluation_prompt = f"""
        請評估以下兩個程式設計問答對的語義一致性：

        **原始英文問答：**
        問題：{original_qa.input_question}
        回答：{original_qa.generated_answer}

        **翻譯中文問答：**
        問題：{translated_qa.input_question}
        回答：{translated_qa.generated_answer}

        請從以下維度評估語義一致性（0-10分）：
        1. 問題核心意圖是否相同
        2. 回答的技術內容是否一致
        3. 解決方案的邏輯是否相同
        4. 程式碼片段是否對應
        5. 關鍵技術概念是否保持

        請僅回答一個 0-10 的數字分數，代表整體語義一致性。
        """
        
        try:
            messages = [{"role": "user", "content": evaluation_prompt}]
            response = self.llm_service.invoke(messages)
            
            # 提取分數
            score_text = response.content.strip()
            
            # 嘗試提取數字
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(10.0, score))  # 確保在 0-10 範圍內
            else:
                self.logger.warning(f"Could not extract score from: {score_text}")
                return 5.0  # 預設中等分數
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate semantic consistency: {e}")
            return 5.0  # 預設分數
    
    def evaluate_code_integrity(self, original_answer: str, translated_answer: str) -> float:
        """
        評估程式碼完整性
        
        Args:
            original_answer: 原始英文回答
            translated_answer: 翻譯中文回答
            
        Returns:
            程式碼完整性分數 (0-10)
        """
        try:
            # 簡化的程式碼完整性檢查
            original_code_blocks = original_answer.count('```')
            translated_code_blocks = translated_answer.count('```')
            
            # 檢查程式碼區塊數量是否一致
            if original_code_blocks != translated_code_blocks:
                return 5.0  # 程式碼區塊數量不一致
            
            # 檢查是否包含程式碼
            if original_code_blocks == 0:
                return 10.0  # 沒有程式碼，視為完整
            
            # 檢查程式碼語法（簡化版本）
            if '```' in translated_answer:
                return 8.0  # 包含程式碼區塊，視為良好
            else:
                return 6.0  # 程式碼格式可能有問題
                
        except Exception as e:
            self.logger.warning(f"Failed to evaluate code integrity: {e}")
            return 7.0  # 預設分數
    
    def evaluate_translation_naturalness(self, translated_question: str, translated_answer: str) -> float:
        """
        評估翻譯自然度
        
        Args:
            translated_question: 翻譯後問題
            translated_answer: 翻譯後回答
            
        Returns:
            翻譯自然度分數 (0-10)
        """
        evaluation_prompt = f"""
        請評估以下繁體中文程式設計內容的自然度和流暢度：

        **問題：**
        {translated_question}

        **回答：**
        {translated_answer}

        評估標準：
        1. 語句是否符合繁體中文表達習慣
        2. 技術術語使用是否恰當
        3. 語法是否正確流暢
        4. 是否有明顯的翻譯痕跡
        5. 是否易於理解

        請僅回答一個 0-10 的數字分數。
        """
        
        try:
            messages = [{"role": "user", "content": evaluation_prompt}]
            response = self.llm_service.invoke(messages)
            
            # 提取分數
            score_text = response.content.strip()
            
            import re
            score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return max(0.0, min(10.0, score))
            else:
                return 7.0  # 預設分數
                
        except Exception as e:
            self.logger.warning(f"Failed to evaluate translation naturalness: {e}")
            return 7.0  # 預設分數

    @staticmethod
    def generate_improvement_suggestions(
                                       original_qa: QAExecutionResult,
                                       translated_qa: QAExecutionResult,
                                       quality_scores: Dict[str, float]) -> List[str]:
        """
        生成翻譯改進建議
        
        Args:
            original_qa: 原始 QA 結果
            translated_qa: 翻譯 QA 結果
            quality_scores: 品質分數字典
            
        Returns:
            改進建議列表
        """
        suggestions = []
        
        # 基於分數提供建議
        if quality_scores.get("semantic_consistency", 10) < 7.0:
            suggestions.append("需要改善語義一致性：確保翻譯保持原文的核心技術意圖")
        
        if quality_scores.get("code_integrity", 10) < 7.0:
            suggestions.append("需要保護程式碼完整性：確保程式碼區塊不被誤譯或遺失")
        
        if quality_scores.get("translation_naturalness", 10) < 7.0:
            suggestions.append("需要提升翻譯自然度：使用更符合繁體中文習慣的表達方式")
        
        # 通用改進建議
        suggestions.extend([
            "使用序列式翻譯：先完成問題翻譯，再基於翻譯後問題翻譯答案",
            "保持技術術語一致性：使用業界標準的繁體中文技術術語",
            "保護程式碼片段：確保 ``` 內的程式碼不被翻譯"
        ])
        
        return suggestions[:5]  # 限制建議數量
    
    def perform_quality_assessment(self, 
                                 original_qa: QAExecutionResult,
                                 translated_qa: QAExecutionResult,
                                 translated_question: str,
                                 translated_answer: str,
                                 record_id: str) -> QualityAssessment:
        """
        執行品質評估
        
        Args:
            original_qa: 原始 QA 結果
            translated_qa: 翻譯 QA 結果
            translated_question: 翻譯後問題
            translated_answer: 翻譯後回答
            record_id: 記錄ID
            
        Returns:
            品質評估結果
        """
        self.logger.info(f"Performing quality assessment for record {record_id}")
        
        try:
            # 評估各個維度
            semantic_score = self.evaluate_semantic_consistency(original_qa, translated_qa)
            code_score = self.evaluate_code_integrity(original_qa.generated_answer, translated_qa.generated_answer)
            naturalness_score = self.evaluate_translation_naturalness(translated_question, translated_answer)
            
            # 計算綜合品質分數
            overall_score = (semantic_score * 0.5 + code_score * 0.3 + naturalness_score * 0.2)
            
            quality_scores = {
                "semantic_consistency": semantic_score,
                "code_integrity": code_score,
                "translation_naturalness": naturalness_score
            }
            
            # 生成改進建議
            improvement_suggestions = self.generate_improvement_suggestions(
                original_qa, translated_qa, quality_scores
            )
            
            # 創建品質評估結果
            quality_assessment = QualityAssessment(
                record_id=record_id,
                semantic_consistency_score=semantic_score,
                code_integrity_score=code_score,
                translation_naturalness_score=naturalness_score,
                overall_quality_score=overall_score,
                semantic_analysis=f"語義一致性分析：{semantic_score}/10",
                code_analysis=f"程式碼完整性分析：{code_score}/10",
                naturalness_analysis=f"翻譯自然度分析：{naturalness_score}/10",
                improvement_suggestions=improvement_suggestions,
                evaluator_model=self.primary_model
            )
            
            self.logger.info(f"Quality assessment completed for record {record_id}, overall score: {overall_score:.2f}")
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed for record {record_id}: {e}")
            raise


def evaluator_node(state: WorkflowState) -> WorkflowState:
    """
    評估者節點 - LangGraph 節點函數
    
    負責品質評估和翻譯指導
    
    Args:
        state: 當前工作流狀態
        
    Returns:
        更新後的工作流狀態
    """
    logger = logging.getLogger(__name__ + ".evaluator_node")
    
    try:
        # 檢查前置條件
        current_record = state.get("current_record")
        translation_result = state.get("translation_result")
        original_qa_result = state.get("original_qa_result")
        translated_qa_result = state.get("translated_qa_result")
        
        if not all([current_record, translation_result, original_qa_result, translated_qa_result]):
            raise ValueError("Missing required data for evaluation")
        
        # 創建評估者 Agent
        agent = EvaluatorAgent()
        
        # 執行品質評估
        logger.info(f"Starting quality assessment for record {current_record.id}")
        
        quality_assessment = agent.perform_quality_assessment(
            original_qa_result,
            translated_qa_result,
            translation_result.translated_question,
            translation_result.translated_answer,
            current_record.id
        )
        
        # 判斷是否需要重試
        current_retry_count = state.get("retry_count", 0)
        needs_retry = (quality_assessment.overall_quality_score < agent.min_semantic_score and
                      current_retry_count < agent.max_retry_attempts)
        
        if needs_retry:
            # 需要重試
            logger.info(f"Quality score {quality_assessment.overall_quality_score:.2f} below threshold, triggering retry")
            
            updates = {
                "quality_assessment": quality_assessment,
                "processing_status": ProcessingStatus.RETRY_NEEDED,
                "retry_count": current_retry_count + 1,
                "improvement_suggestions": quality_assessment.improvement_suggestions
            }
        else:
            # 不需要重試或已達重試上限
            if current_retry_count >= agent.max_retry_attempts:
                logger.warning(f"Max retries reached for record {current_record.id}")
            
            final_status = (ProcessingStatus.COMPLETED if quality_assessment.overall_quality_score >= agent.min_semantic_score
                          else ProcessingStatus.FAILED)
            
            updates = {
                "quality_assessment": quality_assessment,
                "processing_status": final_status
            }
        
        return update_state_safely(state, updates)
        
    except Exception as e:
        logger.error(f"Evaluator node failed: {e}")
        
        # 記錄錯誤
        error_record = ErrorRecord(
            error_type=ErrorType.OTHER,
            error_message=str(e),
            timestamp=time.time(),
            retry_attempt=state.get("retry_count", 0),
            agent_name="evaluator",
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
    "EvaluatorAgent",
    "evaluator_node",
]
