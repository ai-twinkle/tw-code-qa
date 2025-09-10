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

from ..state import WorkflowState, update_state_safely, StateUpdateValue
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
                provider = LLMProvider.OLLAMA  # 否則使用 Ollama
                
            model_enum = LLMModel(self.primary_model)
            
            logger.info(f"Initializing EvaluatorAgent with {provider.value} - {model_enum.value}")
            
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

        請從以下維度詳細評估語義一致性（0-10分）：

        1. **問題核心意圖一致性**：翻譯後的問題是否保持了原問題的核心技術需求
        2. **技術內容準確性**：回答中的技術概念、方法、原理是否保持一致
        3. **解決方案邏輯**：解決問題的步驟、方法、思路是否相同
        4. **程式碼對應性**：程式碼片段的功能、邏輯、結構是否一致
        5. **關鍵概念保持**：重要的技術術語、概念是否正確傳達
        6. **上下文理解**：翻譯是否保持了原文的技術背景和適用場景
        7. **深度層次**：技術解釋的詳細程度和深度是否一致

        評分標準：
        - 9-10分：語義完全一致，翻譯準確無誤
        - 7-8分：語義基本一致，有輕微偏差但不影響理解
        - 5-6分：語義大致相同，有一些偏差或遺漏
        - 3-4分：語義有明顯差異，可能影響技術理解
        - 1-2分：語義差異很大，翻譯存在嚴重問題
        - 0分：語義完全不一致或翻譯錯誤

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
        評估程式碼完整性 - 使用 LLM 進行智能分析
        
        Args:
            original_answer: 原始英文回答
            translated_answer: 翻譯中文回答
            
        Returns:
            程式碼完整性分數 (0-10)
        """
        evaluation_prompt = f"""
        請評估以下翻譯對的程式碼完整性和正確性：

        **原始英文回答：**
        {original_answer}

        **翻譯中文回答：**
        {translated_answer}

        評估標準：
        1. 程式碼區塊數量是否一致
        2. 程式碼內容是否完整保留
        3. 程式碼語法是否正確
        4. 程式碼邏輯是否一致
        5. 程式碼註解翻譯是否恰當
        6. 變數名稱是否適當保留或翻譯
        7. 程式碼格式是否正確維持

        重點關注：
        - 程式碼片段不應被意外翻譯或修改
        - 關鍵字（如 class, def, import 等）應保持英文
        - 程式碼結構和縮排應保持一致
        - 註解可以翻譯但不應影響程式碼執行

        請僅回答一個 0-10 的數字分數，代表程式碼完整性。
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
                self.logger.warning(f"Could not extract code integrity score from: {score_text}")
                return 7.0  # 預設分數
                
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

        請從以下維度詳細評估翻譯自然度（0-10分）：

        1. **語言流暢度**：句子結構是否符合繁體中文的表達習慣
        2. **技術術語恰當性**：專業術語的翻譯是否準確且符合業界慣例
        3. **語法正確性**：語法、用詞、標點符號是否正確
        4. **自然度**：是否有明顯的機器翻譯或英文思維痕跡
        5. **可讀性**：對於中文讀者來說是否易於理解和閱讀
        6. **專業性**：是否維持了原文的專業水準和技術深度
        7. **文化適應性**：是否適合繁體中文的技術文檔風格

        評分標準：
        - 9-10分：完全自然，如同原生中文技術文檔
        - 7-8分：基本自然，偶有輕微翻譯痕跡
        - 5-6分：可以理解，但有明顯翻譯感
        - 3-4分：較難閱讀，翻譯痕跡明顯
        - 1-2分：不自然，嚴重影響閱讀體驗
        - 0分：完全不自然或無法理解

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

    def generate_improvement_suggestions(self,
                                       original_qa: QAExecutionResult,
                                       translated_qa: QAExecutionResult,
                                       quality_scores: Dict[str, float]) -> List[str]:
        """
        生成翻譯改進建議 - 使用 LLM 進行智能分析
        
        Args:
            original_qa: 原始 QA 結果
            translated_qa: 翻譯 QA 結果
            quality_scores: 品質分數字典
            
        Returns:
            改進建議列表
        """
        suggestion_prompt = f"""
        基於以下翻譯品質評估結果，請提供具體的翻譯改進建議：

        **原始英文問答：**
        問題：{original_qa.input_question}
        回答：{original_qa.generated_answer}

        **翻譯中文問答：**
        問題：{translated_qa.input_question}
        回答：{translated_qa.generated_answer}

        **品質評估分數：**
        - 語義一致性：{quality_scores.get('semantic_consistency', 0):.1f}/10
        - 程式碼完整性：{quality_scores.get('code_integrity', 0):.1f}/10
        - 翻譯自然度：{quality_scores.get('translation_naturalness', 0):.1f}/10

        請針對以下問題領域提供具體、可操作的改進建議：

        1. **語義一致性改進**：如何確保翻譯保持原文的技術含義
        2. **程式碼保護**：如何正確處理程式碼片段和技術術語
        3. **翻譯自然度**：如何提升中文表達的自然度和可讀性
        4. **術語一致性**：如何統一技術術語的翻譯
        5. **結構保持**：如何維持原文的邏輯結構

        請提供 3-5 條具體的改進建議，每條建議應該：
        - 針對具體問題
        - 提供可操作的解決方案
        - 考慮程式設計領域的特殊性

        格式：每行一條建議，以「- 」開頭。
        """
        
        try:
            messages = [{"role": "user", "content": suggestion_prompt}]
            response = self.llm_service.invoke(messages)
            
            # 解析建議
            response_text = response.content.strip()
            
            # 提取以 "- " 開頭的建議
            import re
            suggestions = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('- '):
                    suggestion = line[2:].strip()  # 移除 "- " 前綴
                    if suggestion:
                        suggestions.append(suggestion)
            
            # 如果沒有找到格式化的建議，嘗試分割整個回應
            if not suggestions:
                # 分割回應為句子，取前幾個作為建議
                sentences = [s.strip() for s in response_text.split('.') if s.strip()]
                suggestions = sentences[:5]
            
            # 確保至少有一些建議
            if not suggestions:
                suggestions = [
                    "檢查技術術語翻譯的一致性和準確性",
                    "確保程式碼片段在翻譯過程中保持完整",
                    "提升中文表達的自然度和流暢性",
                    "保持原文的邏輯結構和技術深度"
                ]
            
            return suggestions[:5]  # 限制建議數量
            
        except Exception as e:
            self.logger.warning(f"Failed to generate improvement suggestions: {e}")
            # 回退到基本建議
            return [
                "改善語義一致性：確保翻譯保持原文的核心技術意圖",
                "保護程式碼完整性：確保程式碼區塊不被誤譯或遺失",
                "提升翻譯自然度：使用更符合繁體中文習慣的表達方式",
                "保持技術術語一致性：使用業界標準的繁體中文技術術語"
            ]
    
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
            
            # 生成改進建議 (現在是實例方法)
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
            
            updates: StateUpdateValue = {
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
