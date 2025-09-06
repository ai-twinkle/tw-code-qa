"""
分析設計者 Agent 節點
Analyzer Designer Agent Node

負責語義分析與序列式翻譯執行：
1. 語義複雜度分析
2. Question 和 Answer 的序列式翻譯（先 Question 後 Answer）
3. 程式碼區塊識別與保護
4. 基於 Prompt 的術語一致性管理
"""

import json
import logging
import re
import time
from typing import List

from typing_extensions import TypedDict

from ..state import WorkflowState, update_state_safely, StateUpdateValue
from ...config.llm_config import get_agent_config
from ...constants.llm import LLMProvider, LLMModel
from ...models.dataset import TranslationResult, ProcessingStatus
from ...models.quality import ErrorRecord, ErrorType
from ...services.llm_service import LLMFactory

logger = logging.getLogger(__name__)


class SemanticComplexityContext(TypedDict):
    """語義複雜度分析上下文類型定義"""
    complexity: str  # "Simple", "Medium", "Complex"
    programming_languages: List[str]
    key_concepts: List[str]
    code_block_count: int
    translation_challenges: List[str]


class AnalyzerDesignerAgent:
    """分析設計者 Agent 類別"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".AnalyzerDesignerAgent")
        
        # 從配置取得模型設定
        agent_config = get_agent_config("analyzer_designer")
        if not agent_config:
            raise ValueError("Failed to get analyzer_designer configuration")
        
        self.primary_model = agent_config["primary_model"]
        self.fallback_model = agent_config.get("fallback_model")
        self.temperature = agent_config["temperature"]
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
                provider = LLMProvider.OPENAI  # 預設
                
            model_enum = LLMModel(self.primary_model)
            return LLMFactory.create_llm(provider, model_enum)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {e}")
            raise

    def analyze_semantic_complexity(self, question: str, answer: str) -> SemanticComplexityContext:
        """
        分析語義複雜度

        Args:
            question: 英文問題
            answer: 英文答案

        Returns:
            語義複雜度分析結果
        """
        analysis_prompt = f"""
        分析語義複雜度 (Analyze the semantic complexity) of the following programming Q&A pair and return your analysis in JSON format.

        **Question:**
        {question}

        **Answer:**
        {answer}

        Please provide analysis for:
        1. Complexity level: "Simple", "Medium", or "Complex"
        2. Programming languages involved (list)
        3. Key technical concepts (list)
        4. Number of code blocks
        5. Translation challenges (list)

        Response format:
        {{
            "complexity": "Simple|Medium|Complex",
            "programming_languages": ["language1", "language2"],
            "key_concepts": ["concept1", "concept2"],
            "code_block_count": number,
            "translation_challenges": ["challenge1", "challenge2"]
        }}

        Complexity criteria:
        - Simple: Basic syntax, single concept, minimal code
        - Medium: Multiple concepts, moderate code complexity, some abstractions
        - Complex: Advanced patterns, multiple languages, complex algorithms, extensive code

        Only return the JSON, no additional explanation.
        """

        try:
            messages = [{"role": "user", "content": analysis_prompt}]
            response = self.llm_service.invoke(messages)

            # 解析 LLM 回應
            response_text = response.content.strip()

            # 嘗試從回應中提取 JSON
            json_match = re.search(r'\{.*}', response_text, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group())

                    # 驗證和標準化分析結果
                    complexity = analysis_data.get("complexity", "Medium")
                    if complexity not in ["Simple", "Medium", "Complex"]:
                        complexity = "Medium"

                    programming_languages = analysis_data.get("programming_languages", [])
                    if not isinstance(programming_languages, list):
                        programming_languages = []

                    key_concepts = analysis_data.get("key_concepts", [])
                    if not isinstance(key_concepts, list):
                        key_concepts = ["programming"]

                    code_block_count = analysis_data.get("code_block_count", 0)
                    if not isinstance(code_block_count, int):
                        # 回退到計算 ``` 對數
                        code_block_count = answer.count("```") // 2

                    translation_challenges = analysis_data.get("translation_challenges", [])
                    if not isinstance(translation_challenges, list):
                        translation_challenges = ["technical_terms"]

                    # 補充分析：如果 LLM 沒有檢測到程式語言，嘗試自動檢測
                    if not programming_languages:
                        programming_languages = self._detect_programming_languages(question + " " + answer)

                    # 補充分析：如果沒有檢測到關鍵概念，基於內容分析
                    if not key_concepts:
                        key_concepts = self._extract_key_concepts(question + " " + answer)

                    return {
                        "complexity": complexity,
                        "programming_languages": programming_languages,
                        "key_concepts": key_concepts,
                        "code_block_count": code_block_count,
                        "translation_challenges": translation_challenges
                    }

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse LLM JSON response: {e}")
                    self.logger.debug(f"Raw response: {response_text[:500]}...")

            # 如果 JSON 解析失敗，使用基於規則的分析作為回退
            return self._fallback_complexity_analysis(question, answer)

        except Exception as e:
            self.logger.warning(f"Failed to analyze semantic complexity: {e}")
            return self._fallback_complexity_analysis(question, answer)

    @staticmethod
    def _detect_programming_languages(text: str) -> List[str]:
        """
        基於內容檢測程式語言
        
        Args:
            text: 要分析的文本
            
        Returns:
            檢測到的程式語言列表
        """
        languages = []
        text_lower = text.lower()
        
        # 常見程式語言關鍵詞檢測
        language_keywords = {
            "Python": ["python", "def ", "import ", "from ", "__init__", "self.", "pip", "numpy", "pandas"],
            "JavaScript": ["javascript", "js", "function", "const ", "let ", "var ", "=>", "npm", "node"],
            "Java": ["java", "class ", "public ", "private ", "static ", "void ", "import java"],
            "C++": ["c++", "cpp", "#include", "std::", "cout", "cin", "namespace"],
            "C": ["#include", "printf", "scanf", "main()", "malloc", "free"],
            "SQL": ["select ", "from ", "where ", "insert ", "update ", "delete ", "join"],
            "HTML": ["<html", "<div", "<span", "<body", "<!doctype"],
            "CSS": ["css", "{", "}", "color:", "background:", "margin:", "padding:"],
            "Shell": ["bash", "#!/bin", "echo ", "grep ", "awk ", "sed "],
            "Go": ["golang", "go ", "func ", "package ", "import ", "fmt."],
            "Rust": ["rust", "fn ", "let mut", "use ", "cargo", "impl"],
            "TypeScript": ["typescript", "interface ", "type ", ": string", ": number"]
        }
        
        for language, keywords in language_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                languages.append(language)
        
        return languages if languages else ["Unknown"]

    @staticmethod
    def _extract_key_concepts(text: str) -> List[str]:
        """
        基於內容提取關鍵技術概念
        
        Args:
            text: 要分析的文本
            
        Returns:
            關鍵概念列表
        """
        concepts = []
        text_lower = text.lower()
        
        # 技術概念關鍵詞
        concept_keywords = {
            "web_development": ["web", "http", "html", "css", "frontend", "backend", "api", "rest"],
            "data_science": ["data", "analysis", "visualization", "pandas", "numpy", "matplotlib", "jupyter"],
            "machine_learning": ["ml", "ai", "model", "training", "neural", "tensorflow", "pytorch", "sklearn"],
            "database": ["database", "sql", "query", "table", "index", "join", "mysql", "postgresql"],
            "algorithms": ["algorithm", "sort", "search", "complexity", "time", "space", "optimization"],
            "object_oriented": ["class", "object", "inheritance", "polymorphism", "encapsulation"],
            "functional_programming": ["function", "lambda", "map", "filter", "reduce", "closure"],
            "testing": ["test", "unittest", "pytest", "mock", "assert", "coverage"],
            "security": ["security", "authentication", "encryption", "hash", "token", "ssl"],
            "performance": ["performance", "optimization", "memory", "cpu", "cache", "profiling"]
        }
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts if concepts else ["programming"]
    
    def _fallback_complexity_analysis(self, question: str, answer: str) -> SemanticComplexityContext:
        """
        當 LLM 服務不可用時的回退分析
        
        Args:
            question: 英文問題
            answer: 英文答案
            
        Returns:
            基礎複雜度分析結果（使用 Unknown 作為預設值）
        """
        self.logger.warning("Using fallback analysis due to LLM service unavailability")
        
        # 不嘗試猜測複雜度，直接使用 "Unknown"
        # 這比基於任意規則（如長度）的猜測更誠實和可預測
        complexity = "Unknown"
        
        # 檢測程式語言（保留這個功能，因為它基於明確的關鍵詞）
        combined_text = question + " " + answer
        programming_languages = self._detect_programming_languages(combined_text)
        if not programming_languages:
            programming_languages = ["programming"]
            
        # 檢測技術概念（基於關鍵詞，相對可靠）
        key_concepts = self._extract_key_concepts(combined_text)
        if not key_concepts:
            key_concepts = ["unknown_concepts"]
        
        # 計算程式碼區塊數量（這是可以準確計算的）
        code_block_count = answer.count("```") // 2
        
        # 翻譯挑戰無法準確評估，使用空列表
        translation_challenges = []
        
        return {
            "complexity": complexity,
            "programming_languages": programming_languages,
            "key_concepts": key_concepts,
            "code_block_count": code_block_count,
            "translation_challenges": translation_challenges
        }
    
    def translate_question(self, question: str, context: SemanticComplexityContext) -> str:
        """
        翻譯問題
        
        Args:
            question: 英文問題
            context: 語義分析上下文
            
        Returns:
            繁體中文翻譯
        """
        translation_prompt = f"""
        請將以下英文程式碼問題翻譯成繁體中文：

        **原文問題：**
        {question}

        **翻譯要求：**
        1. 保持技術術語的準確性
        2. 使用台灣地區的繁體中文表達習慣
        3. 保護程式碼片段不被翻譯
        4. 保持原文的邏輯結構和格式
        5. 專業術語使用業界標準翻譯

        **語義上下文：**
        - 複雜度：{context.get('complexity', 'Medium')}
        - 程式語言：{', '.join(context.get('programming_languages', ['Python']))}
        - 主要概念：{', '.join(context.get('key_concepts', ['programming']))}

        請直接回答翻譯結果，不需要額外解釋。
        """
        
        try:
            messages = [{"role": "user", "content": translation_prompt}]
            response = self.llm_service.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to translate question: {e}")
            raise
    
    def translate_answer(self, question: str, answer: str, translated_question: str, context: SemanticComplexityContext) -> str:
        """
        根據翻譯後的問題翻譯答案
        
        Args:
            question: 原始英文問題
            answer: 原始英文答案
            translated_question: 已翻譯的中文問題
            context: 語義分析上下文
            
        Returns:
            繁體中文翻譯
        """
        translation_prompt = f"""
        請將以下英文程式碼答案翻譯成繁體中文：

        **原始英文問題：**
        {question}

        **已翻譯的中文問題：**
        {translated_question}

        **原始英文答案：**
        {answer}

        **翻譯要求：**
        1. 答案必須與翻譯後的中文問題保持一致性
        2. 保持技術術語的準確性和一致性
        3. 使用台灣地區的繁體中文表達習慣
        4. 嚴格保護程式碼片段（```內容```），不得翻譯
        5. 保持原答案的邏輯結構、格式和完整性
        6. 註解內容可以翻譯，但變數名、函數名等保持原文
        7. 確保技術解釋清晰易懂

        **語義上下文：**
        - 複雜度：{context.get('complexity', 'Medium')}
        - 程式語言：{', '.join(context.get('programming_languages', ['Python']))}
        - 程式碼區塊數：{context.get('code_block_count', 0)}

        請直接回答翻譯結果，不需要額外解釋。
        """
        
        try:
            messages = [{"role": "user", "content": translation_prompt}]
            response = self.llm_service.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to translate answer: {e}")
            raise
    
    def perform_translation(self, record_id: str, question: str, answer: str) -> TranslationResult:
        """
        執行完整的翻譯流程
        
        Args:
            record_id: 記錄ID
            question: 英文問題
            answer: 英文答案
            
        Returns:
            翻譯結果
        """
        start_time = time.time()
        
        try:
            # 1. 語義複雜度分析
            self.logger.info(f"Analyzing semantic complexity for record {record_id}")
            context = self.analyze_semantic_complexity(question, answer)
            
            # 2. 序列式翻譯：先翻譯問題
            self.logger.info(f"Translating question for record {record_id}")
            translated_question = self.translate_question(question, context)
            
            # 3. 基於翻譯後問題翻譯答案
            self.logger.info(f"Translating answer for record {record_id}")
            translated_answer = self.translate_answer(question, answer, translated_question, context)
            
            # 4. 創建翻譯結果
            translation_result = TranslationResult(
                original_record_id=record_id,
                translated_question=translated_question,
                translated_answer=translated_answer,
                translation_strategy="sequential_context_aware",
                terminology_notes=[
                    f"Complexity: {context.get('complexity')}",
                    f"Languages: {', '.join(context.get('programming_languages', []))}",
                    f"Code blocks: {context.get('code_block_count', 0)}"
                ]
            )
            
            processing_time = time.time() - start_time
            self.logger.info(f"Translation completed for record {record_id} in {processing_time:.2f}s")
            
            return translation_result
            
        except Exception as e:
            self.logger.error(f"Translation failed for record {record_id}: {e}")
            raise


def analyzer_designer_node(state: WorkflowState) -> WorkflowState:
    """
    分析設計者節點 - LangGraph 節點函數
    
    負責語義分析與序列式翻譯執行
    
    Args:
        state: 當前工作流狀態
        
    Returns:
        更新後的工作流狀態
    """
    logger = logging.getLogger(__name__ + ".analyzer_designer_node")
    
    try:
        # 檢查當前記錄
        current_record = state["current_record"]
        if not current_record:
            raise ValueError("No current record to process")
        
        # 檢查是否已有翻譯結果且不需要重試
        if (state.get("translation_result") and 
            state["processing_status"] != ProcessingStatus.RETRY_NEEDED):
            logger.info(f"Translation already exists for record {current_record.id}, skipping")
            return state
        
        # 創建分析設計者 Agent
        agent = AnalyzerDesignerAgent()
        
        # 執行翻譯
        logger.info(f"Starting translation for record {current_record.id}")
        translation_result = agent.perform_translation(
            current_record.id,
            current_record.question,
            current_record.answer
        )
        
        # 更新狀態
        updates: StateUpdateValue = {
            "translation_result": translation_result,
            "processing_status": ProcessingStatus.PROCESSING
        }
        
        # 如果是重試，重置改進建議
        if state["processing_status"] == ProcessingStatus.RETRY_NEEDED:
            updates["improvement_suggestions"] = []
        
        return update_state_safely(state, updates)
        
    except Exception as e:
        logger.error(f"Analyzer Designer node failed: {e}")
        
        # 記錄錯誤
        error_record = ErrorRecord(
            error_type=ErrorType.TRANSLATION_QUALITY,
            error_message=str(e),
            timestamp=time.time(),
            retry_attempt=state.get("retry_count", 0),
            agent_name="analyzer_designer",
            recovery_action="logged_error"
        )
        
        # 更新狀態為失敗
        try:
            return update_state_safely(state, {
                "processing_status": ProcessingStatus.FAILED,
                "error_history": [error_record]  # 直接添加新錯誤，LangGraph會處理消息機制
            })
        except Exception:
            # 如果狀態更新失敗，至少更新錯誤歷史
            state["error_history"].append(error_record)
            state["processing_status"] = ProcessingStatus.FAILED
            return state


__all__ = [
    "AnalyzerDesignerAgent",
    "analyzer_designer_node",
]
