from hddu.config import (
    create_text_model,
    TRANSLATION_TEMPERATURE,
    TRANSLATION_BATCH_SIZE,
    TRANSLATION_MAX_RETRIES,
    TRANSLATION_RETRY_DELAY,
    TRANSLATION_BATCH_REDUCTION_FACTOR,
    TRANSLATION_TARGET_LANGUAGE
)

import re
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from .state import ParseState



from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationTask:
    element_id: str
    field_type: str
    original_text: str
    element_index: int
    needs_translation: bool = False
    target_language: str = "English"


@dataclass
class TranslationMapping:
    """번역 결과 매핑"""
    element_id: str
    element_index: int
    translations: Dict[str, str]


class LanguageDetection(BaseModel):
    needs_translation: bool = Field(description="텍스트가 번역이 필요한지 여부")
    detected_language: str = Field(description="감지된 언어")
    target_language: str = Field(description="번역 대상 언어 (English/Korean)")
    confidence: float = Field(description="감지 신뢰도 (0.0-1.0)")
    reason: str = Field(description="판단 이유")


class TranslationResult(BaseModel):
    translated_text: str = Field(description="번역된 텍스트")
    original_language: str = Field(description="원본 언어")
    target_language: str = Field(description="대상 언어", default="Korean")


@dataclass
class TranslationConfig:
    model_name: str = None
    temperature: float = TRANSLATION_TEMPERATURE
    batch_size: int = TRANSLATION_BATCH_SIZE
    max_retries: int = TRANSLATION_MAX_RETRIES
    verbose: bool = True
    target_language: str = TRANSLATION_TARGET_LANGUAGE
    
    def __post_init__(self):
        if self.model_name is not None:
            logger.warning("model_name 파라미터는 더 이상 사용되지 않습니다. .env 파일에서 TEXT_LLM_PROVIDER를 설정하세요.")


class MultiFieldTextProcessor:
    
    @staticmethod
    def clean_text(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        try:
            cleaned = text.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Text cleaning error: {e}")
            return str(text).strip() if text else ""
    
    @staticmethod
    def is_empty_or_invalid(text: str) -> bool:
        if not text:
            return True
        
        cleaned = MultiFieldTextProcessor.clean_text(text)
        if not cleaned:
            return True
        
        meaningful_chars = re.sub(r'[\s\-\=\|\+\*#<>\/\\]+', '', cleaned)
        return len(meaningful_chars) < 2
    
    @staticmethod
    def extract_translation_tasks(elements: List[Dict]) -> List[TranslationTask]:
        tasks = []
        
        for i, element in enumerate(elements):
            try:
                element_id = element.get("id", f"element_{i}")
                
                content = element.get("content", {})
                
                for field_type in ["text", "markdown", "html"]:
                    field_value = content.get(field_type, "")
                    cleaned_text = MultiFieldTextProcessor.clean_text(field_value)
                    
                    if not MultiFieldTextProcessor.is_empty_or_invalid(cleaned_text):
                        task = TranslationTask(
                            element_id=element_id,
                            field_type=field_type,
                            original_text=cleaned_text,
                            element_index=i
                        )
                        tasks.append(task)
                    
            except Exception as e:
                logger.warning(f"Element {i} processing error: {e}")
                continue
        
        return tasks
    
    @staticmethod
    def prepare_batch_from_tasks(tasks: List[TranslationTask]) -> List[str]:
        return [task.original_text for task in tasks]


class TextProcessor:
    
    @staticmethod
    def clean_text(text: str) -> str:
        return MultiFieldTextProcessor.clean_text(text)
    
    @staticmethod
    def is_empty_or_invalid(text: str) -> bool:
        return MultiFieldTextProcessor.is_empty_or_invalid(text)
    
    @staticmethod
    def prepare_batch(elements: List[Dict]) -> List[str]:
        texts = []
        for element in elements:
            try:
                markdown = element.get("content", {}).get("markdown", "")
                cleaned = TextProcessor.clean_text(markdown)
                
                if TextProcessor.is_empty_or_invalid(cleaned):
                    texts.append("[빈 텍스트]")
                else:
                    texts.append(cleaned)
                    
            except Exception as e:
                logger.warning(f"Element processing error: {e}")
                texts.append("[Unprocessable text]")
        
        return texts


class TranslationTaskManager:
    
    @staticmethod
    def update_task_translation_needs(
        tasks: List[TranslationTask], 
        detection_results: List[LanguageDetection]
    ) -> List[TranslationTask]:
        if len(tasks) != len(detection_results):
            logger.warning(f"Task count ({len(tasks)}) does not match detection result count ({len(detection_results)})")
            
        for i, (task, detection) in enumerate(zip(tasks, detection_results)):
            task.needs_translation = detection.needs_translation
            task.target_language = detection.target_language
            
        return tasks
    
    @staticmethod
    def create_translation_mappings(
        tasks: List[TranslationTask], 
        translation_results: List[TranslationResult]
    ) -> List[TranslationMapping]:
        element_groups: Dict[str, Dict] = {}
        
        translation_tasks = [task for task in tasks if task.needs_translation]
        
        if len(translation_tasks) != len(translation_results):
            logger.warning(f"Translation task count ({len(translation_tasks)}) does not match translation result count ({len(translation_results)})")
        
        for task, result in zip(translation_tasks, translation_results):
            if task.element_id not in element_groups:
                element_groups[task.element_id] = {
                    'element_index': task.element_index,
                    'translations': {}
                }
            
            element_groups[task.element_id]['translations'][task.field_type] = result.translated_text
        
        mappings = []
        for element_id, group_data in element_groups.items():
            mapping = TranslationMapping(
                element_id=element_id,
                element_index=group_data['element_index'],
                translations=group_data['translations']
            )
            mappings.append(mapping)
        
        return mappings


class ResultMapper:
    
    @staticmethod
    def apply_translations_to_elements(
        original_elements: List[Dict], 
        mappings: List[TranslationMapping]
    ) -> List[Dict]:
        updated_elements = [element.copy() for element in original_elements]
        
        for i, element in enumerate(updated_elements):
            if 'content' in element:
                updated_elements[i]['content'] = element['content'].copy()
        
        id_to_mapping = {mapping.element_id: mapping for mapping in mappings}
        
        for i, element in enumerate(updated_elements):
            element_id = element.get("id", f"element_{i}")
            
            if element_id in id_to_mapping:
                mapping = id_to_mapping[element_id]
                
                if 'content' not in element:
                    element['content'] = {}
                
                for field_type, translated_text in mapping.translations.items():
                    translation_field = f"translation_{field_type}"
                    element['content'][translation_field] = translated_text
        
        return updated_elements


class LanguageDetector:
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.llm = create_text_model(
            temperature=config.temperature
        )
        
        self.structured_llm = self.llm.with_structured_output(
            LanguageDetection,
            method="json_mode"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a language detection expert for bidirectional translation. Analyze the given text and determine:
1. What language it is written in
2. Whether it needs translation (bidirectional logic)
3. The appropriate target language
4. Your confidence level (0.0 to 1.0)
5. The reason for your decision

Bidirectional Translation Guidelines:
- If text is in English → translate to Korean (target_language: "Korean")
- If text is in Korean or other non-English languages → translate to English (target_language: "English")
- If text is empty, contains only symbols/numbers, or is "[빈 텍스트]" or "[처리 불가능한 텍스트]", no translation needed
- Mixed language text should be treated based on the dominant language

Respond in JSON format with exactly these fields:
- "needs_translation": boolean (true/false)
- "detected_language": string (language name)
- "target_language": string ("English" or "Korean")
- "confidence": float (0.0 to 1.0)
- "reason": string (explanation)

Examples:
{{
  "needs_translation": true,
  "detected_language": "English",
  "target_language": "Korean",
  "confidence": 0.95,
  "reason": "English text detected, will translate to Korean"
}}

{{
  "needs_translation": true,
  "detected_language": "Korean",
  "target_language": "English", 
  "confidence": 0.90,
  "reason": "Korean text detected, will translate to English"
}}"""),
            ("human", "Analyze this text: {text}")
        ])
        
        self.chain = self.prompt | self.structured_llm
    
    def detect_batch(self, texts: List[str]) -> List[LanguageDetection]:
        results = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_results = self._process_batch_with_retry(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch_with_retry(self, batch: List[str]) -> List[LanguageDetection]:
        current_batch_size = len(batch)
        current_batch = batch.copy()
        
        for retry_count in range(self.config.max_retries):
            try:
                if self.config.verbose:
                    logger.info(f"LanguageDetector: 배치 크기 {current_batch_size}로 처리 시도 {retry_count + 1}/{self.config.max_retries}")
                
                results = self._process_batch(current_batch)
                if self.config.verbose:
                    logger.info(f"LanguageDetector: 배치 처리 완료 - {len(results)}개 결과")
                return results
                
            except Exception as e:
                if self.config.verbose:
                    logger.error(f"LanguageDetector: 배치 처리 실패 ({retry_count + 1}/{self.config.max_retries}): {str(e)}")
                
                if retry_count < self.config.max_retries - 1:
                    new_batch_size = max(1, int(current_batch_size * TRANSLATION_BATCH_REDUCTION_FACTOR))
                    if new_batch_size < current_batch_size:
                        current_batch = current_batch[:new_batch_size]
                        current_batch_size = new_batch_size
                        if self.config.verbose:
                            logger.info(f"LanguageDetector: 배치 크기를 {current_batch_size}로 감소")
                    
                    wait_time = TRANSLATION_RETRY_DELAY * (2 ** retry_count)
                    if self.config.verbose:
                        logger.info(f"LanguageDetector: {wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                else:
                    if self.config.verbose:
                        logger.info("LanguageDetector: 개별 처리로 fallback")
                    return self._process_individual_fallback(batch)
        
        return self._process_individual_fallback(batch)
    
    def _process_individual_fallback(self, batch: List[str]) -> List[LanguageDetection]:
        results = []
        for text in batch:
            try:
                result = self.chain.invoke({"text": text})
                results.append(result)
            except Exception as e:
                if self.config.verbose:
                    logger.error(f"LanguageDetector: 개별 처리도 실패: {str(e)}")
                results.append(LanguageDetection(
                    needs_translation=False,
                    detected_language="Unknown",
                    target_language="English",
                    confidence=0.0,
                    reason="Detection failed"
                ))
        return results
    
    def _process_batch(self, batch: List[str]) -> List[LanguageDetection]:
        results = []
        
        for text in batch:
            result = self.chain.invoke({"text": text})
            results.append(result)
        
        return results


class TextTranslator:
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.llm = create_text_model(
            temperature=config.temperature
        )
        
        self.structured_llm = self.llm.with_structured_output(
            TranslationResult,
            method="json_mode"
        )
    

    
    def translate_batch_with_targets(self, translation_tasks: List[TranslationTask]) -> List[TranslationResult]:
        tasks_to_translate = [task for task in translation_tasks if task.needs_translation]
        
        if not tasks_to_translate:
            return []
        
        if self.config.verbose:
            logger.info(f"TextTranslator: 번역 작업 시작 - {len(tasks_to_translate)}개 작업")
        
        language_groups = {}
        task_index_map = {}
        
        for i, task in enumerate(tasks_to_translate):
            if task.target_language not in language_groups:
                language_groups[task.target_language] = []
            
            language_groups[task.target_language].append(task)
            task_index_map[id(task)] = i
        
        all_results = [None] * len(tasks_to_translate)
        
        for target_language, tasks in language_groups.items():
            if self.config.verbose:
                logger.info(f"TextTranslator: {target_language} 번역 시작 - {len(tasks)}개 작업")
            
            group_results = []
            for i in range(0, len(tasks), self.config.batch_size):
                batch_tasks = tasks[i:i + self.config.batch_size]
                batch_results = self._process_batch_with_retry(batch_tasks)
                group_results.extend(batch_results)
            
            for task, result in zip(tasks, group_results):
                original_index = task_index_map[id(task)]
                all_results[original_index] = result
        
        final_results = [result for result in all_results if result is not None]
        
        if self.config.verbose:
            logger.info(f"TextTranslator: 번역 완료 - {len(final_results)}개 결과")
        
        return final_results
    
    def _process_batch_with_retry(self, batch_tasks: List[TranslationTask]) -> List[TranslationResult]:
        current_batch_size = len(batch_tasks)
        current_batch = batch_tasks.copy()
        
        for retry_count in range(self.config.max_retries):
            try:
                if self.config.verbose:
                    logger.info(f"TextTranslator: 배치 크기 {current_batch_size}로 처리 시도 {retry_count + 1}/{self.config.max_retries}")
                
                results = self._process_batch(current_batch)
                if self.config.verbose:
                    logger.info(f"TextTranslator: 배치 처리 완료 - {len(results)}개 결과")
                return results
                
            except Exception as e:
                if self.config.verbose:
                    logger.error(f"TextTranslator: 배치 처리 실패 ({retry_count + 1}/{self.config.max_retries}): {str(e)}")
                
                if retry_count < self.config.max_retries - 1:
                    new_batch_size = max(1, int(current_batch_size * TRANSLATION_BATCH_REDUCTION_FACTOR))
                    if new_batch_size < current_batch_size:
                        current_batch = current_batch[:new_batch_size]
                        current_batch_size = new_batch_size
                        if self.config.verbose:
                            logger.info(f"TextTranslator: 배치 크기를 {current_batch_size}로 감소")
                    
                    wait_time = TRANSLATION_RETRY_DELAY * (2 ** retry_count)
                    if self.config.verbose:
                        logger.info(f"TextTranslator: {wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                else:
                    if self.config.verbose:
                        logger.info("TextTranslator: 개별 처리로 fallback")
                    return self._process_individual_fallback(batch_tasks)
        
        return self._process_individual_fallback(batch_tasks)
    
    def _process_individual_fallback(self, batch_tasks: List[TranslationTask]) -> List[TranslationResult]:
        results = []
        for task in batch_tasks:
            try:
                result = self._translate_single_with_target(task.original_text, task.target_language)
                results.append(result)
            except Exception as e:
                if self.config.verbose:
                    logger.error(f"TextTranslator: 개별 처리도 실패: {str(e)}")
                results.append(TranslationResult(
                    translated_text=task.original_text,
                    original_language="Unknown",
                    target_language=task.target_language
                ))
        return results
    
    def _process_batch(self, batch_tasks: List[TranslationTask]) -> List[TranslationResult]:
        results = []
        
        for task in batch_tasks:
            result = self._translate_single_with_target(task.original_text, task.target_language)
            results.append(result)
        
        return results
    
    def _translate_single_with_target(self, text: str, target_language: str) -> TranslationResult:
        dynamic_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a professional translator. Translate the given text to {target_language}.

Important instructions:
- Maintain the original markdown formatting
- Preserve any special characters or symbols
- Keep the meaning and tone of the original text
- If the text is already in {target_language}, just return it as is
- Provide the detected original language

Respond in JSON format with exactly these fields:
- "translated_text": string (the translated text)
- "original_language": string (detected language of input)
- "target_language": string (always "{target_language}")

Example for English to Korean:
{{{{
  "translated_text": "번역된 텍스트",
  "original_language": "English",
  "target_language": "Korean"
}}}}

Example for Korean to English:
{{{{
  "translated_text": "Translated text",
  "original_language": "Korean", 
  "target_language": "English"
}}}}"""),
            ("human", "Translate this text: {text}")
        ])
        
        dynamic_chain = dynamic_prompt | self.structured_llm
        return dynamic_chain.invoke({"text": text})
    



class TranslationState(TypedDict):
    elements: List[Dict]
    translation_tasks: List[TranslationTask]
    detection_results: List[LanguageDetection]
    translation_results: List[TranslationResult]
    translation_mappings: List[TranslationMapping]
    updated_elements: List[Dict]


class TranslationWorkflow:
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.detector = LanguageDetector(config)
        self.translator = TextTranslator(config)
    
    def create_graph(self) -> StateGraph:
        
        def extract_tasks_node(state: TranslationState) -> TranslationState:
            elements = state["elements"]
            
            if self.config.verbose:
                logger.info(f"Translation tasks extraction started - {len(elements)} elements")
            
            translation_tasks = MultiFieldTextProcessor.extract_translation_tasks(elements)
            
            if self.config.verbose:
                logger.info(f"Translation tasks extraction completed - {len(translation_tasks)} tasks")
                
                field_stats = {}
                for task in translation_tasks:
                    field_stats[task.field_type] = field_stats.get(task.field_type, 0) + 1
                
                for field_type, count in field_stats.items():
                    logger.info(f"  - {field_type}: {count} tasks")
            
            return {"translation_tasks": translation_tasks}
        
        def detect_language_node(state: TranslationState) -> TranslationState:
            translation_tasks = state["translation_tasks"]
            
            if not translation_tasks:
                if self.config.verbose:
                    logger.info("No translation tasks")
                return {"detection_results": []}
            
            if self.config.verbose:
                logger.info(f"Language detection started - {len(translation_tasks)} tasks")
            
            texts = MultiFieldTextProcessor.prepare_batch_from_tasks(translation_tasks)
            
            detection_results = self.detector.detect_batch(texts)
            
            updated_tasks = TranslationTaskManager.update_task_translation_needs(
                translation_tasks, detection_results
            )
            
            if self.config.verbose:
                need_translation = sum(1 for task in updated_tasks if task.needs_translation)
                no_translation = len(updated_tasks) - need_translation
                logger.info(f"Language detection completed - {need_translation} tasks need translation, {no_translation} tasks remain original")
            
            return {
                "translation_tasks": updated_tasks,
                "detection_results": detection_results
            }
        
        def translate_node(state: TranslationState) -> TranslationState:
            translation_tasks = state["translation_tasks"]
            
            tasks_to_translate = [task for task in translation_tasks if task.needs_translation]
            
            if not tasks_to_translate:
                if self.config.verbose:
                    logger.info("No tasks to translate")
                return {
                    "translation_results": [],
                    "translation_mappings": []
                }
            
            if self.config.verbose:
                logger.info(f"Translation started - {len(tasks_to_translate)} tasks")
                
                en_to_kr = sum(1 for task in tasks_to_translate if task.target_language == "Korean")
                kr_to_en = sum(1 for task in tasks_to_translate if task.target_language == "English")
                logger.info(f"  - English to Korean: {en_to_kr} tasks")
                logger.info(f"  - Korean to English: {kr_to_en} tasks")
            
            translation_results = self.translator.translate_batch_with_targets(tasks_to_translate)
            
            translation_mappings = TranslationTaskManager.create_translation_mappings(
                translation_tasks, translation_results
            )
            
            if self.config.verbose:
                logger.info(f"Translation completed - {len(translation_results)} texts translated")
                logger.info(f"Translation mapping created - {len(translation_mappings)} elements")
            
            return {
                "translation_results": translation_results,
                "translation_mappings": translation_mappings
            }
        
        def apply_results_node(state: TranslationState) -> TranslationState:
            elements = state["elements"]
            translation_mappings = state.get("translation_mappings", [])
            
            if self.config.verbose:
                logger.info(f"Translation results applied - {len(translation_mappings)} mappings")
            
            updated_elements = ResultMapper.apply_translations_to_elements(
                elements, translation_mappings
            )
            
            if self.config.verbose:
                logger.info(f"Translation results applied - {len(updated_elements)} elements updated")
            
            return {"updated_elements": updated_elements}
        
        def should_translate(state: TranslationState) -> str:
            translation_tasks = state.get("translation_tasks", [])
            
            needs_translation = any(task.needs_translation for task in translation_tasks)
            
            if needs_translation:
                if self.config.verbose:
                    logger.info("Translation node")
                return "translate"
            else:
                if self.config.verbose:
                    logger.info("Translation not needed, moving to apply_results node")
                return "apply_results"
        
        workflow = StateGraph(TranslationState)
        
        workflow.add_node("extract_tasks", extract_tasks_node)
        workflow.add_node("detect", detect_language_node)
        workflow.add_node("translate", translate_node)
        workflow.add_node("apply_results", apply_results_node)
        
        workflow.add_edge(START, "extract_tasks")
        workflow.add_edge("extract_tasks", "detect")
        workflow.add_conditional_edges(
            "detect",
            should_translate,
            {
                "translate": "translate",
                "apply_results": "apply_results"
            }
        )
        workflow.add_edge("translate", "apply_results")
        workflow.add_edge("apply_results", END)
        
        return workflow.compile()


def add_translation_module(
    state: ParseState,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    batch_size: int = 20,
    max_retries: int = 3,
    verbose: bool = True,
    target_language: str = "auto"
) -> ParseState:

    try:
        config = TranslationConfig(
            model_name=model_name,
            temperature=temperature,
            batch_size=batch_size,
            max_retries=max_retries,
            verbose=verbose,
            target_language=target_language
        )
        
        if verbose:
            logger.info(f"Translation module started")
            logger.info(f"Translation mode: {target_language} (auto = bidirectional translation)")
        
        elements_to_translate = state.get("elements_from_parser", [])
        
        if not elements_to_translate:
            if verbose:
                logger.info("No elements to translate")
            return {}
        
        workflow = TranslationWorkflow(config)
        graph = workflow.create_graph()
        
        translation_state = {
            "elements": elements_to_translate.copy(),
            "translation_tasks": [],
            "detection_results": [],
            "translation_results": [],
            "translation_mappings": [],
            "updated_elements": []
        }
        
        result = graph.invoke(translation_state)
        
        updated_elements = result.get("updated_elements", elements_to_translate)

        return_dict = {
            "elements_from_parser": updated_elements
        }
        
        if target_language == "auto":
            return_dict["language"] = "bidirectional"
        else:
            return_dict["language"] = target_language
        
        if verbose:
            original_count = len(elements_to_translate)
            updated_count = len(updated_elements)
            
            translation_stats = {"translation_text": 0, "translation_markdown": 0, "translation_html": 0}
            for element in updated_elements:
                content = element.get("content", {})
                for field in translation_stats.keys():
                    if content.get(field):
                        translation_stats[field] += 1
            
            logger.info(f"Translation module completed - {original_count} elements processed")
            for field, count in translation_stats.items():
                if count > 0:
                    logger.info(f"  - {field}: {count} fields translated")
        
        return return_dict
        
    except Exception as e:
        logger.error(f"Translation module error: {e}")
        return {}


def create_translation_graph(**kwargs) -> StateGraph:
    config = TranslationConfig(**kwargs)
    workflow = TranslationWorkflow(config)
    return workflow.create_graph()

