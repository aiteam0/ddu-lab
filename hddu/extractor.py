from hddu.config import (
    create_vision_model,
    EXTRACTOR_TEMPERATURE,
    EXTRACTOR_BATCH_SIZE,
    EXTRACTOR_MAX_TOKENS,
    EXTRACTOR_MAX_RETRIES,
    EXTRACTOR_RETRY_DELAY,
    EXTRACTOR_BATCH_REDUCTION_FACTOR
)

from langchain_core.runnables import chain
from langchain_utils.models import MultiModal
from .state import ParseState
from .base import BaseNode
from .preprocessing import IMAGE_TYPES, TEXT_TYPES, TABLE_TYPES
from langchain_core.prompts import load_prompt

import time
from typing import List, Dict, Optional, Tuple
from hddu.logging_config import get_logger

logger = get_logger(__name__)


class PageElementsExtractorNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PageElementsExtractorNode"

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]
        elements_by_page = dict()
        max_page = 0

        for elem in elements:
            page_num = int(elem.page)
            max_page = max(max_page, page_num)
            if page_num not in elements_by_page:
                elements_by_page[page_num] = []
            if elem.category in (IMAGE_TYPES + TABLE_TYPES):
                elements_by_page[page_num] = []
            elements_by_page[page_num].append(elem)

        texts_by_page = dict()
        images_by_page = dict()
        tables_by_page = dict()

        for page_num in range(1, max_page + 1):
            texts_by_page[page_num] = ""
            images_by_page[page_num] = []
            tables_by_page[page_num] = []

        for page_num, elems in elements_by_page.items():
            logger.debug(f"Page {page_num}:")
            for elem in elems:
                if elem.category in IMAGE_TYPES:
                    images_by_page[page_num].append(elem)
                elif elem.category in TABLE_TYPES:
                    tables_by_page[page_num].append(elem)
                else:
                    texts_by_page[page_num] += elem.content

        return {
            "texts_by_page": texts_by_page,
            "images_by_page": images_by_page,
            "tables_by_page": tables_by_page,
        }



@chain
def image_entity_extractor(data_batches):
    llm = create_vision_model(
        temperature=EXTRACTOR_TEMPERATURE,
        max_tokens=EXTRACTOR_MAX_TOKENS
    )

    system_prompt = load_prompt("hddu/prompts/IMAGE-SYSTEM-PROMPT.yaml", encoding="utf-8").template

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["context"]
        image_path = data_batch["image"]
        language = data_batch["language"]
        user_prompt_template = load_prompt(
            "hddu/prompts/IMAGE-USER-PROMPT.yaml", encoding="utf-8"
        ).template.format(context=context, language=language)
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    multimodal_llm = MultiModal(llm)

    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


@chain
def table_entity_extractor(data_batches):
    llm = create_vision_model(
        temperature=EXTRACTOR_TEMPERATURE,
        max_tokens=EXTRACTOR_MAX_TOKENS
    )

    system_prompt = load_prompt("hddu/prompts/TABLE-SYSTEM-PROMPT.yaml", encoding="utf-8").template

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["context"]
        image_path = data_batch["image"]
        language = data_batch["language"]
        user_prompt_template = load_prompt(
            "hddu/prompts/TABLE-USER-PROMPT.yaml", encoding="utf-8"
        ).template.format(context=context, language=language)

        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    multimodal_llm = MultiModal(llm)

    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


class ImageEntityExtractorNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ImageEntityExtractorNode"

    def execute(self, state: ParseState) -> ParseState:
        images_files = []
        for page_images in state["images_by_page"].values():
            images_files.extend(page_images)

        language = state["language"]
        extracted_image_entities = []

        valid_images = []
        skipped_elements = []
        
        for image_element in images_files:
            if (image_element.image_filename is None or 
                not isinstance(image_element.image_filename, str) or 
                not image_element.image_filename.strip()):
                
                if self.verbose:
                    logger.warning(f"ImageEntityExtractor: 이미지 파일명이 없는 요소 건너뛰기 - "
                                 f"ID {image_element.id}, 페이지 {image_element.page}")
                
                element = image_element.copy()
                element.entity = "No image file available for entity extraction"
                skipped_elements.append(element)
            else:
                valid_images.append(image_element)

        if self.verbose and skipped_elements:
            logger.info(f"ImageEntityExtractor: {len(skipped_elements)}개 요소를 이미지 파일명 부재로 건너뛰었습니다")

        for i in range(0, len(valid_images), EXTRACTOR_BATCH_SIZE):
            batch = valid_images[i : i + EXTRACTOR_BATCH_SIZE]
            batch_results = self._process_batch_with_retry(batch, state, language)
            extracted_image_entities.extend(batch_results)
        
        if self.verbose and skipped_elements:
            logger.info(f"ImageEntityExtractor: {len(skipped_elements)}개 요소가 최종 결과에서 제외됨")
        
        return {"extracted_image_entities": extracted_image_entities}
    
    def _process_batch_with_retry(self, batch: List, state: ParseState, language: str) -> List:
        """배치를 retry 전략과 함께 처리합니다."""
        current_batch_size = len(batch)
        current_batch = batch.copy()
        
        for retry_count in range(EXTRACTOR_MAX_RETRIES):
            try:
                if self.verbose:
                    logger.info(f"ImageEntityExtractor: 배치 크기 {current_batch_size}로 처리 시도 {retry_count + 1}/{EXTRACTOR_MAX_RETRIES}")
                
                return self._process_batch(current_batch, state, language)
                
            except Exception as e:
                if self.verbose:
                    logger.error(f"ImageEntityExtractor: 배치 처리 실패 ({retry_count + 1}/{EXTRACTOR_MAX_RETRIES}): {str(e)}")
                
                if retry_count < EXTRACTOR_MAX_RETRIES - 1:
                    new_batch_size = max(1, int(current_batch_size * EXTRACTOR_BATCH_REDUCTION_FACTOR))
                    if new_batch_size < current_batch_size:
                        current_batch = current_batch[:new_batch_size]
                        current_batch_size = new_batch_size
                        if self.verbose:
                            logger.info(f"ImageEntityExtractor: 배치 크기를 {current_batch_size}로 감소")
                    
                    wait_time = EXTRACTOR_RETRY_DELAY * (2 ** retry_count)
                    if self.verbose:
                        logger.info(f"ImageEntityExtractor: {wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        logger.info("ImageEntityExtractor: 개별 처리로 fallback")
                    return self._process_individual_fallback(batch, state, language)
        
        return self._process_individual_fallback(batch, state, language)
    
    def _process_individual_fallback(self, batch: List, state: ParseState, language: str) -> List:
        results = []
        for image_element in batch:
            try:
                batch_data = [{
                    "image": image_element.image_filename,
                    "context": state["texts_by_page"][image_element.page],
                    "language": language,
                }]
                batch_result = image_entity_extractor.invoke(batch_data)
                element = image_element.copy()
                element.entity = batch_result[0] if batch_result else "Entity extraction failed"
                results.append(element)
            except Exception as e:
                if self.verbose:
                    logger.error(f"ImageEntityExtractor: 개별 처리도 실패: {str(e)}")
                element = image_element.copy()
                element.entity = "Entity extraction failed"
                results.append(element)
        return results
    
    def _process_batch(self, batch: List, state: ParseState, language: str) -> List:
        """단일 배치를 처리합니다."""
        batch_data = []
        for image_element in batch:
            batch_data.append({
                "image": image_element.image_filename,
                "context": state["texts_by_page"][image_element.page],
                "language": language,
            })
        
        batch_result = image_entity_extractor.invoke(batch_data)
        
        results = []
        for j, result in enumerate(batch_result):
            element = batch[j].copy()
            element.entity = result
            results.append(element)
        
        return results


class TableEntityExtractorNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TableEntityExtractorNode"

    def execute(self, state: ParseState) -> ParseState:
        tables_files = []
        for page_tables in state["tables_by_page"].values():
            tables_files.extend(page_tables)

        language = state["language"]
        extracted_table_entities = []

        valid_tables = []
        skipped_elements = []
        
        for table_element in tables_files:
            if (table_element.image_filename is None or 
                not isinstance(table_element.image_filename, str) or 
                not table_element.image_filename.strip()):
                
                if self.verbose:
                    logger.warning(f"TableEntityExtractor: 이미지 파일명이 없는 테이블 요소 건너뛰기 - "
                                 f"ID {table_element.id}, 페이지 {table_element.page}")
                
                element = table_element.copy()
                element.entity = "No image file available for entity extraction"
                skipped_elements.append(element)
            else:
                valid_tables.append(table_element)

        if self.verbose and skipped_elements:
            logger.info(f"TableEntityExtractor: {len(skipped_elements)}개 요소를 이미지 파일명 부재로 건너뛰었습니다")

        for i in range(0, len(valid_tables), EXTRACTOR_BATCH_SIZE):
            batch = valid_tables[i : i + EXTRACTOR_BATCH_SIZE]
            batch_results = self._process_batch_with_retry(batch, state, language)
            extracted_table_entities.extend(batch_results)
        
        if self.verbose and skipped_elements:
            logger.info(f"TableEntityExtractor: {len(skipped_elements)}개 요소가 최종 결과에서 제외됨")
        
        return {"extracted_table_entities": extracted_table_entities}
    
    def _process_batch_with_retry(self, batch: List, state: ParseState, language: str) -> List:
        """배치를 retry 전략과 함께 처리합니다."""
        current_batch_size = len(batch)
        current_batch = batch.copy()
        
        for retry_count in range(EXTRACTOR_MAX_RETRIES):
            try:
                if self.verbose:
                    logger.info(f"TableEntityExtractor: 배치 크기 {current_batch_size}로 처리 시도 {retry_count + 1}/{EXTRACTOR_MAX_RETRIES}")
                
                return self._process_batch(current_batch, state, language)
                
            except Exception as e:
                if self.verbose:
                    logger.error(f"TableEntityExtractor: 배치 처리 실패 ({retry_count + 1}/{EXTRACTOR_MAX_RETRIES}): {str(e)}")
                
                if retry_count < EXTRACTOR_MAX_RETRIES - 1:
                    new_batch_size = max(1, int(current_batch_size * EXTRACTOR_BATCH_REDUCTION_FACTOR))
                    if new_batch_size < current_batch_size:
                        current_batch = current_batch[:new_batch_size]
                        current_batch_size = new_batch_size
                        if self.verbose:
                            logger.info(f"TableEntityExtractor: 배치 크기를 {current_batch_size}로 감소")
                    
                    wait_time = EXTRACTOR_RETRY_DELAY * (2 ** retry_count)
                    if self.verbose:
                        logger.info(f"TableEntityExtractor: {wait_time}초 대기 후 재시도")
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        logger.info("TableEntityExtractor: 개별 처리로 fallback")
                    return self._process_individual_fallback(batch, state, language)
        
        return self._process_individual_fallback(batch, state, language)
    
    def _process_individual_fallback(self, batch: List, state: ParseState, language: str) -> List:
        results = []
        for table_element in batch:
            try:
                batch_data = [{
                    "image": table_element.image_filename,
                    "context": state["texts_by_page"][table_element.page],
                    "language": language,
                }]
                batch_result = table_entity_extractor.invoke(batch_data)
                element = table_element.copy()
                element.entity = batch_result[0] if batch_result else "Entity extraction failed"
                results.append(element)
            except Exception as e:
                if self.verbose:
                    logger.error(f"TableEntityExtractor: 개별 처리도 실패: {str(e)}")
                element = table_element.copy()
                element.entity = "Entity extraction failed"
                results.append(element)
        return results
    
    def _estimate_batch_tokens(self, batch: List, state: ParseState) -> int:
        base_tokens_per_table = 1200
        context_tokens = 0
        
        for element in batch:
            context = state["texts_by_page"].get(element.page, "")
            context_tokens += len(context) // 4
        
        total_tokens = (base_tokens_per_table * len(batch)) + context_tokens
        return total_tokens
    
    def _process_batch(self, batch: List, state: ParseState, language: str) -> List:
        batch_data = []
        for table_element in batch:
            batch_data.append({
                "image": table_element.image_filename,
                "context": state["texts_by_page"][table_element.page],
                "language": language,
            })
        
        batch_result = table_entity_extractor.invoke(batch_data)
        
        results = []
        for j, result in enumerate(batch_result):
            element = batch[j].copy()
            element.entity = result
            results.append(element)
        
        return results
