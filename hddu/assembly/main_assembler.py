import logging
import asyncio
import os
import base64
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from . import config, preparer, postprocessor
from .matcher import TextMatcher
from .merger import LLMMerger, MergedElement, MergedContent
from .config import Element
from ..logging_config import get_logger

logger = get_logger(__name__)
logger.info("[ASSEMBLY_INIT] main_assembler.py logger initialized - testing logging system")

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "0.5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

def _generate_simple_formats(category: str, text: str) -> Tuple[str, str]:
    if category.startswith('heading'):
        level = category[-1] if category[-1].isdigit() else '1'
        markdown = f"{'#' * int(level)} {text}"
        html = f"<h{level}>{text}</h{level}>"
    else:
        markdown = text
        html = f"<p>{text}</p>"
    return markdown, html

def get_page_image_as_base64(image_dir: str, base_filename: str, page_num: int) -> Optional[str]:
    import re
    
    base_pattern = re.match(r'^(.+?)_(\d{4})_\d{4}$', base_filename)
    if base_pattern:
        true_base = base_pattern.group(1)
        start_page = int(base_pattern.group(2))
        actual_page = start_page + page_num
        image_filename = f"{true_base}_page_{actual_page}.png"
        logger.debug(f"[IMAGE_LOAD] Calculated actual page: {start_page} + {page_num} = {actual_page}")
    else:
        image_filename = f"{base_filename}_page_{page_num}.png"
        logger.debug(f"[IMAGE_LOAD] Using standard naming: {image_filename}")
        actual_page = page_num
    
    image_path = os.path.join(image_dir, image_filename)
    logger.info(f"[IMAGE_LOAD] Looking for page image: {image_path}")
    
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                logger.info(f"[IMAGE_LOAD] Page {actual_page}: ✓ loaded (size: {len(image_data)} chars)")
                return image_data
        except Exception as e:
            logger.warning(f"Failed to encode page image {image_path}: {e}")
            logger.warning(f"[IMAGE_LOAD] Page {actual_page}: ✗ encoding failed")
    else:
        logger.warning(f"[IMAGE_LOAD] Page {actual_page}: ✗ file not found")
    return None

class RateLimitHandler:
    
    def __init__(self):
        self.request_times = []
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def wait_if_needed(self):
        await asyncio.sleep(RATE_LIMIT_DELAY)
    
    async def execute_with_retry(self, coroutine_func, *args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                async with self.semaphore:
                    await self.wait_if_needed()
                    return await coroutine_func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str or "tokens per min" in error_str:
                    wait_time = self._extract_wait_time(str(e))
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}/{MAX_RETRIES}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded for rate limit: {e}")
                        raise
                else:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Max retries exceeded: {e}")
                        raise
    
    def _extract_wait_time(self, error_message: str) -> float:
        try:
            import re
            match = re.search(r'try again in (\d+)ms', error_message)
            if match:
                return float(match.group(1)) / 1000.0 + 1.0
            return 60.0
        except:
            return 60.0

class DocumentAssembler:
    def __init__(self):
        logger.info("Initializing DocumentAssembler...")
        
        self.rate_limiter = RateLimitHandler()
        
        if config.TEXT_LLM_PROVIDER.upper() == "AZURE":
            self.text_llm = AzureChatOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY, 
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT, 
                api_version=config.AZURE_OPENAI_API_VERSION, 
                azure_deployment=config.TEXT_AZURE_DEPLOYMENT, 
                temperature=0.1
            )
        elif config.TEXT_LLM_PROVIDER.upper() == "OPENAI":
            self.text_llm = ChatOpenAI(
                api_key=config.OPENAI_API_KEY, 
                model=config.TEXT_OPENAI_MODEL, 
                temperature=0.1
            )
        elif config.TEXT_LLM_PROVIDER.upper() == "OLLAMA":
            self.text_llm = ChatOllama(
                base_url=config.TEXT_OLLAMA_BASE_URL, 
                model=config.TEXT_OLLAMA_MODEL, 
                temperature=0.1
            )
        
        if config.VISION_LLM_PROVIDER.upper() == "AZURE":
            self.vision_llm = AzureChatOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY, 
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT, 
                api_version=config.AZURE_OPENAI_API_VERSION, 
                azure_deployment=config.VISION_AZURE_DEPLOYMENT, 
                temperature=0.1
            )
        elif config.VISION_LLM_PROVIDER.upper() == "OPENAI":
            self.vision_llm = ChatOpenAI(
                api_key=config.OPENAI_API_KEY, 
                model=config.VISION_OPENAI_MODEL, 
                temperature=0.1
            )
        elif config.VISION_LLM_PROVIDER.upper() == "OLLAMA":
            self.vision_llm = ChatOllama(
                base_url=config.VISION_OLLAMA_BASE_URL, 
                model=config.VISION_OLLAMA_MODEL, 
                temperature=0.1
            )
        
        self.embedding_model = config.create_embedding_model()
        
        self.matcher = TextMatcher(self.embedding_model)
        self.merger = LLMMerger(self.text_llm, self.vision_llm)
        
        logger.info("DocumentAssembler initialized successfully.")
        logger.info(f"Configuration Summary:")
        logger.info(f"  ├─ LLM Mode: {config.LLM_BASED_ID_ASSIGNMENT}")
        logger.info(f"  ├─ Assembly Mode: {config.ASSEMBLY_MODE}")
        logger.info(f"  ├─ Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
        logger.info(f"  ├─ Text LLM: {config.TEXT_LLM_PROVIDER} ({config.TEXT_OPENAI_MODEL if config.TEXT_LLM_PROVIDER.upper() == 'OPENAI' else 'N/A'})")
        logger.info(f"  └─ Embedding: {config.EMBEDDING_PROVIDER} ({config.EMBEDDING_OPENAI_MODEL if config.EMBEDDING_PROVIDER.upper() == 'OPENAI' else 'N/A'})")

    async def _execute_batch_with_rate_limiting(self, llm: BaseChatModel, prompts: List[Dict[str, Any]], use_async: bool) -> List[Any]:
        if not prompts:
            return []
        
        messages_list = [p['messages'] for p in prompts]
        
        if use_async and len(messages_list) > 1:
            logger.info(f"Processing {len(prompts)} prompts with rate limiting (max concurrent: {MAX_CONCURRENT_REQUESTS})...")
            
            async def process_single_message(message):
                return await self.rate_limiter.execute_with_retry(llm.ainvoke, message)
            
            results = []
            for message in messages_list:
                result = await self.rate_limiter.execute_with_retry(llm.ainvoke, message)
                results.append(result)
            
            return results
        else:
            logger.info(f"Processing {len(prompts)} prompts synchronously...")
            results = []
            for message in messages_list:
                result = await self.rate_limiter.execute_with_retry(llm.ainvoke, message)
                results.append(result)
            return results

    async def _execute_batch(self, llm: BaseChatModel, prompts: List[Dict[str, Any]], use_async: bool) -> List[Any]:
        return await self._execute_batch_with_rate_limiting(llm, prompts, use_async)

    async def _process(self, docling_path: Optional[str], docyolo_path: Optional[str], 
                     output_path: str, use_async: bool, processing_mode: str):

        logger.info(f"Phase 1: Preparing data for {processing_mode} mode...")
        
        if processing_mode == "merge":
            pages_data = preparer.load_and_group_by_page(docling_path, docyolo_path)
        elif processing_mode == "docling_only":
            pages_data = self._load_single_parser_data(docling_path, "docling")
        elif processing_mode == "docyolo_only":
            pages_data = self._load_single_parser_data(docyolo_path, "docyolo")
        else:
            raise ValueError(f"Unsupported processing mode: {processing_mode}")
        
        all_final_elements = []

        image_dir, base_filename = self._determine_image_directory(docling_path, docyolo_path, processing_mode)
        logger.info(f"Image directory: {image_dir}")

        from . import postprocessor
        logger.info("Standardizing page images before visual processing...")
        postprocessor.create_standard_page_images(image_dir, base_filename)

        for page_num in sorted(pages_data.keys()):
            logger.info(f"--- Starting Page {page_num} ({processing_mode} mode) ---")
            page_content = pages_data[page_num]
            page_image_b64 = get_page_image_as_base64(image_dir, base_filename, page_num)
            
            logger.info(f"Phase 2: Classifying elements for page {page_num}...")
            page_elements = await self._process_page_by_mode(page_content, processing_mode, page_image_b64, use_async, page_num)
            
            all_final_elements.extend(page_elements)
            logger.info(f"Page {page_num} completed: {len(page_elements)} elements")
            logger.info(f"--- Finished Page {page_num} ---")

        logger.info(f"Phase 4: Post-processing all elements... ({len(all_final_elements)} total elements)")
        
        llm_mode = config.LLM_BASED_ID_ASSIGNMENT.lower()
        logger.info(f"LLM Mode Decision: {llm_mode.upper()}")
        logger.info(f"Processing Mode: {processing_mode}")
        
        if processing_mode == "merge" and llm_mode in ["simple", "advanced"]:
            logger.info(f"ACTIVATED: LLM-based ID assignment with deduplication")
            logger.info(f"  ├─ Deduplication: ENABLED")
            logger.info(f"  ├─ LLM Processing: ENABLED ({llm_mode})")
            logger.info(f"  └─ Sort Mode: integrated")
            
            logger.info(f"Starting deduplication process...")
            final_unique_elements = self._deduplicate_elements(all_final_elements)
            
            logger.info(f"Starting LLM processing...")
            processed_elements = await postprocessor.finalize_elements_with_llm(
                final_unique_elements, self.vision_llm, image_dir, base_filename, use_async, llm_mode
            )
        else:
            logger.info(f"ACTIVATED: Coordinate-based processing without deduplication")
            logger.info(f"  ├─ Deduplication: DISABLED")
            logger.info(f"  ├─ LLM Processing: DISABLED")
            logger.info(f"  └─ Sort Mode: disabled (parser grouping)")
            
            logger.info(f"Starting coordinate-based sorting...")
            processed_elements = postprocessor.finalize_elements(all_final_elements, sort_mode="disabled")
        
        logger.info(f"Generating unified content from {len(processed_elements)} elements...")
        unified_content = postprocessor.generate_unified_content(processed_elements)
        
        reference_file = docling_path or docyolo_path
        final_data = preparer.create_final_structure(reference_file, processed_elements, unified_content)
        postprocessor.save_final_json(final_data, output_path)
        
        logger.info(f"Document assembly completed successfully!")
        logger.info(f"  ├─ Processing Mode: {processing_mode}")
        logger.info(f"  ├─ LLM Mode: {llm_mode.upper()}")
        logger.info(f"  ├─ Final Elements: {len(processed_elements)}")
        logger.info(f"  ├─ Input Elements: {len(all_final_elements)}")
        if processing_mode == "merge" and llm_mode in ["simple", "advanced"]:
            reduction = len(all_final_elements) - len(processed_elements)
            reduction_rate = (reduction / len(all_final_elements)) * 100 if all_final_elements else 0
            logger.info(f"  ├─ Elements Reduced: {reduction} ({reduction_rate:.1f}%)")
        logger.info(f"  └─ Output File: {output_path}")

    def run(self, docling_path: Optional[str], docyolo_path: Optional[str], output_path: str, use_async: bool = False):

        processing_mode = self._determine_processing_mode(docling_path, docyolo_path)
        
        logger.info(f"Starting assembly process (mode={processing_mode}, async_mode={use_async})...")
        logger.info(f"Docling file: {docling_path or 'None'}")
        logger.info(f"DocYOLO file: {docyolo_path or 'None'}")
        
        try:
            asyncio.run(self._process(docling_path, docyolo_path, output_path, use_async, processing_mode))
        except Exception as e:
            logger.critical(f"An unhandled error occurred in the async process: {e}", exc_info=True)
            raise
    

    def _determine_processing_mode(self, docling_path: Optional[str], docyolo_path: Optional[str]) -> str:

        if docling_path is None and docyolo_path is not None:
            logger.info("DocYOLO file only provided - using docyolo_only mode")
            return "docyolo_only"
        elif docling_path is not None and docyolo_path is None:
            logger.info("Docling file only provided - using docling_only mode")
            return "docling_only"
        elif docling_path is not None and docyolo_path is not None:
            if config.ASSEMBLY_MODE == "docling_only":
                logger.info("Both files provided but config set to docling_only - ignoring DocYOLO file")
                return "docling_only"
            elif config.ASSEMBLY_MODE == "docyolo_only":
                logger.info("Both files provided but config set to docyolo_only - ignoring Docling file")
                return "docyolo_only"
            else:
                logger.info("Both files provided and config allows merging - using merge mode")
                return "merge"
        else:
            raise ValueError("At least one input file (docling_path or docyolo_path) must be provided")

    def _load_single_parser_data(self, file_path: str, parser_type: str) -> Dict[int, Dict[str, List[Element]]]:

        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_data = preparer._remove_single_noise_and_reindex(data, parser_type)
        elements = cleaned_data.get('elements', [])
        
        pages_data = {}
        
        for element in elements:
            page_num = element.get('page', 0)
            if page_num not in pages_data:
                pages_data[page_num] = {parser_type: []}
            pages_data[page_num][parser_type].append(element)
        
        logger.info(f"Loaded {len(elements)} elements from {parser_type} file across {len(pages_data)} pages")
        return pages_data

    def _determine_image_directory(self, docling_path: Optional[str], docyolo_path: Optional[str], 
                                 processing_mode: str) -> Tuple[str, str]:

        if processing_mode == "docling_only" or (processing_mode == "merge" and docling_path):
            docling_dir = os.path.dirname(docling_path)
            base_filename = os.path.basename(docling_path).replace("_docling.json", "")
            image_dir = os.path.join(os.path.dirname(docling_dir), "docling_output")
        elif processing_mode == "docyolo_only":
            docyolo_dir = os.path.dirname(docyolo_path)
            base_filename = os.path.basename(docyolo_path).replace("_docyolo.json", "")
            image_dir = os.path.join(os.path.dirname(docyolo_dir), "docyolo_output")
        else:
            if docling_path:
                docling_dir = os.path.dirname(docling_path)
                base_filename = os.path.basename(docling_path).replace("_docling.json", "")
                image_dir = os.path.join(os.path.dirname(docling_dir), "docling_output")
            else:
                docyolo_dir = os.path.dirname(docyolo_path)
                base_filename = os.path.basename(docyolo_path).replace("_docyolo.json", "")
                image_dir = os.path.join(os.path.dirname(docyolo_dir), "docyolo_output")
        
        return image_dir, base_filename

    async def _process_page_by_mode(self, page_content: Dict[str, List[Element]], processing_mode: str,
                                   page_image_b64: Optional[str], use_async: bool, page_num: int) -> List[Element]:

        if processing_mode == "merge":
            return await self._process_merged_page(page_content, page_image_b64, use_async, page_num)
        elif processing_mode == "docling_only":
            docling_elements = page_content.get('docling', [])
            return await self._process_single_parser_page(docling_elements, "docling", page_image_b64, use_async, page_num)
        elif processing_mode == "docyolo_only":
            docyolo_elements = page_content.get('docyolo', [])
            return await self._process_single_parser_page(docyolo_elements, "docyolo", page_image_b64, use_async, page_num)
        else:
            raise ValueError(f"Unsupported processing mode: {processing_mode}")

    async def _process_merged_page(self, page_content: Dict[str, List[Element]], page_image_b64: Optional[str],
                                  use_async: bool, page_num: int) -> List[Element]:

        d_texts, d_tables, d_figures = preparer.classify_elements(page_content.get('docling', []))
        y_texts, y_tables, y_figures = preparer.classify_elements(page_content.get('docyolo', []))
        
        logger.info(f"Docling: {len(d_texts)} texts, {len(d_tables)} tables, {len(d_figures)} figures")
        logger.info(f"DocYOLO: {len(y_texts)} texts, {len(y_tables)} tables, {len(y_figures)} figures")

        logger.info(f"Phase 3A: Processing texts for page {page_num}...")
        text_elements = await self._process_texts(d_texts, y_texts, use_async)
        
        all_visual_elements = d_tables + d_figures + y_tables + y_figures
        visual_elements = []
        if all_visual_elements:
            logger.info(f"Phase 3B: Processing {len(all_visual_elements)} visual elements (tables + figures) individually for page {page_num}...")
            visual_elements = await self._process_visual_elements_unified(all_visual_elements, page_image_b64, text_elements, use_async)
        
        all_processed_elements = text_elements + visual_elements
        logger.info(f"Page {page_num} processing complete: {len(text_elements)} texts, {len(visual_elements)} visual elements")
        
        return all_processed_elements

    async def _process_single_parser_page(self, elements: List[Element], parser_type: str, 
                                        page_image_b64: Optional[str], use_async: bool, page_num: int) -> List[Element]:

        texts, tables, figures = preparer.classify_elements(elements)
        
        logger.info(f"{parser_type.title()}: {len(texts)} texts, {len(tables)} tables, {len(figures)} figures")
        
        processed_elements = []
        
        logger.info(f"Processing {len(texts)} text elements...")
        for text_elem in texts:
            if 'content' not in text_elem:
                text_content = text_elem.get('text', '')
                category = text_elem.get('category', 'paragraph')
                md, html = _generate_simple_formats(category, text_content)
                text_elem['content'] = {'text': text_content, 'markdown': md, 'html': html}
            text_elem['source_parser'] = parser_type
            text_elem['processing_status'] = 'single_parser'
            processed_elements.append(text_elem)
        
        all_visual_elements = tables + figures
        logger.info(f"Processing {len(all_visual_elements)} visual elements individually...")
        if all_visual_elements:
            enhanced_visual_elements = await self._process_single_parser_visual_elements(
                all_visual_elements, parser_type, page_image_b64, use_async
            )
            processed_elements.extend(enhanced_visual_elements)
        
        return processed_elements

    async def _process_visual_elements_unified(self, all_visual_elements: List[Element], 
                                             page_image_b64: Optional[str], text_elements: List[Element], 
                                             use_async: bool) -> List[Element]:

        if not all_visual_elements:
            return []
            
        processed_elements = []
        
        surrounding_texts = self._collect_surrounding_context_by_index(all_visual_elements, text_elements)
        
        if use_async and len(all_visual_elements) > 1:
            logger.info(f"Processing {len(all_visual_elements)} visual elements in parallel (individual processing)...")
            
            async def process_single_visual_element(element, index):
                try:
                    task_id = f"visual_element_{index}"
                    context_text = surrounding_texts.get(index, "")
                    enhanced_element = await self._enhance_visual_element(element, page_image_b64, context_text, task_id)
                    return enhanced_element
                except Exception as e:
                    logger.error(f"Failed to process visual element {index}: {e}")
                    fallback_element = element.copy()
                    fallback_element['processing_status'] = 'failed'
                    fallback_element['error_message'] = str(e)
                    return fallback_element
            
            tasks = [process_single_visual_element(elem, i) for i, elem in enumerate(all_visual_elements)]
            processed_elements = await asyncio.gather(*tasks, return_exceptions=True)
            
            final_elements = []
            for result in processed_elements:
                if isinstance(result, Exception):
                    logger.error(f"Visual element processing failed with exception: {result}")
                else:
                    final_elements.append(result)
            processed_elements = final_elements
            
        else:
            logger.info(f"Processing {len(all_visual_elements)} visual elements sequentially (individual processing)...")
            
            for i, element in enumerate(all_visual_elements):
                try:
                    task_id = f"visual_element_{i}"
                    context_text = surrounding_texts.get(i, "")
                    enhanced_element = await self._enhance_visual_element(element, page_image_b64, context_text, task_id)
                    processed_elements.append(enhanced_element)
                except Exception as e:
                    logger.error(f"Failed to process visual element {i}: {e}")
                    fallback_element = element.copy()
                    fallback_element['processing_status'] = 'failed'
                    fallback_element['error_message'] = str(e)
                    processed_elements.append(fallback_element)
        
        enhanced_count = len([elem for elem in processed_elements if elem.get('processing_status') == 'enhanced'])
        failed_count = len([elem for elem in processed_elements if elem.get('processing_status') == 'failed'])
        
        logger.info(f"Individual visual processing complete: {enhanced_count} enhanced, {failed_count} failed")
        return processed_elements

    async def _process_single_parser_visual_elements(self, figures: List[Element], parser_type: str,
                                                   page_image_b64: Optional[str], use_async: bool) -> List[Element]:

        return await self._process_visual_elements_unified(figures, page_image_b64, [], use_async)

    async def _process_texts(self, d_texts: List[Element], y_texts: List[Element], use_async: bool) -> List[Element]:

        if not d_texts and not y_texts:
            return []
        
        matched_texts, d_texts_only, y_texts_only = self.matcher.match_text_elements(d_texts, y_texts)
        text_prompts_with_meta = self.merger._create_text_merge_prompts(matched_texts, d_texts_only, y_texts_only, d_texts, y_texts)
        
        if not text_prompts_with_meta:
            return []
        
        llm_text_results = await self._execute_batch(self.merger.text_llm_structured, text_prompts_with_meta, use_async)
        processed_texts = []
        
        for i, res in enumerate(llm_text_results):
            original_element = text_prompts_with_meta[i]['original_element']
            if res and isinstance(res, MergedElement) and res.is_valid:
                new_elem = original_element.copy()
                new_elem['category'] = res.category
                if 'content' not in new_elem:
                    new_elem['content'] = {}
                md, html = _generate_simple_formats(res.category, res.text)
                new_elem['content'] = {'text': res.text, 'markdown': md, 'html': html}
                
                if original_element.get('base64_encoding'):
                    new_elem['base64_encoding'] = original_element['base64_encoding']
                if original_element.get('image_path'):
                    new_elem['image_path'] = original_element['image_path']
                
                if i < len(matched_texts):
                    d_elem, y_elem = matched_texts[i]
                    docyolo_text = str(y_elem.get('content', {}).get('text', ''))
                    if docyolo_text and res.text.strip() == docyolo_text.strip():
                        new_elem['source_parser'] = 'docyolo'
                
                processed_texts.append(new_elem)
            elif isinstance(original_element, dict):
                processed_texts.append(original_element)
        
        return processed_texts

    def _collect_surrounding_context_by_index(self, visual_elements: List[Element], text_elements: List[Element]) -> Dict[int, str]:

        surrounding_texts = {}
        
        all_elements = []
        
        for text_elem in text_elements:
            all_elements.append({
                'element': text_elem,
                'type': 'text',
                'y_coord': self._extract_y_coordinate(text_elem),
                'visual_index': None
            })
        
        for i, visual_elem in enumerate(visual_elements):
            all_elements.append({
                'element': visual_elem,
                'type': 'visual',
                'y_coord': self._extract_y_coordinate(visual_elem),
                'visual_index': i
            })
        
        all_elements.sort(key=lambda x: x['y_coord'])
        
        for i, elem_info in enumerate(all_elements):
            if elem_info['type'] == 'visual':
                visual_index = elem_info['visual_index']
                context_parts = []
                
                for j in range(max(0, i-2), i):
                    if all_elements[j]['type'] == 'text':
                        text_content = all_elements[j]['element'].get('content', {}).get('text', '')
                        if text_content:
                            context_parts.append(f"[이전] {text_content}")
                
                for j in range(i+1, min(len(all_elements), i+3)):
                    if all_elements[j]['type'] == 'text':
                        text_content = all_elements[j]['element'].get('content', {}).get('text', '')
                        if text_content:
                            context_parts.append(f"[이후] {text_content}")
                
                surrounding_texts[visual_index] = " ".join(context_parts)
        
        return surrounding_texts



    def _extract_y_coordinate(self, element: Element) -> float:
        coords = element.get('coordinates', [])
        
        if coords and isinstance(coords, list) and len(coords) > 0:
            first_coord = coords[0]
            if isinstance(first_coord, dict) and 'y' in first_coord:
                return float(first_coord['y'])
            elif isinstance(first_coord, (int, float)) and len(coords) > 1:
                return float(coords[1])
        
        return float(element.get('id', 0)) * 1000

    async def _enhance_visual_element(self, element: Element, page_context: Optional[str], 
                                    surrounding_text: str, task_id: str) -> Element:
            
        if not page_context:
            logger.debug(f"[VISUAL] {task_id}: No page context available")
        
        base64_data = element.get('base64_encoding')
        category = element.get('category', 'figure')
        existing_content = element.get('content', {})
        
        if not base64_data:
            if category == 'table' and existing_content:
                logger.info(f"Processing table without base64 encoding using existing content: {task_id}")
                return await self._enhance_text_based_table(element, page_context, surrounding_text, task_id)
            else:
                logger.warning(f"No base64 encoding for visual element {task_id} - skipping enhancement")
                enhanced_element = element.copy()
                enhanced_element['processing_status'] = 'no_image_data'
                return enhanced_element
        
        source_parser = element.get('source_parser', 'unknown')
        
        prompt = self._create_unified_visual_prompt(element, page_context, surrounding_text, category)
        
        try:
            enhanced_content = await self.merger.vision_llm_content.ainvoke(prompt['messages'])
            
            enhanced_element = element.copy()
            
            if element.get('base64_encoding'):
                enhanced_element['base64_encoding'] = element.get('base64_encoding')
            
            if element.get('has_caption'):
                enhanced_element['caption'] = element.get('caption', '')
                enhanced_element['has_caption'] = True
            
            if enhanced_content and hasattr(enhanced_content, 'dict'):
                content_dict = enhanced_content.dict()
                
                original_markdown = content_dict.get('markdown', '')
                original_html = content_dict.get('html', '')
                
                comprehensive_text = content_dict.get('text', '')
                if not comprehensive_text:
                    comprehensive_text = str(enhanced_content)
                
                content_result = {
                    'text': comprehensive_text,
                    'markdown': comprehensive_text,
                    'html': comprehensive_text
                }
                
                if category == 'table':
                    content_result['raw_output'] = original_markdown
                
                enhanced_element['content'] = content_result
            elif enhanced_content and isinstance(enhanced_content, dict):
                comprehensive_text = enhanced_content.get('text', str(enhanced_content))
                original_markdown = enhanced_content.get('markdown', '')
                
                content_result = {
                    'text': comprehensive_text,
                    'markdown': comprehensive_text,
                    'html': comprehensive_text
                }
                
                if category == 'table':
                    content_result['raw_output'] = original_markdown
                
                enhanced_element['content'] = content_result
            else:
                comprehensive_text = str(enhanced_content) if enhanced_content else ''
                content_result = {
                    'text': comprehensive_text,
                    'markdown': comprehensive_text,
                    'html': comprehensive_text
                }
                
                if category == 'table':
                    content_result['raw_output'] = ''
                
                enhanced_element['content'] = content_result
            
            enhanced_element['processing_status'] = 'enhanced'
            enhanced_element['source_parser'] = source_parser
            enhanced_element['task_id'] = task_id
            
            return enhanced_element
            
        except Exception as e:
            logger.error(f"Failed to enhance visual element {task_id}: {e}")
            
            enhanced_element = element.copy()
            if element.get('base64_encoding'):
                enhanced_element['base64_encoding'] = element.get('base64_encoding')
            
            enhanced_element['processing_status'] = 'failed'
            enhanced_element['error_message'] = str(e)
            enhanced_element['source_parser'] = source_parser
            enhanced_element['task_id'] = task_id
            
            return enhanced_element

    async def _enhance_text_based_table(self, element: Element, page_context: Optional[str], 
                                      surrounding_text: str, task_id: str) -> Element:

        category = element.get('category', 'table')
        source_parser = element.get('source_parser', 'unknown')
        existing_content = element.get('content', {})
        
        existing_text = existing_content.get('text', '')
        existing_markdown = existing_content.get('markdown', '')
        existing_html = existing_content.get('html', '')
        
        prompt = self._create_text_based_visual_prompt(
            element, page_context, surrounding_text, category, 
            existing_text, existing_markdown, existing_html
        )
        
        try:
            enhanced_content = await self.merger.vision_llm_content.ainvoke(prompt['messages'])
            
            enhanced_element = element.copy()
            
            if element.get('has_caption'):
                enhanced_element['caption'] = element.get('caption', '')
                enhanced_element['has_caption'] = True
            
            if enhanced_content and hasattr(enhanced_content, 'dict'):
                content_dict = enhanced_content.dict()
                
                original_markdown = content_dict.get('markdown', existing_markdown)
                original_html = content_dict.get('html', existing_html)
                
                comprehensive_text = content_dict.get('text', '')
                if not comprehensive_text:
                    comprehensive_text = str(enhanced_content)
                
                content_result = {
                    'text': comprehensive_text,
                    'markdown': comprehensive_text,
                    'html': comprehensive_text
                }
                
                if category == 'table':
                    content_result['raw_output'] = original_markdown if original_markdown else existing_markdown
                
                enhanced_element['content'] = content_result
            elif enhanced_content and isinstance(enhanced_content, dict):
                comprehensive_text = enhanced_content.get('text', str(enhanced_content))
                original_markdown = enhanced_content.get('markdown', existing_markdown)
                
                content_result = {
                    'text': comprehensive_text,
                    'markdown': comprehensive_text,
                    'html': comprehensive_text
                }
                
                if category == 'table':
                    content_result['raw_output'] = original_markdown if original_markdown else existing_markdown
                
                enhanced_element['content'] = content_result
            else:
                content_result = {
                    'text': existing_text or existing_markdown,
                    'markdown': existing_text or existing_markdown,
                    'html': existing_text or existing_markdown
                }
                
                if category == 'table':
                    content_result['raw_output'] = existing_markdown
                
                enhanced_element['content'] = content_result
            
            enhanced_element['processing_status'] = 'enhanced_text_based'
            enhanced_element['source_parser'] = source_parser
            enhanced_element['task_id'] = task_id
            
            logger.info(f"Successfully enhanced text-based table: {task_id}")
            return enhanced_element
            
        except Exception as e:
            logger.error(f"Error enhancing text-based table {task_id}: {e}", exc_info=True)
            
            enhanced_element = element.copy()
            
            existing_content = element.get('content', {})
            content_result = {
                'text': existing_content.get('text', existing_content.get('markdown', '')),
                'markdown': existing_content.get('text', existing_content.get('markdown', '')),
                'html': existing_content.get('text', existing_content.get('markdown', ''))
            }
            
            if category == 'table':
                content_result['raw_output'] = existing_content.get('markdown', '')
            
            enhanced_element['content'] = content_result
            enhanced_element['processing_status'] = 'text_based_fallback'
            enhanced_element['error_message'] = str(e)
            enhanced_element['source_parser'] = source_parser
            enhanced_element['task_id'] = task_id
            
            return enhanced_element

    def _create_unified_visual_prompt(self, element: Element, page_context: Optional[str], 
                                    surrounding_text: str, category: str) -> Dict[str, Any]:
        
        if category == 'table':
            analysis_instructions = """
**핵심 목표**: 사용자가 실제 테이블을 보지 않고도 이 해석 결과만 읽고 테이블의 모든 정보를 완전히 이해할 수 있도록 상세하게 분석하세요.

**포괄적 테이블 분석 요구사항**:

1. **테이블의 목적과 의미 분석**:
   - 페이지 이미지와 주변 컨텍스트를 참고하여 이 테이블이 왜 필요한지, 무엇을 나타내는지 설명
   - 문서 전체에서 이 테이블이 차지하는 역할과 위치 파악
   - 테이블 제목, 캡션, 주변 텍스트와의 연관성 분석

2. **구조 상세 분석**:
   - 행과 열의 정확한 개수와 구조 설명
   - 헤더 정보와 각 컬럼의 의미 상세 분석
   - 병합된 셀(rowspan/colspan)이 있다면 그 의미와 구조 설명
   - 전체 테이블 구성과 레이아웃 분석

3. **각 셀의 내용 상세 설명**:
   - 테이블 이미지를 보고 각 셀의 내용을 하나하나 읽어서 설명
   - 각 셀이 나타내는 데이터의 의미와 중요성 분석
   - 빈 셀이나 특수 기호가 있다면 그 의미 해석
   - 수치 데이터의 경우 단위, 포맷, 정확한 값 명시

4. **데이터 패턴 및 관계 분석**:
   - 데이터 간의 연관성, 패턴, 트렌드 분석
   - 행 간, 열 간 데이터의 상관관계 파악
   - 증가/감소 패턴, 주기적 변화, 특이사항 식별

**테이블 형태 로드맵 및 프로세스 특별 분석** (해당하는 경우):
   **※ 테이블이 로드맵, 프로세스, 워크플로우, 일정표, 단계별 진행도를 나타내는 경우 아래 추가 분석 수행**
   
   - **테이블 프로세스 유형 식별**: 
     * 시간축 기반 일정표 (Timeline Table)
     * 단계별 체크리스트 테이블
     * 간트 차트 형태의 프로젝트 진행도
     * 워크플로우 매트릭스
     * 승인/검토 프로세스 테이블
     * 의사결정 매트릭스
   - **시간축 및 단계축 식별**: 
     * 시간을 나타내는 행 또는 열 식별 (날짜, 주차, 월, 분기 등)
     * 단계나 순서를 나타내는 행 또는 열 식별
     * 우선순위나 중요도를 나타내는 축 파악
   - **각 단계/시점별 상세 분석**:
     * 각 셀에서 해당 시점/단계의 구체적인 활동이나 작업 설명
     * 각 단계의 담당자, 책임자, 관련 부서 정보
     * 각 단계별 목표, 산출물, 완료 기준 분석
     * 예상 소요 시간, 리소스, 비용 정보 (명시된 경우)
     * 각 단계의 상태나 진행도 (완료/진행중/예정 등)
   - **테이블 내 연결관계 및 의존성 분석**:
     * 행 간 연결관계 (순차적 진행, 병렬 수행, 조건부 분기)
     * 열 간 연결관계 (부서별, 역할별, 기능별 연관성)
     * 선행 작업과 후행 작업의 의존성 관계
     * 동시 진행 가능한 작업들과 순차 진행 필요한 작업들 구분
   - **마일스톤 및 중요 시점 식별**:
     * 프로젝트나 프로세스의 핵심 전환점 식별
     * 검토, 승인, 결정이 필요한 지점
     * 외부 이해관계자 참여 시점
     * 위험 요소나 주의 사항이 표시된 구간
   - **테이블 기반 프로세스 흐름 분석**:
     * 테이블 셀 간의 논리적 흐름과 순서 설명
     * 조건부 경로나 대안 프로세스 (있는 경우)
     * 반복 구간이나 피드백 루프 (있는 경우)
     * 예외 상황이나 에스컬레이션 프로세스

5. **수치 데이터 분석**:
   - 최댓값, 최솟값, 평균값 등 통계적 특성 분석
   - 수치의 변화 추세와 그 의미 해석
   - 백분율, 비율, 증감률 등 계산된 값의 의미

6. **범주형 데이터 분석**:
   - 카테고리별 분포와 특성 분석
   - 각 범주의 의미와 구분 기준 설명
   - 범주 간 비교와 차이점 분석

7. **중요한 정보 강조**:
   - 핵심 메시지나 주목할 만한 데이터 포인트 강조
   - 특별한 의미를 가지는 값이나 항목 식별
   - 의사결정에 중요한 정보 부각

8. **누락된 정보 파악**:
   - 빈 셀이나 누락된 데이터의 의미 해석
   - 데이터 수집의 한계나 제약사항 추정
   - 완전하지 않은 정보에 대한 추가 설명

9. **한국어 맥락 해석**:
   - 한국어 특성을 고려한 내용 해석
   - 문화적, 사회적 맥락에서의 데이터 의미 분석
   - 한국의 제도나 관습과 연관된 내용 설명

10. **실용적 활용 방안**:
    - 이 테이블 정보를 어떻게 활용할 수 있는지 제안
    - 후속 분석이나 의사결정에 도움이 되는 통찰 제공
    - 테이블이 전달하고자 하는 핵심 메시지 도출

**중요**: 각 분석 항목을 체계적으로 다루되, 단순 나열이 아닌 의미 있는 해석과 통찰을 제공하세요.
사용자가 이 해석만 읽고도 원본 테이블의 완전한 이해가 가능하도록 작성하세요.
            """
        else:
            analysis_instructions = """
**핵심 목표**: 사용자가 실제 이미지를 보지 않고도 이 해석 결과만 읽고 이미지의 모든 정보를 완전히 이해할 수 있도록 상세하게 분석하세요.

**포괄적 이미지 분석 요구사항**:

1. **이미지의 목적과 의미 분석**:
   - 페이지 이미지와 주변 컨텍스트를 참고하여 이 이미지가 왜 필요한지, 무엇을 나타내는지 설명
   - 문서 전체에서 이 이미지가 차지하는 역할과 위치 파악
   - 이미지 캡션, 제목, 주변 텍스트와의 연관성 분석

2. **시각적 구조 상세 분석**:
   - 이미지의 전체 구성과 레이아웃 설명
   - 차트의 경우: 차트 유형, 축, 범례, 제목 등 모든 요소 분석
   - 다이어그램의 경우: 구조, 흐름, 관계도 등 상세 설명
   - 사진의 경우: 주요 피사체, 배경, 구성 요소 분석

3. **모든 텍스트 내용 추출 및 분석**:
   - 이미지 내의 모든 텍스트를 정확히 OCR 추출
   - 각 텍스트의 위치, 크기, 강조 정도 분석
   - 텍스트가 전달하는 메시지와 의미 해석
   - 제목, 레이블, 수치, 설명 등 모든 문자 정보 상세 분석

4. **데이터 및 수치 분석**:
   - 차트나 그래프의 모든 데이터 포인트 분석
   - 수치의 변화 추세, 패턴, 특이사항 식별
   - 최댓값, 최솟값, 평균값 등 통계적 특성 분석
   - 비율, 증감률, 상관관계 등 데이터 간 관계 파악

5. **색상과 시각적 표현 분석**:
   - 색상 코딩의 의미와 구분 기준 설명
   - 시각적 강조 요소(굵기, 크기, 위치)의 의미 분석
   - 아이콘, 기호, 패턴 등 시각적 요소의 의미 해석

6. **관계와 흐름 분석**:
   - 요소 간의 연결관계, 흐름도, 프로세스 분석
   - 화살표, 선, 연결선의 의미와 방향성 설명
   - 시간적 순서나 인과관계 파악

**로드맵 및 프로세스 다이어그램 특별 분석** (해당하는 경우):
   **※ 이미지가 로드맵, 프로세스, 워크플로우, 단계별 진행도를 나타내는 경우 아래 추가 분석 수행**
   
   - **프로세스 유형 식별**: 순차적 프로세스, 병렬 프로세스, 조건부 분기, 반복 루프 등 구분
   - **시작점과 종료점**: 프로세스의 명확한 시작과 끝 지점 식별
   - **각 단계별 상세 분석**:
     * 각 단계의 번호, 이름, 위치 명시
     * 각 단계에서 수행되는 구체적인 작업이나 활동 설명
     * 각 단계의 목적과 중요성 분석
     * 각 단계에서 필요한 입력과 예상되는 출력 설명
     * 각 단계의 소요 시간이나 기간 (명시된 경우)
   - **단계 간 연결관계 심화 분석**:
     * 각 단계 간 연결의 조건과 기준 설명
     * 병렬로 수행되는 단계들의 관계 분석
     * 조건부 분기 지점에서의 판단 기준과 경로 설명
     * 되돌아가는 루프나 반복 구조의 조건 분석
   - **마일스톤 및 체크포인트**: 중요한 결정점이나 검증 지점 식별
   - **의존성 관계**: 특정 단계가 다른 단계의 완료에 의존하는 관계 분석
   - **리스크 및 주의사항**: 각 단계에서 발생할 수 있는 문제점이나 주의사항
   - **대안 경로**: 메인 프로세스 외의 대안적 경로나 예외 처리 방법

7. **중요한 정보 강조**:
   - 핵심 메시지나 주목할 만한 시각적 포인트 강조
   - 특별한 의미를 가지는 요소나 데이터 식별
   - 의사결정에 중요한 정보 부각

8. **한국어 맥락 해석**:
   - 한국어 텍스트의 의미와 맥락 분석
   - 문화적, 사회적 맥락에서의 이미지 의미 해석
   - 한국의 제도나 관습과 연관된 내용 설명

9. **실용적 활용 방안**:
   - 이 이미지 정보를 어떻게 활용할 수 있는지 제안
   - 후속 분석이나 의사결정에 도움이 되는 통찰 제공
   - 이미지가 전달하고자 하는 핵심 메시지 도출

**중요**: 각 분석 항목을 체계적으로 다루되, 단순 나열이 아닌 의미 있는 해석과 통찰을 제공하세요.
사용자가 이 해석만 읽고도 원본 이미지의 완전한 이해가 가능하도록 작성하세요.
            """
        
        content = [
            {"type": "text", "text": f"""
**COMPREHENSIVE VISUAL ANALYSIS TASK**

**ELEMENT TYPE**: {category}
**SURROUNDING CONTEXT**: {surrounding_text}

{analysis_instructions}

**OUTPUT REQUIREMENTS**:
- **Text**: 위 분석 요구사항에 따른 포괄적이고 상세한 해석 결과
- **Markdown**: 
  - 테이블의 경우: 실제 마크다운 테이블 구조 (| 형태)
  - 이미지의 경우: 구조화된 마크다운 형식의 설명
- **HTML**: 
  - 테이블의 경우: 완전한 HTML 테이블 구조 (<table>...</table>)
  - 이미지의 경우: 깔끔한 HTML 표현

**중요**: 테이블의 경우 Markdown과 HTML은 실제 구조적 표현을, Text는 포괄적 해석을 제공하세요.

**핵심 원칙**:
1. **완전한 이해**: 사용자가 원본을 보지 않고도 이 해석만으로 모든 내용 이해 가능
2. **상세한 분석**: 각 요소의 내용을 하나하나 자세히 설명
3. **맥락적 해석**: 페이지 이미지와 주변 컨텍스트를 적극 활용
4. **정확성**: 모든 데이터와 정보를 정확히 추출하고 분석
5. **한국어 맥락**: 한국어 특성을 고려한 해석 및 의미 부여

**언어**: 한국어로 작성하되, 전문 용어는 적절히 설명과 함께 사용
"""}
        ]
        
        if element.get('base64_encoding'):
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{element.get('base64_encoding')}"}
            })
            content.append({"type": "text", "text": "↑ **분석 대상 이미지**: 위 이미지를 기반으로 상세한 해석을 수행하세요."})
        
        if page_context:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{page_context}"}
            })
            content.append({"type": "text", "text": "↑ **페이지 컨텍스트**: 위 페이지 전체 이미지를 참고하여 이 요소의 문서 내 위치와 맥락을 파악하세요."})
        
        return {"messages": [HumanMessage(content=content)]}

    def _create_text_based_visual_prompt(self, element: Element, page_context: Optional[str], 
                                       surrounding_text: str, category: str,
                                       existing_text: str, existing_markdown: str, 
                                       existing_html: str) -> Dict[str, Any]:

        
        content_info = []
        if existing_text:
            content_info.append(f"**기존 텍스트**: {existing_text}")
        if existing_markdown:
            content_info.append(f"**기존 마크다운**: {existing_markdown}")
        if existing_html:
            content_info.append(f"**기존 HTML**: {existing_html}")
        
        content_summary = "\n".join(content_info) if content_info else "내용 정보 없음"
        
        analysis_instructions = """
**핵심 목표**: 제공된 텍스트 기반 테이블 정보를 분석하여, 사용자가 실제 테이블을 보지 않고도 이 해석 결과만 읽고 테이블의 모든 정보를 완전히 이해할 수 있도록 상세하게 분석하세요.

**텍스트 기반 포괄적 테이블 분석 요구사항**:

1. **테이블의 목적과 의미 분석**:
   - 주변 컨텍스트를 참고하여 이 테이블이 왜 필요한지, 무엇을 나타내는지 설명
   - 문서 전체에서 이 테이블이 차지하는 역할과 위치 파악
   - 테이블 제목, 캡션, 주변 텍스트와의 연관성 분석

2. **구조 상세 분석**:
   - 제공된 마크다운/HTML 구조를 분석하여 행과 열의 정확한 개수와 구조 설명
   - 헤더 정보와 각 컬럼의 의미 상세 분석
   - 병합된 셀(rowspan/colspan)이 있다면 그 의미와 구조 설명
   - 전체 테이블 구성과 레이아웃 분석

3. **각 셀의 내용 상세 설명**:
   - 텍스트 정보를 기반으로 각 셀의 내용을 하나하나 읽어서 설명
   - 각 셀이 나타내는 데이터의 의미와 중요성 분석
   - 빈 셀이나 특수 기호가 있다면 그 의미 해석
   - 수치 데이터의 경우 단위, 포맷, 정확한 값 명시

4. **데이터 패턴 및 관계 분석**:
   - 데이터 간의 연관성, 패턴, 트렌드 분석
   - 행 간, 열 간 데이터의 상관관계 파악
   - 증가/감소 패턴, 주기적 변화, 특이사항 식별

**테이블 형태 로드맵 및 프로세스 특별 분석** (해당하는 경우):
   **※ 테이블이 로드맵, 프로세스, 워크플로우, 일정표, 단계별 진행도를 나타내는 경우 아래 추가 분석 수행**
   
   - **테이블 프로세스 유형 식별**: 
     * 시간축 기반 일정표 (Timeline Table)
     * 단계별 체크리스트 테이블
     * 간트 차트 형태의 프로젝트 진행도
     * 워크플로우 매트릭스
     * 승인/검토 프로세스 테이블
     * 의사결정 매트릭스
   - **시간축 및 단계축 식별**: 
     * 시간을 나타내는 행 또는 열 식별 (날짜, 주차, 월, 분기 등)
     * 단계나 순서를 나타내는 행 또는 열 식별
     * 우선순위나 중요도를 나타내는 축 파악
   - **각 단계/시점별 상세 분석**:
     * 각 셀에서 해당 시점/단계의 구체적인 활동이나 작업 설명
     * 각 단계의 담당자, 책임자, 관련 부서 정보
     * 각 단계별 목표, 산출물, 완료 기준 분석
     * 예상 소요 시간, 리소스, 비용 정보 (명시된 경우)
     * 각 단계의 상태나 진행도 (완료/진행중/예정 등)
   - **테이블 내 연결관계 및 의존성 분석**:
     * 행 간 연결관계 (순차적 진행, 병렬 수행, 조건부 분기)
     * 열 간 연결관계 (부서별, 역할별, 기능별 연관성)
     * 선행 작업과 후행 작업의 의존성 관계
     * 동시 진행 가능한 작업들과 순차 진행 필요한 작업들 구분
   - **마일스톤 및 중요 시점 식별**:
     * 프로젝트나 프로세스의 핵심 전환점 식별
     * 검토, 승인, 결정이 필요한 지점
     * 외부 이해관계자 참여 시점
     * 위험 요소나 주의 사항이 표시된 구간
   - **테이블 기반 프로세스 흐름 분석**:
     * 테이블 셀 간의 논리적 흐름과 순서 설명
     * 조건부 경로나 대안 프로세스 (있는 경우)
     * 반복 구간이나 피드백 루프 (있는 경우)
     * 예외 상황이나 에스컬레이션 프로세스

5. **수치 데이터 분석**:
   - 최댓값, 최솟값, 평균값 등 통계적 특성 분석
   - 수치의 변화 추세와 그 의미 해석
   - 백분율, 비율, 증감률 등 계산된 값의 의미

6. **범주형 데이터 분석**:
   - 카테고리별 분포와 특성 분석
   - 각 범주의 의미와 구분 기준 설명
   - 범주 간 비교와 차이점 분석

7. **중요한 정보 강조**:
   - 핵심 메시지나 주목할 만한 데이터 포인트 강조
   - 특별한 의미를 가지는 값이나 항목 식별
   - 의사결정에 중요한 정보 부각

8. **누락된 정보 파악**:
   - 빈 셀이나 누락된 데이터의 의미 해석
   - 데이터 수집의 한계나 제약사항 추정
   - 완전하지 않은 정보에 대한 추가 설명

9. **한국어 맥락 해석**:
   - 한국어 특성을 고려한 내용 해석
   - 문화적, 사회적 맥락에서의 데이터 의미 분석
   - 한국의 제도나 관습과 연관된 내용 설명

10. **실용적 활용 방안**:
    - 이 테이블 정보를 어떻게 활용할 수 있는지 제안
    - 후속 분석이나 의사결정에 도움이 되는 통찰 제공
    - 테이블이 전달하고자 하는 핵심 메시지 도출

**중요**: 각 분석 항목을 체계적으로 다루되, 단순 나열이 아닌 의미 있는 해석과 통찰을 제공하세요.
사용자가 이 해석만 읽고도 원본 테이블의 완전한 이해가 가능하도록 작성하세요.
        """
        
        content = [
            {"type": "text", "text": f"""
**TEXT-BASED COMPREHENSIVE TABLE ANALYSIS TASK**

**ELEMENT TYPE**: {category}
**SURROUNDING CONTEXT**: {surrounding_text}

**기존 테이블 정보**:
{content_summary}

{analysis_instructions}

**OUTPUT REQUIREMENTS**:
- **Text**: 위 분석 요구사항에 따른 포괄적이고 상세한 해석 결과
- **Markdown**: 기존 마크다운 구조를 개선하거나 유지 (| 형태)
- **HTML**: 기존 HTML 구조를 개선하거나 유지 (<table>...</table>)

**중요**: Markdown과 HTML은 실제 구조적 표현을, Text는 포괄적 해석을 제공하세요.

**핵심 원칙**:
1. **완전한 이해**: 사용자가 원본을 보지 않고도 이 해석만으로 모든 내용 이해 가능
2. **상세한 분석**: 각 요소의 내용을 하나하나 자세히 설명
3. **맥락적 해석**: 주변 컨텍스트를 적극 활용
4. **정확성**: 모든 데이터와 정보를 정확히 추출하고 분석
5. **한국어 맥락**: 한국어 특성을 고려한 해석 및 의미 부여

**언어**: 한국어로 작성하되, 전문 용어는 적절히 설명과 함께 사용
"""}
        ]
        
        if page_context:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{page_context}"}
            })
            content.append({"type": "text", "text": "↑ **페이지 컨텍스트**: 위 페이지 전체 이미지를 참고하여 이 테이블의 문서 내 위치와 맥락을 파악하세요."})
        
        return {"messages": [HumanMessage(content=content)]}


    def _normalize_coordinates(self, coordinates) -> str:

        if not coordinates:
            return ""
        
        try:
            if isinstance(coordinates, list) and len(coordinates) > 0:
                first_coord = coordinates[0]
                if isinstance(first_coord, dict) and 'x' in first_coord and 'y' in first_coord:
                    x = round(float(first_coord['x']), 1)
                    y = round(float(first_coord['y']), 1)
                    return f"{x},{y}"
            
            if isinstance(coordinates, dict):
                if 'bbox' in coordinates:
                    bbox = coordinates['bbox']
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        normalized_bbox = [round(float(coord), 1) for coord in bbox[:4]]
                        return f"{normalized_bbox[0]},{normalized_bbox[1]},{normalized_bbox[2]},{normalized_bbox[3]}"
                
                if 'x' in coordinates and 'y' in coordinates:
                    x = round(float(coordinates['x']), 1)
                    y = round(float(coordinates['y']), 1)
                    return f"{x},{y}"
            
            return str(coordinates)
            
        except (ValueError, TypeError, KeyError):
            return str(coordinates)

    def _deduplicate_elements(self, elements: List[Element]) -> List[Element]:

        if not elements:
            logger.info("Deduplication: No elements to process")
            return []
        
        logger.info(f"Deduplication: Processing {len(elements)} elements...")
        
        categories = {}
        for elem in elements:
            cat = elem.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        logger.info(f"   Categories: {dict(sorted(categories.items()))}")
        logger.info(f"   Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
        
        unique_elements = []
        signature_to_index = {}
        duplicates_found = 0
        similarity_checks = 0
        
        for elem in elements:
            signature = self._calculate_element_signature(elem)
            
            if signature not in signature_to_index:
                unique_elements.append(elem)
                signature_to_index[signature] = len(unique_elements) - 1
            else:
                existing_index = signature_to_index[signature]
                existing_elem = unique_elements[existing_index]
                
                similarity_checks += 1
                is_similar = self._are_similar_by_embedding(elem, existing_elem)
                
                if is_similar:
                    duplicates_found += 1
                    text1 = elem.get('content', {}).get('text', '')[:30]
                    text2 = existing_elem.get('content', {}).get('text', '')[:30]
                    logger.info(f"  🔄 Duplicate found: '{text1}' ≈ '{text2}'")
                    
                    if self._is_better_content(elem, existing_elem):
                        updated_elem = elem.copy()
                        if not updated_elem.get('base64_encoding') and existing_elem.get('base64_encoding'):
                            updated_elem['base64_encoding'] = existing_elem['base64_encoding']
                        unique_elements[existing_index] = updated_elem
                        logger.debug(f"    → Replaced with better content")
                    else:
                        logger.debug(f"    → Kept existing element")
                else:
                    modified_signature = f"{signature}_variant_{len(unique_elements)}"
                    unique_elements.append(elem)
                    signature_to_index[modified_signature] = len(unique_elements) - 1
        
        removal_count = len(elements) - len(unique_elements)
        removal_rate = (removal_count / len(elements)) * 100 if elements else 0
        
        logger.info(f"Deduplication completed:")
        logger.info(f"  ├─ Input elements: {len(elements)}")
        logger.info(f"  ├─ Output elements: {len(unique_elements)}")
        logger.info(f"  ├─ Removed duplicates: {removal_count}")
        logger.info(f"  ├─ Removal rate: {removal_rate:.1f}%")
        logger.info(f"  ├─ Similarity checks: {similarity_checks}")
        logger.info(f"  └─ Duplicates detected: {duplicates_found}")
        return unique_elements

    def _calculate_element_signature(self, element: Element) -> str:

        page = element.get('page', 0)
        category = element.get('category', '')
        
        coords = self._normalize_coordinates(element.get('coordinates', []))
        
        base_signature = f"{page}_{category}_{coords}"
        
        if category == 'figure' and element.get('base64_encoding'):
            img_size = len(element.get('base64_encoding', ''))
            return f"{base_signature}_{img_size}"
        
        if not coords:
            element_id = element.get('id', '')
            return f"{base_signature}_{element_id}"
        
        return base_signature

    def _is_better_content(self, new_elem: Element, existing_elem: Element) -> bool:

        status_priority = {
            'enhanced': 3,
            'processed': 2,
            'single_parser': 1,
            'failed': 0
        }
        
        new_status = status_priority.get(new_elem.get('processing_status', ''), 0)
        existing_status = status_priority.get(existing_elem.get('processing_status', ''), 0)
        
        if new_status != existing_status:
            return new_status > existing_status
        
        new_has_base64 = bool(new_elem.get('base64_encoding'))
        existing_has_base64 = bool(existing_elem.get('base64_encoding'))
        
        if new_has_base64 != existing_has_base64:
            return new_has_base64
        
        new_text = new_elem.get('content', {}).get('text', '')
        existing_text = existing_elem.get('content', {}).get('text', '')
        
        return len(new_text) > len(existing_text)

    def _are_similar_by_embedding(self, elem1: Element, elem2: Element) -> bool:

        try:
            non_text_categories = {'figure', 'table', 'chart', 'equation'}
            
            category1 = elem1.get('category', '')
            category2 = elem2.get('category', '')
            
            if category1 in non_text_categories or category2 in non_text_categories:
                logger.debug(f"Skipping similarity check for non-text categories: {category1} vs {category2}")
                return False
            
            text1 = elem1.get('content', {}).get('text', '')
            text2 = elem2.get('content', {}).get('text', '')
            
            if not text1 or not text2:
                return text1 == text2
            
            if text1.strip() == text2.strip():
                return True
            
            embeddings = self.embedding_model.embed_documents([text1, text2])
            
            if len(embeddings) != 2:
                logger.warning("Failed to generate embeddings for similarity check")
                return False
            
            import numpy as np
            
            vec1 = np.array(embeddings[0])
            vec2 = np.array(embeddings[1])
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return False
            
            cosine_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            
            is_similar = cosine_similarity >= config.SIMILARITY_THRESHOLD
            
            if is_similar:
                logger.debug(f"Embedding similarity: {cosine_similarity:.3f} >= {config.SIMILARITY_THRESHOLD} - SIMILAR")
            else:
                logger.debug(f"Embedding similarity: {cosine_similarity:.3f} < {config.SIMILARITY_THRESHOLD} - DIFFERENT")
            
            return is_similar
            
        except Exception as e:
            logger.error(f"Error in embedding similarity check: {e}")
            text1 = elem1.get('content', {}).get('text', '').strip()
            text2 = elem2.get('content', {}).get('text', '').strip()
            return text1 == text2