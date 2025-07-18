from hddu.config import (
    create_text_model,
    PREPROCESSING_TEMPERATURE,
    PREPROCESSING_MAX_TOKENS,
    PREPROCESSING_BATCH_SIZE,
    PREPROCESSING_MAX_RETRIES,
    PREPROCESSING_RETRY_DELAY,
    PREPROCESSING_BATCH_REDUCTION_FACTOR
)

from hddu.state import ParseState
from .element import Element
from .logging_config import get_logger, setup_verbose_logging

# 모듈 로거 생성
logger = get_logger(__name__)
import base64
import os
import re
import json
import pickle
from datetime import datetime
from typing import Dict, List
from .base import BaseNode
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


IMAGE_TYPES = ["figure", "chart"]
TEXT_TYPES = ["text", "equation", "caption", "paragraph", "list", "index", "heading1", "heading2", "heading3"]
TABLE_TYPES = ["table"]


def parse_llm_entity_output(llm_output: str, entity_type: str) -> Dict:

    try:
        tag_pattern = f"<{entity_type}>(.*?)</{entity_type}>"
        match = re.search(tag_pattern, llm_output, re.DOTALL)
        if not match:
            return _create_empty_entity(entity_type, llm_output)
        
        content = match.group(1).strip()
        
        title = _extract_section(content, "title")
        details = _extract_section(content, "details")
        entities_text = _extract_section(content, "entities")
        questions = _extract_section(content, "hypothetical_questions")
        
        keywords = []
        if entities_text:
            keywords = [keyword.strip() for keyword in entities_text.split(",")]
            keywords = [k for k in keywords if k]
        
        return {
            "type": entity_type,
            "title": title,
            "details": details,
            "keywords": keywords,
            "hypothetical_questions": questions,
            "raw_output": llm_output
        }
    
    except Exception as e:
        logger.error(f"Entity parsing error: {e}")
        return _create_empty_entity(entity_type, llm_output)


def _extract_section(content: str, section_name: str) -> str:
    pattern = f"<{section_name}>(.*?)</{section_name}>"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""


def _create_empty_entity(entity_type: str, raw_output: str) -> Dict:
    return {
        "type": entity_type,
        "title": "",
        "details": "",
        "keywords": [],
        "hypothetical_questions": "",
        "raw_output": raw_output
    }


class SaveImageNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def _save_base64_image(self, base64_str, basename, page_num, element_id, directory, category):
        img_dir = os.path.join(directory, "images", category)
        os.makedirs(img_dir, exist_ok=True)
        
        img_filename = f"{basename}_{category}_Page_{page_num}_Index_{element_id}.png"
        img_filepath = os.path.join(img_dir, img_filename)
        
        img_data = base64.b64decode(base64_str)
        with open(img_filepath, "wb") as f:
            f.write(img_data)
        return img_filepath

    def execute(self, state: ParseState) -> ParseState:
        if self.verbose:
            setup_verbose_logging(self.verbose)
            logger.info("SaveImageNode: image saving started")
        
        directory = os.path.dirname(state["filepath"])
        base_filename = os.path.splitext(os.path.basename(state["filepath"]))[0]
        image_paths = {}
        
        saved_count = 0
        for element in state["elements_from_parser"]:
            element_id = element["id"]
            category = element["category"]
            base64_encoding = element.get("base64_encoding")
            
            if base64_encoding and category in ["table", "figure", "chart"]:
                try:
                    image_filepath = self._save_base64_image(
                        base64_encoding,
                        base_filename,
                        element["page"],
                        element_id,
                        directory,
                        category
                    )
                    image_paths[element_id] = image_filepath
                    saved_count += 1
                    
                    if self.verbose:
                        logger.info(f"Image saved successfully: {category} ID {element_id} -> {image_filepath}")
                        
                except Exception as e:
                    if self.verbose:
                        logger.error(f"Image saving failed: ID {element_id}, Error: {e}")
        
        if self.verbose:
            logger.info(f"SaveImageNode: total {saved_count} images saved")
        
        return {"image_paths": image_paths}


class RefineContentNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.llm = create_text_model(
            temperature=PREPROCESSING_TEMPERATURE,
            max_tokens=PREPROCESSING_MAX_TOKENS
        )
        self.output_parser = StrOutputParser()
        
        self.refine_prompt = PromptTemplate.from_template("""
You are a text refinement assistant. Your task is to improve and refine the given text content by:

1. Replacing any temporary image paths or placeholders with actual saved image paths
2. Improving readability and coherence
3. Maintaining the original meaning and structure
4. Ensuring proper markdown formatting
5. Transform Arabic text to Korean text considering the context of the text

**Available Image Information:**
{image_info}

**Original Text:**
{original_text}

**Instructions:**
- If you find references to images in the text, replace them with appropriate markdown image links using the saved image paths
- Improve the text quality while preserving the original meaning
- Return only the refined text, do not add explanations
- Keep the same language as the original text

**Refined Text:**
""")

    def _get_image_info_for_element(self, element_id: str, image_paths: Dict) -> str:
        if element_id in image_paths:
            return f"Element {element_id}: {image_paths[element_id]}"
        return "No image available for this element"

    def _should_refine_element(self, element: dict) -> bool:
        content = element["content"]
        
        for field_name in ["text", "markdown", "html"]:
            if field_name in content and content[field_name] and len(content[field_name]) > 20:
                return True
        return False

    def _prepare_batch_data(self, elements: List[dict], image_paths: Dict) -> List[dict]:
        batch_data = []
        
        for element in elements:
            element_id = element["id"]
            content = element["content"]
            image_info = self._get_image_info_for_element(element_id, image_paths)
            
            for field_name in ["text", "markdown", "html"]:
                if field_name in content and content[field_name] and len(content[field_name]) > 20:
                    batch_data.append({
                        "element_id": element_id,
                        "field_name": field_name,
                        "original_text": content[field_name],
                        "image_info": image_info
                    })
        
        return batch_data

    def _refine_batch_with_retry(self, batch_data: List[dict], max_retries: int = None) -> Dict[str, Dict[str, str]]:
        if max_retries is None:
            max_retries = PREPROCESSING_MAX_RETRIES
        
        if not batch_data:
            return {}
        
        results = {}
        
        for attempt in range(max_retries):
            try:
                if self.verbose:
                    logger.info(f"  배치 처리 시도 {attempt + 1}/{max_retries}, 배치 크기: {len(batch_data)}")
                
                prompt_data = [
                    {
                        "original_text": item["original_text"],
                        "image_info": item["image_info"]
                    }
                    for item in batch_data
                ]
                
                chain = self.refine_prompt | self.llm | self.output_parser
                refined_texts = chain.batch(prompt_data)
                
                for batch_item, refined_text in zip(batch_data, refined_texts):
                    element_id = batch_item["element_id"]
                    field_name = batch_item["field_name"]
                    
                    if element_id not in results:
                        results[element_id] = {}
                    results[element_id][field_name] = refined_text.strip()
                
                if self.verbose:
                    logger.info(f"Batch processing successful: {len(batch_data)} texts refined")
                
                return results
                
            except Exception as e:
                if self.verbose:
                    logger.error(f"  Batch processing failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    import time
                    wait_time = PREPROCESSING_RETRY_DELAY * (2 ** attempt)
                    if self.verbose:
                        logger.info(f"  {wait_time:.1f}초 대기 후 재시도...")
                    time.sleep(wait_time)
                    
                    if len(batch_data) > 1:
                        new_size = max(1, int(len(batch_data) * PREPROCESSING_BATCH_REDUCTION_FACTOR))
                        batch_data = batch_data[:new_size]
                        if self.verbose:
                            logger.info(f"Batch size reduced to {new_size}")
                else:
                    if self.verbose:
                        logger.warning(f"Batch processing final failure, fallback to individual processing")
                    return self._fallback_individual_processing(batch_data)
        
        return {}

    def _fallback_individual_processing(self, batch_data: List[dict]) -> Dict[str, Dict[str, str]]:
        results = {}
        
        for item in batch_data:
            try:
                chain = self.refine_prompt | self.llm | self.output_parser
                refined_text = chain.invoke({
                    "original_text": item["original_text"],
                    "image_info": item["image_info"]
                })
                
                element_id = item["element_id"]
                field_name = item["field_name"]
                
                if element_id not in results:
                    results[element_id] = {}
                results[element_id][field_name] = refined_text.strip()
                
                if self.verbose:
                    logger.info(f"Individual processing successful: Element {element_id} - {field_name}")
                    
            except Exception as e:
                if self.verbose:
                    logger.error(f"Individual processing failed: Element {item['element_id']} - {item['field_name']}: {e}")
        
        return results

    def _apply_refined_results(self, elements: List[dict], results: Dict[str, Dict[str, str]]) -> int:
        applied_count = 0
        
        for element in elements:
            element_id = element["id"]
            if element_id in results:
                content = element["content"]
                for field_name, refined_text in results[element_id].items():
                    if field_name in content:
                        content[field_name] = refined_text
                        applied_count += 1
        
        return applied_count

    def execute(self, state: ParseState) -> ParseState:
        if self.verbose:
            setup_verbose_logging(self.verbose)
            logger.info("RefineContentNode: batch text refinement started")
        
        image_paths = state.get("image_paths", {})
        elements = state["elements_from_parser"]
        
        refinable_elements = [elem for elem in elements if self._should_refine_element(elem)]
        
        if not refinable_elements:
            if self.verbose:
                logger.info("RefineContentNode: no elements to refine")
            return {"elements_from_parser": elements}
        
        if self.verbose:
            logger.info(f"RefineContentNode: {len(refinable_elements)} elements to refine")
        
        batch_size = PREPROCESSING_BATCH_SIZE
        total_refined = 0
        
        for i in range(0, len(refinable_elements), batch_size):
            batch_elements = refinable_elements[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(refinable_elements) + batch_size - 1) // batch_size
            
            if self.verbose:
                logger.info(f"RefineContentNode: batch {batch_num}/{total_batches} processing ({len(batch_elements)} elements)")
            
            batch_data = self._prepare_batch_data(batch_elements, image_paths)
            
            if not batch_data:
                if self.verbose:
                    logger.info(f"Batch {batch_num}: no text to process")
                continue
            
            batch_results = self._refine_batch_with_retry(batch_data)
            
            applied_count = self._apply_refined_results(batch_elements, batch_results)
            total_refined += applied_count
            
            if self.verbose:
                logger.info(f"Batch {batch_num} completed: {applied_count} texts refined")
        
        if self.verbose:
            logger.info(f"RefineContentNode: total {total_refined} texts refined")
        
        return {"elements_from_parser": elements}


class CreateElementsNode(BaseNode):
    def __init__(self, verbose=False, add_newline=True, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.add_newline = add_newline
        self.newline = "\n" if add_newline else ""

    def execute(self, state: ParseState) -> ParseState:
        post_processed_elements = []
        image_paths = state.get("image_paths", {})

        for element in state["elements_from_parser"]:
            category = element["category"]
            content = element["content"]
            
            base64_encoding = element.get("base64_encoding")
            coordinates = element.get("coordinates", [])
            caption = element.get("caption", "")
            processing_type = element.get("processing_type", "")
            processing_status = element.get("processing_status", "")
            source_parser = element.get("source_parser", "")
            
            text = content.get("text", "")
            markdown = content.get("markdown", "")
            html = content.get("html", "")
            raw_output = content.get("raw_output", "")
            translation_text = content.get("translation_text", "")
            translation_markdown = content.get("translation_markdown", "")
            translation_html = content.get("translation_html", "")
            contextualize_text = content.get("contextualize_text", "")

            if category in ["footnote", "header", "footer"]:
                continue

            image_filename = image_paths.get(element["id"])

            if category in ["equation"]:
                content_text = markdown + self.newline if markdown else text + self.newline
            elif category in ["heading1", "heading2", "heading3"]:
                content_text = f'# {text}{self.newline}' if text else markdown + self.newline
            else:
                content_text = text + self.newline if text else markdown + self.newline

            elem = Element(
                category=category,
                content=content_text,
                html=html,
                markdown=markdown,
                base64_encoding=base64_encoding,
                image_filename=image_filename,
                page=element["page"],
                id=element["id"],
                coordinates=coordinates,
                caption=caption,
                translation_text=translation_text,
                translation_html=translation_html,
                translation_markdown=translation_markdown,
                contextualize_text=contextualize_text,
                processing_type=processing_type,
                processing_status=processing_status,
                source_parser=source_parser,
                raw_output=raw_output
            )
            
            post_processed_elements.append(elem)

        return {"elements": post_processed_elements}


class MergeEntityNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]

        for extracted_elem in state["extracted_image_entities"]:
            for elem in elements:
                if extracted_elem.id == elem.id:
                    parsed_entity = parse_llm_entity_output(
                        extracted_elem.entity, 
                        "image"
                    )
                    elem.entity = parsed_entity
                    break

        for extracted_elem in state["extracted_table_entities"]:
            for elem in elements:
                if extracted_elem.id == elem.id:
                    parsed_entity = parse_llm_entity_output(
                        extracted_elem.entity, 
                        "table"
                    )
                    elem.entity = parsed_entity
                    break

        for elem in elements:
            if not hasattr(elem, 'entity') or elem.entity is None:
                elem.entity = {
                    "type": "text",
                    "title": "",
                    "details": "",
                    "keywords": [],
                    "hypothetical_questions": "",
                    "raw_output": ""
                }

        return {"elements": elements}


class ReconstructElementsNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def _add_src_to_markdown(self, image_filename):
        if not image_filename:
            return ""
        abs_image_path = os.path.abspath(image_filename)
        image_md = f"![](file:///{abs_image_path})"
        return image_md

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]
        filepath = state["filepath"]
        
        original_source = filepath.replace("\\", "/")

        pages = sorted(list(state["texts_by_page"].keys()))
        max_page = pages[-1]

        reconstructed_elements = dict()
        for page_num in range(1, max_page + 1):
            reconstructed_elements[int(page_num)] = {
                "text": "",
                "image": [],
                "table": [],
                "elements": []
            }

        for elem in elements:
            page_data = reconstructed_elements[elem.page]
            
            element_data = {
                "id": elem.id,
                "category": elem.category,
                "page": elem.page,
                "original_text": elem.content,
                "original_markdown": elem.markdown,
                "original_html": elem.html,
                "raw_output": getattr(elem, 'raw_output', ''),
                "translation_text": getattr(elem, 'translation_text', ''),
                "translation_markdown": getattr(elem, 'translation_markdown', ''),
                "translation_html": getattr(elem, 'translation_html', ''),
                "contextualize_text": getattr(elem, 'contextualize_text', ''),
                "caption": getattr(elem, 'caption', ''),
                "entity": getattr(elem, 'entity', {}),
                "image_path": getattr(elem, 'image_filename', ''),
                "coordinates": getattr(elem, 'coordinates', []),
                "processing_type": getattr(elem, 'processing_type', ''),
                "processing_status": getattr(elem, 'processing_status', ''),
                "source_parser": getattr(elem, 'source_parser', ''),
                "source": original_source
            }
            page_data["elements"].append(element_data)
            
            if elem.category in TABLE_TYPES:
                entity_summary = elem.entity.get("title", "") or elem.entity.get("details", "")
                
                table_content = getattr(elem, 'raw_output', '') or elem.markdown
                
                table_elem = {
                    "content": table_content + "\n\n" + entity_summary,
                    "metadata": element_data
                }
                page_data["table"].append(table_elem)
            elif elem.category in IMAGE_TYPES:
                entity_summary = elem.entity.get("title", "") or elem.entity.get("details", "")
                image_elem = {
                    "content": self._add_src_to_markdown(elem.image_filename) + "\n\n" + entity_summary,
                    "metadata": element_data
                }
                page_data["image"].append(image_elem)
            elif elem.category in TEXT_TYPES:
                page_data["text"] += elem.markdown

        return {"reconstructed_elements": reconstructed_elements}


class GenerateComprehensiveMarkdownNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)
    
    def execute(self, state: ParseState) -> ParseState:
        reconstructed_elements = state["reconstructed_elements"]
        filepath = state["filepath"]
        
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        md_filename = f"{base_filename}.md"
        
        export_dir = "export"
        os.makedirs(export_dir, exist_ok=True)
        md_filepath = os.path.join(export_dir, md_filename)
        
        all_elements = []
        for page_num, page_data in reconstructed_elements.items():
            all_elements.extend(page_data["elements"])
        
        sorted_elements = sorted(all_elements, key=lambda x: (x["page"], x["id"]))
        
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(f"# {base_filename}\n\n")
            
            current_page = -1
            for elem in sorted_elements:
                if elem["page"] != current_page:
                    f.write(f"\n## 페이지 {elem['page']}\n\n")
                    current_page = elem["page"]
                
                f.write(f"### 요소 ID: {elem['id']} ({elem['category']})\n\n")

                if elem["image_path"]:
                    rel_path = os.path.relpath(elem["image_path"], export_dir)
                    f.write(f"**이미지:**\n![{elem['category']}]({rel_path})\n\n")
                
                if elem["category"] in TEXT_TYPES:
                    f.write(f"**원문:**\n{elem['original_markdown']}\n\n")
                else:
                    f.write(f"**원문:**\n{elem['original_text']}\n\n")
                
                if elem["translation_text"]:
                    f.write(f"**번역:**\n{elem['translation_text']}\n\n")
                
                if elem["contextualize_text"]:
                    f.write(f"**문맥화 정보:**\n{elem['contextualize_text']}\n\n")
                
                if elem["caption"]:
                    f.write(f"**캡션:**\n{elem['caption']}\n\n")
                
                entity = elem["entity"]
                if entity and entity.get("title"):
                    f.write(f"**AI 분석 제목:**\n{entity['title']}\n\n")
                
                if entity and entity.get("details"):
                    f.write(f"**AI 분석 상세:**\n{entity['details']}\n\n")
                
                if entity and entity.get("keywords"):
                    keywords = ", ".join(entity["keywords"])
                    f.write(f"**AI 추출 키워드:**\n{keywords}\n\n")
                
                if entity and entity.get("hypothetical_questions"):
                    f.write(f"**AI 생성 질문:**\n{entity['hypothetical_questions']}\n\n")
                
                f.write("---\n\n")
        
        return {"comprehensive_markdown": md_filepath}


class LangChainDocumentNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def execute(self, state: ParseState) -> ParseState:
        reconstructed_elements = state["reconstructed_elements"]
        filepath = state["filepath"]
        
        original_source = filepath.replace("\\", "/")
        
        documents = []
        
        for page_num, page_data in reconstructed_elements.items():
            for element in page_data["elements"]:
                doc = Document(
                    page_content=element["original_markdown"] if element["category"] in TEXT_TYPES else element["original_text"],
                    metadata={
                        "source": original_source,
                        "page": element["page"],
                        "category": element["category"],
                        "id": element["id"],
                        "raw_output": element["raw_output"],
                        "translation_text": element["translation_text"],
                        "translation_markdown": element["translation_markdown"],
                        "translation_html": element["translation_html"],
                        "contextualize_text": element["contextualize_text"],
                        "caption": element["caption"],
                        "entity": element["entity"],
                        "image_path": element["image_path"],
                        "coordinates": element["coordinates"],
                        "processing_type": element["processing_type"],
                        "processing_status": element["processing_status"],
                        "source_parser": element["source_parser"],
                        "element_type": "text" if element["category"] in TEXT_TYPES else 
                                      "image" if element["category"] in IMAGE_TYPES else "table",
                        "human_feedback": ""
                    }
                )
                documents.append(doc)

        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        
        export_dir = "export"
        os.makedirs(export_dir, exist_ok=True)
        
        pickle_filepath = os.path.join(export_dir, f"{base_filename}_documents.pkl")
        
        try:
            with open(pickle_filepath, "wb") as f:
                pickle.dump(documents, f)
            self.log(f"LangChain Documents saved to pickle file: {pickle_filepath}")
        except Exception as e:
            self.log(f"Error saving pickle file: {e}")

        return {"documents": documents, "documents_pickle_path": pickle_filepath}


class SaveFinalStateNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def execute(self, state: ParseState) -> ParseState:
        filepath = state["filepath"]
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        
        export_dir = "export"
        os.makedirs(export_dir, exist_ok=True)
        
        state_json_path = os.path.join(export_dir, f"{base_filename}_final_state.json")
        state_pickle_path = os.path.join(export_dir, f"{base_filename}_final_state.pkl")
        
        serializable_state = {}
        for key, value in state.items():
            if key == "elements" and isinstance(value, list):
                serializable_state[key] = []
                for elem in value:
                    if hasattr(elem, '__dict__'):
                        elem_dict = elem.__dict__.copy()
                        serializable_state[key].append(elem_dict)
                    else:
                        serializable_state[key].append(value)
            elif key == "documents" and isinstance(value, list):
                serializable_state[key] = []
                for doc in value:
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        doc_dict = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        serializable_state[key].append(doc_dict)
                    else:
                        serializable_state[key].append(str(doc))
            else:
                try:
                    json.dumps(value)
                    serializable_state[key] = value
                except (TypeError, ValueError):
                    serializable_state[key] = str(value)

        serializable_state["final_processing_completed_at"] = datetime.now().isoformat()
        
        try:
            with open(state_json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_state, f, ensure_ascii=False, indent=2)
            self.log(f"Final state saved to JSON file: {state_json_path}")
            
            with open(state_pickle_path, "wb") as f:
                pickle.dump(state, f)
            self.log(f"Final state saved to pickle file: {state_pickle_path}")
            
        except Exception as e:
            self.log(f"Error saving final state: {e}")
        
        return {
            "final_state_json_path": state_json_path,
            "final_state_pickle_path": state_pickle_path,
            "workflow_completed": True
        }
