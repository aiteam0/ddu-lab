from hddu.state import ParseState
from .element import Element
from .logging_config import get_logger, setup_verbose_logging


logger = get_logger(__name__)
import base64
import os
import re
import json
import pickle
import re
from datetime import datetime
from typing import Dict
from .base import BaseNode
from langchain_core.documents import Document


IMAGE_TYPES = ["figure", "chart"]
TEXT_TYPES = ["equation", "caption", "paragraph", "list", "heading1", "heading2", "heading3", "footnote", "header", "footer", "reference"]
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
        logger.error(f"Entity 파싱 오류: {e}")
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
            logger.info("SaveImageNode: 이미지 저장 시작")
        
        directory = os.path.dirname(state["filepath"])
        base_filename = os.path.splitext(os.path.basename(state["filepath"]))[0]
        image_paths = {}  # element_id -> image_path 매핑
        
        saved_count = 0
        for element in state["elements_from_parser"]:
            element_id = element["id"]
            category = element["category"]
            base64_encoding = element.get("base64_encoding")
            
            # 이미지 저장 (base64_encoding이 있는 경우만) -> 텍스트 타입 요소들도 이미지 파일로 저장되고, 전체 파이프라인에서 이미지 경로 정보가 보존
            #if base64_encoding and category in ["table", "figure", "chart"]:
            if base64_encoding:
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
                        logger.info(f"  이미지 저장 성공: {category} ID {element_id} -> {image_filepath}")
                        
                except Exception as e:
                    if self.verbose:
                        logger.error(f"  이미지 저장 실패: ID {element_id}, Error: {e}")
        
        if self.verbose:
            logger.info(f"SaveImageNode: 총 {saved_count}개 이미지 저장 완료")
        
        return {"image_paths": image_paths}


class RefineContentNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        
        self.image_link_pattern = re.compile(r'!\[.*?\]\(.*?\)')
        self.whitespace_pattern = re.compile(r'\s+')
        self.newline_pattern = re.compile(r'\n\s*\n\s*\n+')
        
        self.process_fields = [
            "text", "markdown", "html", 
            "translation_text", "translation_markdown", "translation_html", 
            "contextualize_text"
        ]

    def _clean_text(self, text: str) -> str:
        if not text or len(text) <= 20:
            return text
            
        cleaned = self.image_link_pattern.sub('', text)
        
        cleaned = self.whitespace_pattern.sub(' ', cleaned)
        
        cleaned = self.newline_pattern.sub('\n\n', cleaned)
        
        return cleaned.strip()
    
    def _should_process_field(self, text: str) -> bool:
        return text and len(text) > 20 and self.image_link_pattern.search(text) is not None

    def execute(self, state: ParseState) -> ParseState:
        if self.verbose:
            setup_verbose_logging(self.verbose)
            logger.info("RefineContentNode: 정규식 기반 텍스트 정제 시작")
        
        elements = state["elements_from_parser"]
        
        total_cleaned = 0
        total_fields_processed = 0
        
        for element in elements:
            content = element["content"]
            
            for field_name in self.process_fields:
                if field_name in content and content[field_name]:
                    original_text = content[field_name]
                    
                    if self._should_process_field(original_text):
                        cleaned_text = self._clean_text(original_text)
                        
                        if cleaned_text != original_text:
                            content[field_name] = cleaned_text
                            total_cleaned += 1
                            
                            if self.verbose:
                                logger.debug(f"  정제 완료: Element {element['id']} - {field_name}")
                    
                    total_fields_processed += 1
        
        if self.verbose:
            logger.info(f"RefineContentNode: {total_fields_processed}개 필드 중 {total_cleaned}개 필드 정제 완료")
        
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

             #"header", 
            if category in ["footnote", "footer"]:
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
                "entity": getattr(elem, 'entity', {}),  # 구조화된 딕셔너리
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
                #page_data["text"] += elem.content
                # 원문 마크다운 추가
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
            self.log(f"LangChain Documents가 pickle 파일로 저장되었습니다: {pickle_filepath}")
        except Exception as e:
            self.log(f"pickle 저장 중 오류 발생: {e}")

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
            self.log(f"최종 state가 JSON 파일로 저장되었습니다: {state_json_path}")
            
            with open(state_pickle_path, "wb") as f:
                pickle.dump(state, f)
            self.log(f"최종 state가 pickle 파일로 저장되었습니다: {state_pickle_path}")
            
        except Exception as e:
            self.log(f"최종 state 저장 중 오류 발생: {e}")
        
        return {
            "final_state_json_path": state_json_path,
            "final_state_pickle_path": state_pickle_path,
            "workflow_completed": True
        }
