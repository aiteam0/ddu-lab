import json, logging
from typing import List, Dict, Tuple, Optional
from .config import Element, GroupedData
from pathlib import Path
from ..logging_config import get_logger

logger = get_logger(__name__)
logger.info("[ASSEMBLY_INIT] preparer.py logger initialized - testing logging system")

def load_and_group_by_page(docling_path: str, docyolo_path: str) -> GroupedData:
    try:
        with open(docling_path, 'r', encoding='utf-8') as f: docling_data = json.load(f)
        with open(docyolo_path, 'r', encoding='utf-8') as f: docyolo_data = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}"); raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {e}"); raise

    cleaned_docling = _remove_single_noise_and_reindex(docling_data, 'docling')
    cleaned_docyolo = _remove_single_noise_and_reindex(docyolo_data, 'docyolo')
    
    docling_elements, docyolo_elements = cleaned_docling.get('elements', []), cleaned_docyolo.get('elements', [])
    grouped_data: GroupedData = {}

    for elem in docling_elements:
        page_num = elem.get('page')
        if page_num is not None:
            if page_num not in grouped_data: grouped_data[page_num] = {'docling': [], 'docyolo': []}
            grouped_data[page_num]['docling'].append(elem)
    for elem in docyolo_elements:
        page_num = elem.get('page')
        if page_num is not None:
            if page_num not in grouped_data: grouped_data[page_num] = {'docling': [], 'docyolo': []}
            grouped_data[page_num]['docyolo'].append(elem)
    logger.info(f"Loaded and grouped data for {len(grouped_data)} pages.")
    return grouped_data

def classify_elements(elements: List[Element]) -> Tuple[List[Element], List[Element], List[Element]]:
    text_like, tables, figures = [], [], []
    text_categories = {'paragraph', 'heading1', 'heading2', 'heading3', 'list', 'footer', 'header', 'caption'}
    for elem in elements:
        category = elem.get('category', '').lower()
        if category in text_categories: 
            if category == 'caption':
                elem_copy = elem.copy()
                elem_copy['element_type'] = 'caption'
                elem_copy['is_caption'] = True
                text_like.append(elem_copy)
            else:
                text_like.append(elem)
        elif category == 'table': 
            elem_copy = elem.copy()
            elem_copy['original_category'] = 'table'
            elem_copy['processing_type'] = 'table'
            
            caption_text = elem_copy.get('content', {}).get('caption', '')
            if caption_text:
                elem_copy['caption'] = caption_text
                elem_copy['has_caption'] = True
                logger.debug(f"[CAPTION_EXTRACT] Table caption: '{caption_text[:30]}...' (len={len(caption_text)})")
            
            tables.append(elem_copy)
        elif category == 'figure': 
            elem_copy = elem.copy()
            elem_copy['original_category'] = 'figure'
            elem_copy['processing_type'] = 'figure'
            
            caption_text = elem_copy.get('content', {}).get('caption', '')
            if caption_text:
                elem_copy['caption'] = caption_text
                elem_copy['has_caption'] = True
                logger.debug(f"[CAPTION_EXTRACT] Figure caption: '{caption_text[:30]}...' (len={len(caption_text)})")
            
            figures.append(elem_copy)
        else: 
            text_like.append(elem)
    
    logger.info(f"Classified elements: {len(text_like)} texts (including captions), {len(tables)} tables, {len(figures)} figures")
    caption_count = len([e for e in text_like if e.get('is_caption')])
    table_with_caption_count = len([e for e in tables if e.get('has_caption')])
    figure_with_caption_count = len([e for e in figures if e.get('has_caption')])
    logger.info(f"Caption processing: {caption_count} standalone captions, {table_with_caption_count} tables with captions, {figure_with_caption_count} figures with captions")
    logger.debug(f"[ELEMENT_CLASSIFY] Total processed: {len(elements)} → "
                f"texts: {len(text_like)}, tables: {len(tables)}, figures: {len(figures)}")
    
    return text_like, tables, figures



def _remove_single_noise_and_reindex(parser_result: Dict, parser_type: str) -> Dict:

    if 'elements' not in parser_result:
        return parser_result
    
    original_elements = parser_result['elements']
    original_count = len(original_elements)
    
    filtered_elements = [
        elem for elem in original_elements 
        if not _is_single_noise(elem)
    ]
    
    for i, elem in enumerate(filtered_elements):
        elem['id'] = i
    
    result = parser_result.copy()
    result['elements'] = filtered_elements
    
    removed_count = original_count - len(filtered_elements)
    logger.info(f"{parser_type} single noise removal: {original_count} -> {len(filtered_elements)} elements ({removed_count} removed)")
    
    return result

def _is_single_noise(element: Dict) -> bool:

    category = element.get('category', '').lower()
    element_id = element.get('id', 'unknown')
    
    # # figure, table 등 시각적 요소는 text content가 비어있어도 노이즈가 아님
    # visual_categories = {'figure', 'table'}
    # if (category in visual_categories) and (element.get("base64_encoding") is not None):
    #     logger.debug(f"[NOISE_CHECK] Element {element_id} ({category}) - Protected from noise removal")
    #     return False
    
    # base64_encoding이나 다른 content가 있는 경우 노이즈가 아님
    if element.get('base64_encoding') or element.get('content', {}).get('html'):
        logger.debug(f"[NOISE_CHECK] Element {element_id} - Protected due to base64/html content")
        return False
    
    text = element.get('content', {}).get('text', '')
    
    stripped_text = text.strip()
    
    if not text:
        logger.debug(f"[NOISE_CHECK] Element {element_id} ({category}) - Marked as noise: empty text")
        return True
    
    if not stripped_text:
        logger.debug(f"[NOISE_CHECK] Element {element_id} ({category}) - Marked as noise: whitespace only")
        return True
    
    if len(stripped_text) == 1 and stripped_text in '._':
        logger.debug(f"[NOISE_CHECK] Element {element_id} ({category}) - Marked as noise: single special char '{stripped_text}'")
        return True
    
    logger.debug(f"[NOISE_CHECK] Element {element_id} ({category}) - Valid content, not noise")
    return False

def create_final_structure(original_docling_path: str, final_elements: List[Element], unified_content: Dict) -> Dict:
    try:
        with open(original_docling_path, 'r', encoding='utf-8') as f: original_data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read original docling file for metadata: {e}. Using a default structure.")
        original_data = {"api": "2.0", "model": "assembled-v1.0", "usage": {"pages": 0}}
    if final_elements:
        max_page = max(e.get('page', 0) for e in final_elements if e.get('page') is not None)
        original_data['usage']['pages'] = max_page
    original_data['model'], original_data['elements'], original_data['content'] = 'assembled-v1.0', final_elements, unified_content
    return original_data