import logging
import os
import json
import base64
import re
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from langchain_core.messages import HumanMessage
from .config import Element
from ..logging_config import get_logger

logger = get_logger(__name__)
logger.info("[ASSEMBLY_INIT] postprocessor.py logger initialized - testing logging system")

def create_standard_page_images(image_dir: str, base_filename: str) -> bool:

    try:
        logger.info(f"[PAGE_CONVERT] Starting page image conversion for {base_filename}")
        logger.info(f"[PAGE_CONVERT] Scanning directory: {image_dir}")
        
        if not os.path.exists(image_dir):
            logger.warning(f"[PAGE_CONVERT] Image directory not found: {image_dir}")
            return False
        
        base_pattern = re.match(r'^(.+?)_\d{4}_\d{4}$', base_filename)
        if not base_pattern:
            logger.warning(f"[PAGE_CONVERT] Cannot parse base filename: {base_filename}")
            return False
        
        true_base = base_pattern.group(1)
        logger.info(f"[PAGE_CONVERT] True base name: {true_base}")
        
        pattern = re.compile(rf"^{re.escape(true_base)}_(\d{{4}})_(\d{{4}})-page-(\d+)\.png$")
        
        found_files = []
        for filename in os.listdir(image_dir):
            match = pattern.match(filename)
            if match:
                start_page = int(match.group(1))
                end_page = int(match.group(2))
                page_in_batch = int(match.group(3))
                actual_page = start_page + page_in_batch
                
                found_files.append({
                    'original_name': filename,
                    'start_page': start_page,
                    'end_page': end_page,
                    'page_in_batch': page_in_batch,
                    'actual_page': actual_page
                })
        
        if not found_files:
            logger.warning(f"[PAGE_CONVERT] No matching files found for pattern: {true_base}_*_*-page-*.png")
            return False
        
        found_files.sort(key=lambda x: x['actual_page'])
        
        logger.info(f"[PAGE_CONVERT] Found {len(found_files)} page image files:")
        for file_info in found_files:
            logger.info(f"[PAGE_CONVERT]   {file_info['original_name']} → page {file_info['actual_page']} "
                       f"(range {file_info['start_page']}-{file_info['end_page']}, pos {file_info['page_in_batch']})")
        
        conversion_count = 0
        for file_info in found_files:
            original_path = os.path.join(image_dir, file_info['original_name'])
            standard_filename = f"{true_base}_page_{file_info['actual_page']}.png"
            standard_path = os.path.join(image_dir, standard_filename)
            
            if os.path.exists(standard_path):
                try:
                    if os.path.samefile(original_path, standard_path):
                        logger.debug(f"[PAGE_CONVERT] Standard file already exists: {standard_filename}")
                        conversion_count += 1
                        continue
                except:
                    pass
            
            try:
                if os.name == 'nt':
                    shutil.copy2(original_path, standard_path)
                    logger.info(f"[PAGE_CONVERT] Copied: {file_info['original_name']} → {standard_filename} (page {file_info['actual_page']})")
                else:
                    if os.path.exists(standard_path):
                        os.unlink(standard_path)
                    os.symlink(file_info['original_name'], standard_path)
                    logger.info(f"[PAGE_CONVERT] Linked: {file_info['original_name']} → {standard_filename} (page {file_info['actual_page']})")
                
                conversion_count += 1
                
            except Exception as e:
                logger.error(f"[PAGE_CONVERT] Failed to create standard file {standard_filename}: {e}")
        
        logger.info(f"[PAGE_CONVERT] Conversion completed: {conversion_count}/{len(found_files)} files converted")
        return conversion_count > 0
        
    except Exception as e:
        logger.error(f"[PAGE_CONVERT] Error during page image conversion: {e}")
        return False

def finalize_elements(merged_elements: List[Element]) -> List[Element]:
    def sort_key(elem):
        page = elem.get('page', 0)
        coords = elem.get('coordinates')
        if coords and isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], dict):
            y_coord = coords[0].get('y', 0)
            return (page, y_coord)
        elif coords and isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], (int, float)):
             y_coord = coords[1]
             return (page, y_coord)
        return (page, float('inf')) 

    try:
        sorted_elements = sorted(merged_elements, key=sort_key)
    except (TypeError, KeyError) as e:
        logger.warning(f"Could not sort elements due to inconsistent data, proceeding without sorting. Error: {e}")
        sorted_elements = merged_elements

    for i, elem in enumerate(sorted_elements):
        elem['id'] = i
        
    logger.info(f"Finalized and re-indexed {len(sorted_elements)} elements.")
    return sorted_elements

async def finalize_elements_with_llm(merged_elements: List[Element], vision_llm, 
                                   image_dir: str, base_filename: str, use_async: bool = True) -> List[Element]:

    logger.info("Starting LLM-based ID assignment...")

    logger.info("[PAGE_CONVERT] Converting page images to standard format...")
    conversion_success = create_standard_page_images(image_dir, base_filename)
    if not conversion_success:
        logger.warning("[PAGE_CONVERT] Page image conversion failed, but continuing with LLM sorting...")
    else:
        logger.info("[PAGE_CONVERT] Page image conversion completed successfully")
    
    pages_dict = {}
    for element in merged_elements:
        page_num = element.get('page', 1)
        if page_num not in pages_dict:
            pages_dict[page_num] = []
        pages_dict[page_num].append(element)
    
    all_sorted_elements = []
    global_id = 1
    
    for page_num in sorted(pages_dict.keys()):
        page_elements = pages_dict[page_num]
        logger.info(f"Processing page {page_num} with {len(page_elements)} elements...")
        
        try:
            logger.info(f"[LLM_SORT_IMG] Attempting to load page image for page {page_num}...")
            page_image_b64 = _get_page_image_as_base64(image_dir, base_filename, page_num)
            
            has_image = page_image_b64 is not None
            has_multiple_elements = len(page_elements) > 1
            logger.info(f"[LLM_SORT_IMG] Conditions check - has_image: {has_image}, has_multiple_elements: {has_multiple_elements} ({len(page_elements)} elements)")
            
            if page_image_b64 and len(page_elements) > 1:
                logger.info(f"[LLM_SORT_IMG] Using LLM-based sorting for page {page_num}")
                sorted_page_elements = await _sort_elements_by_llm(
                    page_elements, page_image_b64, vision_llm, page_num
                )
            else:
                if not has_image:
                    reason = "no page image available"
                elif not has_multiple_elements:
                    reason = f"only {len(page_elements)} element(s)"
                else:
                    reason = "unknown reason"
                
                logger.warning(f"Page {page_num}: Using fallback sorting ({reason})")
                sorted_page_elements = _fallback_sort_elements(page_elements)
            
            for elem in sorted_page_elements:
                elem['id'] = global_id
                global_id += 1
            
            all_sorted_elements.extend(sorted_page_elements)
            
        except Exception as e:
            logger.error(f"LLM sorting failed for page {page_num}: {e}. Using fallback sorting.")
            fallback_sorted = _fallback_sort_elements(page_elements)
            for elem in fallback_sorted:
                elem['id'] = global_id
                global_id += 1
            all_sorted_elements.extend(fallback_sorted)
    
    logger.info(f"LLM-based ID assignment completed: {len(all_sorted_elements)} elements processed")
    return all_sorted_elements

async def _sort_elements_by_llm(page_elements: List[Element], page_image_b64: str, 
                               vision_llm, page_num: int) -> List[Element]:

    element_summaries = []
    for i, elem in enumerate(page_elements):
        summary = _create_element_summary(elem, i)
        element_summaries.append(summary)
    
    prompt = _create_reading_order_prompt(element_summaries, page_num)
    
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{page_image_b64}"}
        }
    ]
    
    try:
        response = await vision_llm.ainvoke([HumanMessage(content=content)])
        
        reading_order = _parse_reading_order_response(response.content, len(page_elements))
        
        sorted_elements = []
        for order_idx in reading_order:
            if 0 <= order_idx < len(page_elements):
                sorted_elements.append(page_elements[order_idx])
        
        included_indices = set(reading_order)
        for i, elem in enumerate(page_elements):
            if i not in included_indices:
                sorted_elements.append(elem)
                logger.warning(f"Added missing element {i} at end of page {page_num}")
        
        logger.info(f"Page {page_num}: LLM successfully ordered {len(sorted_elements)} elements")
        return sorted_elements
        
    except Exception as e:
        logger.error(f"LLM sorting failed for page {page_num}: {e}")
        return _fallback_sort_elements(page_elements)

def _create_element_summary(element: Element, index: int) -> str:
    category = element.get('category', 'unknown')
    
    content = element.get('content', {})
    text = content.get('text', '') or element.get('text', '')
    
    coords = element.get('coordinates', [])
    coord_str = "unknown"
    if coords:
        if isinstance(coords, list) and len(coords) > 0:
            if isinstance(coords[0], dict):
                x = coords[0].get('x', 0)
                y = coords[0].get('y', 0)
                coord_str = f"({x:.0f}, {y:.0f})"
            elif isinstance(coords[0], (int, float)) and len(coords) >= 2:
                coord_str = f"({coords[0]:.0f}, {coords[1]:.0f})"
    
    text_preview = text[:50] + "..." if len(text) > 50 else text
    text_preview = text_preview.replace('\n', ' ').strip()
    
    return f"Element {index}: {category} at {coord_str} - \"{text_preview}\""

def _create_reading_order_prompt(element_summaries: List[str], page_num: int) -> str:
    summaries_text = "\n".join(element_summaries)
    
    return f"""
TASK: Determine the natural reading order for document elements on page {page_num}.

ELEMENTS TO ORDER:
{summaries_text}

INSTRUCTIONS:
1. Analyze the page image to understand the document layout
2. Consider typical reading patterns (top-to-bottom, left-to-right for most documents)
3. Pay attention to document structure (headings before content, captions with figures, etc.)
4. For Korean documents, follow Korean reading conventions
5. Group related elements logically (e.g., figure with its caption)

OUTPUT REQUIREMENT:
Return ONLY a comma-separated list of element indices in reading order.
Example: 0,2,1,3,4

The order should reflect how a human would naturally read through this page.
Focus on logical document flow, not just coordinate positions.
"""

def _parse_reading_order_response(response_content: str, num_elements: int) -> List[int]:
    try:
        numbers = re.findall(r'\d+', response_content)
        
        order_indices = []
        for num_str in numbers:
            try:
                idx = int(num_str)
                if 0 <= idx < num_elements and idx not in order_indices:
                    order_indices.append(idx)
            except ValueError:
                continue
        
        if len(order_indices) < num_elements:
            for i in range(num_elements):
                if i not in order_indices:
                    order_indices.append(i)
        
        logger.debug(f"Parsed reading order: {order_indices}")
        return order_indices
        
    except Exception as e:
        logger.error(f"Failed to parse reading order response: {e}")
        return list(range(num_elements))

def _fallback_sort_elements(elements: List[Element]) -> List[Element]:
    def sort_key(elem):
        page = elem.get('page', 0)
        coords = elem.get('coordinates')
        if coords and isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], dict):
            y_coord = coords[0].get('y', 0)
            return (page, y_coord)
        elif coords and isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], (int, float)):
             y_coord = coords[1]
             return (page, y_coord)
        return (page, float('inf')) 

    try:
        return sorted(elements, key=sort_key)
    except (TypeError, KeyError) as e:
        logger.warning(f"Could not sort elements due to inconsistent data: {e}")
        return elements

def _get_page_image_as_base64(image_dir: str, base_filename: str, page_num: int) -> Optional[str]:

    logger.info(f"[LLM_SORT_IMG] Parameters: image_dir='{image_dir}', base_filename='{base_filename}', page_num={page_num}")
    
    base_pattern = re.match(r'^(.+?)_\d{4}_\d{4}$', base_filename)
    if base_pattern:
        true_base = base_pattern.group(1)
        logger.info(f"[LLM_SORT_IMG] Using true base name: {true_base}")
        image_filename = f"{true_base}_page_{page_num}.png"
    else:
        logger.info(f"[LLM_SORT_IMG] Using original base name: {base_filename}")
        image_filename = f"{base_filename}_page_{page_num}.png"
    
    image_path = os.path.join(image_dir, image_filename)
    logger.info(f"[LLM_SORT_IMG] Looking for page image: {image_path}")
    
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                logger.info(f"[LLM_SORT_IMG] Page {page_num}: ✓ loaded (size: {len(image_data)} chars)")
                return image_data
        except Exception as e:
            logger.warning(f"[LLM_SORT_IMG] Failed to load page image {image_path}: {e}")
            logger.warning(f"[LLM_SORT_IMG] Page {page_num}: ✗ encoding failed")
    else:
        logger.warning(f"[LLM_SORT_IMG] Page {page_num}: ✗ file not found - {image_path}")
        
        try:
            if os.path.exists(image_dir):
                files = os.listdir(image_dir)
                matching_files = [f for f in files if base_filename in f and f.endswith(('.png', '.jpg', '.jpeg'))]
                logger.info(f"[LLM_SORT_IMG] Available image files in {image_dir}: {matching_files}")
            else:
                logger.warning(f"[LLM_SORT_IMG] Image directory does not exist: {image_dir}")
        except Exception as e:
            logger.error(f"[LLM_SORT_IMG] Error listing directory {image_dir}: {e}")
    
    return None

def generate_unified_content(final_elements: List[Element]) -> Dict[str, str]:
    html_parts, md_parts, text_parts = [], [], []
    for elem in final_elements:
        content = elem.get('content', {})
        category = elem.get('category', 'paragraph')
        text = content.get('text', '')
        md = content.get('markdown')
        html = content.get('html')
        
        if not text: continue
        if not md:
            if 'heading' in category:
                level = category[-1] if category[-1].isdigit() else '1'
                md = f"{'#' * int(level)} {text}"
            else: md = text
        if not html:
            if 'heading' in category:
                level = category[-1] if category[-1].isdigit() else '1'
                html = f"<h{level}>{text}</h{level}>"
            elif category == 'table': html = content.get('html', f"<p>Table data: {text}</p>") 
            else: html = f"<p>{text}</p>"
        
        html_parts.append(html)
        md_parts.append(md)
        text_parts.append(text)
        
    unified_content = {
        "html": "<br>\n".join(filter(None, html_parts)),
        "markdown": "\n\n".join(filter(None, md_parts)),
        "text": "\n".join(filter(None, text_parts))
    }
    logger.info("Generated unified content fields (html, markdown, text).")
    return unified_content

def save_final_json(data: Dict, output_path: str):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully saved final assembled data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save final JSON file: {e}")
        raise

