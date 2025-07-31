import logging
import os
import json
import base64
import re
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from langchain_core.messages import HumanMessage
from .config import Element
from ..logging_config import get_logger

logger = get_logger(__name__)
logger.info("[ASSEMBLY_INIT] postprocessor.py logger initialized - testing logging system")

def _extract_enhanced_coordinates(element: Element) -> Dict[str, Any]:

    coords = element.get('coordinates', [])
    
    result = {
        'bbox': None,
        'center': None,
        'size': None,
        'top_left': None,
        'area': 0
    }
    
    if not coords or not isinstance(coords, list) or len(coords) == 0:
        return result
    
    try:
        if isinstance(coords[0], dict):
            if len(coords) >= 2:
                x1, y1 = coords[0].get('x', 0), coords[0].get('y', 0)
                x2, y2 = coords[-1].get('x', 0), coords[-1].get('y', 0)
            else:
                x1, y1 = coords[0].get('x', 0), coords[0].get('y', 0)
                x2, y2 = x1, y1
        elif isinstance(coords[0], (int, float)):
            if len(coords) >= 4:
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            elif len(coords) >= 2:
                x1, y1 = coords[0], coords[1] 
                x2, y2 = x1, y1
            else:
                x1, y1, x2, y2 = coords[0], 0, coords[0], 0
        else:
            return result
            
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        area = width * height
        
        result.update({
            'bbox': (min_x, min_y, max_x, max_y),
            'center': (center_x, center_y),
            'size': (width, height),
            'top_left': (min_x, min_y),
            'area': area
        })
        
    except Exception as e:
        logger.warning(f"Failed to extract enhanced coordinates: {e}")
    
    return result

def _create_enhanced_element_summary(element: Element, index: int, detail_level: str = "full") -> str:

    category = element.get('category', 'unknown')
    
    content = element.get('content', {})
    text = content.get('text', '') or element.get('text', '')
    
    coord_info = _extract_enhanced_coordinates(element)
    
    text_limits = {
        'full': {'heading1': 200, 'heading2': 150, 'heading3': 100, 'paragraph': 150, 'table': 100, 'caption': 150, 'figure': 80, 'list': 120},
        'medium': {'heading1': 100, 'heading2': 80, 'heading3': 60, 'paragraph': 80, 'table': 60, 'caption': 80, 'figure': 40, 'list': 60},
        'minimal': {'heading1': 50, 'heading2': 40, 'heading3': 30, 'paragraph': 40, 'table': 30, 'caption': 40, 'figure': 20, 'list': 30}
    }
    
    text_limit = text_limits.get(detail_level, text_limits['full']).get(category, 50)
    
    text_preview = text[:text_limit] + "..." if len(text) > text_limit else text
    text_preview = text_preview.replace('\n', ' ').replace('\r', ' ').strip()
    
    if coord_info['bbox']:
        bbox = coord_info['bbox']
        center = coord_info['center'] 
        size = coord_info['size']
        coord_str = f"bbox({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}) center({center[0]:.0f},{center[1]:.0f}) size({size[0]:.0f}x{size[1]:.0f})"
    else:
        coord_str = "coordinates_unknown"
    
    structure_info = ""
    if category == 'table' and 'table' in content:
        table_data = content.get('table', {})
        if isinstance(table_data, dict):
            rows = table_data.get('rows', 0) or len(table_data.get('data', []))
            cols = table_data.get('cols', 0) 
            if rows or cols:
                structure_info = f" [{rows}행x{cols}열]"
    elif category == 'list':
        lines = text.count('\n') + 1 if text else 0
        if lines > 1:
            structure_info = f" [{lines}개항목]"
    
    return f"Element {index}: {category}{structure_info} {coord_str} - \"{text_preview}\""

def _create_clustering_prompt(element_summaries: List[str], page_num: int) -> str:
    summaries_text = "\n".join(element_summaries)
    
    return f"""
TASK: Analyze the document page image and group related elements into logical reading clusters.

DOCUMENT ELEMENTS TO CLUSTER:
{summaries_text}

CLUSTERING GUIDELINES:
1. **Visual Analysis**: Examine the page layout carefully to identify visual groupings
2. **Spatial Relationships**: Consider proximity, alignment, and visual separators
3. **Document Structure**: Group by logical sections (title, main content, sidebar, footer)
4. **Reading Flow**: Korean documents typically flow top-to-bottom, left-to-right
5. **Content Relationships**: Group related elements (figure + caption, table + description)

CLUSTER TYPES TO CONSIDER:
- title_section: Page title, headers, main headings
- main_content: Primary body text, main paragraphs
- left_column: Left column content in multi-column layouts
- right_column: Right column content in multi-column layouts  
- sidebar: Supplementary information, boxes, callouts
- figure_group: Images/charts with their captions
- table_group: Tables with their descriptions
- footer: Bottom page information, references, footnotes

READING PRIORITY RULES:
1 = Read first (titles, main headings)
2 = Read second (primary content, left column)
3 = Read third (secondary content, right column) 
4 = Read last (footers, references)

OUTPUT REQUIREMENT:
Return a JSON object with this exact structure:
{{
    "clusters": [
        {{
            "cluster_id": "A",
            "cluster_type": "title_section",
            "elements": [0, 1],
            "reading_priority": 1,
            "spatial_description": "top center of page"
        }}
    ]
}}

IMPORTANT: 
- Every element must be assigned to exactly one cluster
- Use sequential cluster_ids: A, B, C, D, etc.
- Reading priority must be integers 1-4
- Focus on natural reading flow, not just coordinate positions
"""

def _create_cluster_sorting_prompt(element_summaries: List[str], cluster_id: str, cluster_type: str, page_num: int) -> str:
    summaries_text = "\n".join(element_summaries)
    
    return f"""
TASK: Determine the optimal reading order for elements within a specific document cluster.

CLUSTER INFORMATION:
- Cluster ID: {cluster_id}
- Cluster Type: {cluster_type}
- Page: {page_num}

ELEMENTS TO SORT:
{summaries_text}

SORTING GUIDELINES FOR CLUSTER TYPE '{cluster_type}':
1. **Visual Layout**: Analyze the spatial arrangement in the page image
2. **Reading Conventions**: Follow Korean document reading patterns
3. **Content Hierarchy**: Consider heading levels, content importance
4. **Logical Flow**: Maintain narrative or informational sequence

TYPE-SPECIFIC RULES:
- title_section: Main title before subtitles, larger headings first
- main_content: Top-to-bottom, paragraph flow order
- left_column/right_column: Top-to-bottom within the column
- sidebar: Priority by visual emphasis and relevance
- figure_group: Figure before caption, main image before supporting images
- table_group: Table before description/notes
- footer: Left-to-right, then top-to-bottom

SPATIAL ANALYSIS:
- Same vertical level: Left-to-right order
- Same horizontal level: Top-to-bottom order
- Mixed layout: Follow natural eye movement pattern

OUTPUT REQUIREMENT:
Return a JSON object with this exact structure:
{{
    "cluster_id": "{cluster_id}",
    "sorted_elements": [0, 2, 1, 3],
    "confidence": 0.95,
    "reasoning": "brief explanation of sorting logic"
}}

IMPORTANT:
- Return element indices in optimal reading order
- Confidence should be 0.0-1.0 (high confidence for clear layouts)
- All original elements must be included
- Consider both spatial position and semantic relationships
"""

def _estimate_token_count(text: str) -> int:

    korean_chars = len([c for c in text if ord(c) >= 0xAC00 and ord(c) <= 0xD7A3])
    other_chars = len(text) - korean_chars
    return int(korean_chars / 2 + other_chars / 4)

def _manage_token_budget(page_elements: List[Element], max_tokens: int = 6000) -> Tuple[str, List[str]]:

    detail_levels = ["full", "medium", "minimal"]
    
    for detail_level in detail_levels:
        element_summaries = []
        total_tokens = 0
        
        base_prompt_tokens = 800
        total_tokens += base_prompt_tokens
        
        for i, elem in enumerate(page_elements):
            summary = _create_enhanced_element_summary(elem, i, detail_level=detail_level)
            summary_tokens = _estimate_token_count(summary)
            
            if total_tokens + summary_tokens > max_tokens:
                logger.warning(f"[TOKEN_MGMT] Token limit reached at element {i} with detail_level '{detail_level}' "
                              f"(total: {total_tokens + summary_tokens} > {max_tokens})")
                break
                
            element_summaries.append(summary)
            total_tokens += summary_tokens
        
        if len(element_summaries) == len(page_elements):
            logger.info(f"[TOKEN_MGMT] Using detail_level '{detail_level}' "
                       f"(estimated tokens: {total_tokens}/{max_tokens})")
            return detail_level, element_summaries
    
    logger.warning(f"[TOKEN_MGMT] All detail levels exceeded token limit, forcing minimal with truncation")
    element_summaries = []
    total_tokens = 800
    
    for i, elem in enumerate(page_elements):
        summary = _create_enhanced_element_summary(elem, i, detail_level="minimal")
        summary_tokens = _estimate_token_count(summary)
        
        if total_tokens + summary_tokens > max_tokens:
            logger.warning(f"[TOKEN_MGMT] Hard token limit reached, processing only {i}/{len(page_elements)} elements")
            break
            
        element_summaries.append(summary)
        total_tokens += summary_tokens
    
    return "minimal", element_summaries

def _parse_clustering_response(response_content: str, num_elements: int) -> Dict[str, Any]:

    try:
        import re
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            if 'clusters' in result and isinstance(result['clusters'], list):
                assigned_elements = set()
                for cluster in result['clusters']:
                    if 'elements' in cluster:
                        assigned_elements.update(cluster['elements'])
                
                missing_elements = set(range(num_elements)) - assigned_elements
                if missing_elements:
                    logger.warning(f"Adding missing elements to fallback cluster: {missing_elements}")
                    result['clusters'].append({
                        'cluster_id': 'FALLBACK',
                        'cluster_type': 'main_content',
                        'elements': list(missing_elements),
                        'reading_priority': 99,
                        'spatial_description': 'fallback cluster'
                    })
                
                return result
                
    except Exception as e:
        logger.error(f"Failed to parse clustering response: {e}")
    
    return {
        'clusters': [{
            'cluster_id': 'A',
            'cluster_type': 'main_content', 
            'elements': list(range(num_elements)),
            'reading_priority': 1,
            'spatial_description': 'fallback single cluster'
        }]
    }

def _parse_sorting_response(response_content: str, num_elements: int, cluster_id: str) -> Dict[str, Any]:

    try:
        import re
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            if 'sorted_elements' in result and isinstance(result['sorted_elements'], list):
                sorted_elements = result['sorted_elements']
                
                valid_elements = [idx for idx in sorted_elements if 0 <= idx < num_elements]
                missing_elements = [i for i in range(num_elements) if i not in valid_elements]
                
                final_order = valid_elements + missing_elements
                
                return {
                    'cluster_id': cluster_id,
                    'sorted_elements': final_order,
                    'confidence': result.get('confidence', 0.5),
                    'reasoning': result.get('reasoning', 'LLM parsing successful')
                }
                
    except Exception as e:
        logger.error(f"Failed to parse sorting response: {e}")
    
    return {
        'cluster_id': cluster_id,
        'sorted_elements': list(range(num_elements)),
        'confidence': 0.1,
        'reasoning': 'fallback to original order due to parsing failure'
    }

async def _cluster_elements_by_llm(page_elements: List[Element], page_image_b64: str, 
                                   vision_llm, page_num: int) -> Dict[str, Any]:

    logger.info(f"[CLUSTERING] Starting clustering for page {page_num} with {len(page_elements)} elements")
    
    try:
        detail_level, element_summaries = _manage_token_budget(page_elements, max_tokens=6000)
        logger.info(f"[CLUSTERING] Using detail_level '{detail_level}' for {len(element_summaries)} elements")
        
        prompt = _create_clustering_prompt(element_summaries, page_num)
        
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{page_image_b64}"}
            }
        ]
        
        logger.info(f"[CLUSTERING] Calling LLM for clustering page {page_num}")
        response = await vision_llm.ainvoke([HumanMessage(content=content)])
        
        clustering_result = _parse_clustering_response(response.content, len(page_elements))
        
        logger.info(f"[CLUSTERING] Page {page_num}: Found {len(clustering_result['clusters'])} clusters")
        for cluster in clustering_result['clusters']:
            logger.info(f"[CLUSTERING]   Cluster {cluster['cluster_id']} ({cluster['cluster_type']}): "
                       f"{len(cluster['elements'])} elements, priority {cluster['reading_priority']}")
        
        return clustering_result
        
    except Exception as e:
        logger.error(f"[CLUSTERING] Failed for page {page_num}: {e}")
        return {
            'clusters': [{
                'cluster_id': 'ERROR',
                'cluster_type': 'main_content',
                'elements': list(range(len(page_elements))),
                'reading_priority': 1,
                'spatial_description': 'error fallback cluster'
            }]
        }

async def _sort_cluster_by_llm(cluster_elements: List[Element], page_image_b64: str,
                               vision_llm, cluster_id: str, cluster_type: str, page_num: int) -> List[Element]:

    logger.info(f"[CLUSTER_SORT] Sorting cluster {cluster_id} ({cluster_type}) with {len(cluster_elements)} elements")
    
    if len(cluster_elements) <= 1:
        logger.info(f"[CLUSTER_SORT] Cluster {cluster_id}: Single element, no sorting needed")
        return cluster_elements
    
    try:
        detail_level, element_summaries = _manage_token_budget(cluster_elements, max_tokens=4000)
        logger.info(f"[CLUSTER_SORT] Using detail_level '{detail_level}' for cluster {cluster_id}")
        
        prompt = _create_cluster_sorting_prompt(element_summaries, cluster_id, cluster_type, page_num)
        
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{page_image_b64}"}
            }
        ]
        
        logger.info(f"[CLUSTER_SORT] Calling LLM for sorting cluster {cluster_id}")
        response = await vision_llm.ainvoke([HumanMessage(content=content)])
        
        sorting_result = _parse_sorting_response(response.content, len(cluster_elements), cluster_id)
        
        sorted_elements = []
        for order_idx in sorting_result['sorted_elements']:
            if 0 <= order_idx < len(cluster_elements):
                sorted_elements.append(cluster_elements[order_idx])
        
        logger.info(f"[CLUSTER_SORT] Cluster {cluster_id}: Sorted {len(sorted_elements)} elements "
                   f"(confidence: {sorting_result['confidence']:.2f})")
        logger.debug(f"[CLUSTER_SORT] Cluster {cluster_id} reasoning: {sorting_result['reasoning']}")
        
        return sorted_elements
        
    except Exception as e:
        logger.error(f"[CLUSTER_SORT] Failed for cluster {cluster_id}: {e}")
        return finalize_elements(cluster_elements, sort_mode="integrated")

async def _sort_page_with_clustering(page_elements: List[Element], page_image_b64: str, 
                                     vision_llm, page_num: int) -> List[Element]:

    logger.info(f"[PAGE_CLUSTERING] Starting 2-stage clustering sort for page {page_num} with {len(page_elements)} elements")
    
    try:
        logger.info(f"[PAGE_CLUSTERING] Stage 1: Clustering elements for page {page_num}")
        clustering_result = await _cluster_elements_by_llm(page_elements, page_image_b64, vision_llm, page_num)
        
        logger.info(f"[PAGE_CLUSTERING] Stage 2: Sorting clusters by reading priority for page {page_num}")
        final_sorted_elements = []
        
        sorted_clusters = sorted(clustering_result["clusters"], key=lambda x: x.get("reading_priority", 999))
        
        for cluster in sorted_clusters:
            cluster_id = cluster["cluster_id"]
            cluster_type = cluster["cluster_type"]
            element_indices = cluster["elements"]
            
            cluster_elements = []
            for idx in element_indices:
                if 0 <= idx < len(page_elements):
                    cluster_elements.append(page_elements[idx])
                else:
                    logger.warning(f"[PAGE_CLUSTERING] Invalid element index {idx} in cluster {cluster_id}")
            
            if not cluster_elements:
                logger.warning(f"[PAGE_CLUSTERING] Empty cluster {cluster_id}, skipping")
                continue
            
            logger.info(f"[PAGE_CLUSTERING] Processing cluster {cluster_id} ({cluster_type}) with {len(cluster_elements)} elements")
            
            if len(cluster_elements) > 1:
                sorted_cluster = await _sort_cluster_by_llm(
                    cluster_elements, page_image_b64, vision_llm, cluster_id, cluster_type, page_num
                )
            else:
                sorted_cluster = cluster_elements
                logger.info(f"[PAGE_CLUSTERING] Cluster {cluster_id}: Single element, no sorting needed")
            
            final_sorted_elements.extend(sorted_cluster)
            logger.info(f"[PAGE_CLUSTERING] Added {len(sorted_cluster)} elements from cluster {cluster_id}")
        
        if len(final_sorted_elements) != len(page_elements):
            logger.error(f"[PAGE_CLUSTERING] Element count mismatch: expected {len(page_elements)}, got {len(final_sorted_elements)}")
            processed_ids = {id(elem) for elem in final_sorted_elements}
            for elem in page_elements:
                if id(elem) not in processed_ids:
                    final_sorted_elements.append(elem)
                    logger.warning(f"[PAGE_CLUSTERING] Added missing element to end")
        
        logger.info(f"[PAGE_CLUSTERING] Page {page_num}: Clustering-based sorting completed successfully "
                   f"({len(final_sorted_elements)} elements)")
        return final_sorted_elements
        
    except Exception as e:
        logger.error(f"[PAGE_CLUSTERING] Clustering-based sorting failed for page {page_num}: {e}")
        logger.info(f"[PAGE_CLUSTERING] Falling back to coordinate-based sorting for page {page_num}")
        return finalize_elements(page_elements, sort_mode="integrated")

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
                shutil.copy2(original_path, standard_path)
                logger.info(f"[PAGE_CONVERT] Copied: {file_info['original_name']} → {standard_filename} (page {file_info['actual_page']})")
                
                conversion_count += 1
                
            except Exception as e:
                logger.error(f"[PAGE_CONVERT] Failed to create standard file {standard_filename}: {e}")
        
        logger.info(f"[PAGE_CONVERT] Conversion completed: {conversion_count}/{len(found_files)} files converted")
        return conversion_count > 0
        
    except Exception as e:
        logger.error(f"[PAGE_CONVERT] Error during page image conversion: {e}")
        return False

def finalize_elements(merged_elements: List[Element], sort_mode: str = "disabled") -> List[Element]:

    
    def extract_coordinates_and_category(elem):

        coords = elem.get('coordinates', [])
        category = elem.get('category', 'paragraph')
        
        y_coord = 0
        x_coord = 0
        
        if coords and isinstance(coords, list) and len(coords) > 0:
            if isinstance(coords[0], dict):
                y_coord = coords[0].get('y', 0)
                x_coord = coords[0].get('x', 0)
            elif len(coords) >= 2:
                x_coord = coords[0] if isinstance(coords[0], (int, float)) else 0
                y_coord = coords[1] if isinstance(coords[1], (int, float)) else 0
            else:
                x_coord = coords[0] if isinstance(coords[0], (int, float)) else 0
        
        normalized_y = round(y_coord / 0.02) * 0.02
        normalized_x = round(x_coord / 0.02) * 0.02
        
        category_priority = {
            'heading1': 1, 'heading2': 2, 'heading3': 3,
            'paragraph': 4, 'list': 5, 'table': 6, 
            'figure': 7, 'chart': 8, 'equation': 9,
            'caption': 10, 'footnote': 11, 'header': 12,
            'footer': 13, 'reference': 14
        }
        category_order = category_priority.get(category, 15)
        
        return normalized_y, normalized_x, category_order
    
    def sort_key_disabled(elem):
        page = elem.get('page', 0)
        parser_type = elem.get('source_parser', 'unknown') 
        element_id = elem.get('id', 0)
        
        normalized_y, normalized_x, category_order = extract_coordinates_and_category(elem)
        
        return (page, parser_type, element_id, normalized_y, normalized_x, category_order)
    
    def sort_key_integrated(elem):
        page = elem.get('page', 0)
        element_id = elem.get('id', 0)
        
        normalized_y, normalized_x, category_order = extract_coordinates_and_category(elem)
        
        return (page, element_id, normalized_y, normalized_x, category_order)
    
    if sort_mode == "disabled":
        sort_key = sort_key_disabled
        sort_description = "(page, parser_type, element_id, coordinates, priority)"
    elif sort_mode == "integrated":
        sort_key = sort_key_integrated
        sort_description = "(page, element_id, coordinates, priority)"
    else:
        logger.warning(f"Unknown sort_mode: {sort_mode}. Using disabled mode.")
        sort_key = sort_key_disabled
        sort_description = "(page, parser_type, element_id, coordinates, priority)"
    
    sorted_elements = sorted(merged_elements, key=sort_key)

    for i, elem in enumerate(sorted_elements):
        elem['id'] = i
        
    logger.info(f"Finalized and re-indexed {len(sorted_elements)} elements with {sort_mode} mode sorting {sort_description}.")
    return sorted_elements

async def finalize_elements_with_llm(merged_elements: List[Element], vision_llm, 
                                   image_dir: str, base_filename: str, use_async: bool = True, 
                                   llm_mode: str = "simple") -> List[Element]:

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
                if llm_mode == "advanced":
                    logger.info(f"[LLM_SORT_IMG] Using advanced clustering-based sorting for page {page_num}")
                    sorted_page_elements = await _sort_page_with_clustering(
                        page_elements, page_image_b64, vision_llm, page_num
                    )
                else:
                    logger.info(f"[LLM_SORT_IMG] Using simple coordinate-based sorting for page {page_num}")
                    sorted_page_elements = finalize_elements(page_elements, sort_mode="integrated")
            else:
                if not has_image:
                    reason = "no page image available"
                elif not has_multiple_elements:
                    reason = f"only {len(page_elements)} element(s)"
                else:
                    reason = "unknown reason"
                
                logger.warning(f"Page {page_num}: Using fallback sorting ({reason})")
                sorted_page_elements = finalize_elements(page_elements, sort_mode="integrated")
            
            for elem in sorted_page_elements:
                elem['id'] = global_id
                global_id += 1
            
            all_sorted_elements.extend(sorted_page_elements)
            
        except Exception as e:
            logger.error(f"LLM sorting failed for page {page_num}: {e}. Using fallback sorting.")
            fallback_sorted = finalize_elements(page_elements, sort_mode="integrated")
            for elem in fallback_sorted:
                elem['id'] = global_id
                global_id += 1
            all_sorted_elements.extend(fallback_sorted)
    
    logger.info(f"LLM-based ID assignment completed: {len(all_sorted_elements)} elements processed")
    return all_sorted_elements

# 초기버전 : LLM 기반 정렬. 현재는 사용안함
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
        return finalize_elements(page_elements, sort_mode="integrated")

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


def _get_page_image_as_base64(image_dir: str, base_filename: str, page_num: int) -> Optional[str]:

    logger.info(f"[LLM_SORT_IMG] Parameters: image_dir='{image_dir}', base_filename='{base_filename}', page_num={page_num}")
    
    base_pattern = re.match(r'^(.+?)_(\d{4})_\d{4}$', base_filename)
    if base_pattern:
        true_base = base_pattern.group(1)
        start_page = int(base_pattern.group(2))
        actual_page = start_page + page_num 
        logger.info(f"[LLM_SORT_IMG] Using true base name: {true_base}")
        logger.info(f"[LLM_SORT_IMG] Calculated actual page: {start_page} + {page_num} = {actual_page}")
        image_filename = f"{true_base}_page_{actual_page}.png"
    else:
        logger.info(f"[LLM_SORT_IMG] Using original base name: {base_filename}")
        image_filename = f"{base_filename}_page_{page_num}.png"
        actual_page = page_num
    
    image_path = os.path.join(image_dir, image_filename)
    logger.info(f"[LLM_SORT_IMG] Looking for page image: {image_path}")
    
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                logger.info(f"[LLM_SORT_IMG] Page {actual_page}: ✓ loaded (size: {len(image_data)} chars)")
                return image_data
        except Exception as e:
            logger.warning(f"[LLM_SORT_IMG] Failed to load page image {image_path}: {e}")
            logger.warning(f"[LLM_SORT_IMG] Page {actual_page}: ✗ encoding failed")
    else:
        logger.warning(f"[LLM_SORT_IMG] Page {actual_page}: ✗ file not found - {image_path}")
        
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

