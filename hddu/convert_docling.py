import argparse
import json
import pandas as pd
import os
import re
import base64
import logging
from typing import List, Dict, Any, Tuple, Optional
try:
    from markdownify import markdownify as md
except ImportError:
    def md(html_content, **kwargs):
        return re.sub(r'<[^>]+>', '', html_content)

from .logging_config import get_logger

logger = get_logger(__name__)

CATEGORY_MAPPING_SEGMENTS = {
    "section_header": "heading1",
    "text": "paragraph",
    "list_item": "list", 
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "header",
    "page_footer": "footer",
    "table": "table",
    "picture": "figure",
    "equation": "equation",
}

def convert_docling_bbox_to_upstage_coords(
    docling_bbox: List[float] 
) -> List[Dict[str, float]]:
    if not docling_bbox or len(docling_bbox) != 4:
        return []
    x1, y1, x2, y2 = docling_bbox
    return [
        {"x": x1, "y": y1},{"x": x2, "y": y1},
        {"x": x2, "y": y2},{"x": x1, "y": y2},
    ]

def generate_pixel_data_coord(
    bbox_norm: List[float], page_width_pts: float, page_height_pts: float, dpi: int
) -> str:
    if not bbox_norm or len(bbox_norm) != 4:
        return ""
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_norm
    page_width_px = page_width_pts * (dpi / 72.0)
    page_height_px = page_height_pts * (dpi / 72.0)
    px_left = x1_norm * page_width_px
    px_top = y1_norm * page_height_px
    px_right = x2_norm * page_width_px
    px_bottom = y2_norm * page_height_px
    return f"top-left:({px_left},{px_top}); bottom-right:({px_right},{px_bottom})"

def ocr_to_latex_simple(ocr_text: str) -> str:
    return f"$${ocr_text}$$"

def encode_image_to_base64(image_path: str) -> Optional[str]:
    if not image_path:
        logger.warning("Image path is empty")
        return None
    
    if not os.path.exists(image_path):
        logger.warning(f"Image file not found: '{image_path}'")
        logger.debug(f"Current working directory: {os.getcwd()}")
        
        parent_dir = os.path.dirname(image_path)
        if parent_dir and os.path.exists(parent_dir):
            try:
                files = os.listdir(parent_dir)
                logger.debug(f"Contents of directory '{parent_dir}': {files}")
                
                base_name = os.path.basename(image_path)
                similar_files = [f for f in files if base_name.lower() in f.lower() or f.lower() in base_name.lower()]
                if similar_files:
                    logger.debug(f"Similar files found: {similar_files}")
                    
            except Exception as e:
                logger.debug(f"Could not list directory '{parent_dir}': {e}")
        
        return None
    
    try:
        file_size = os.path.getsize(image_path)
        logger.debug(f"Encoding image: '{image_path}' (size: {file_size} bytes)")
        
        with open(image_path, "rb") as image_file:
            encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
            logger.debug(f"Successfully encoded image to base64 (length: {len(encoded_data)})")
            return encoded_data
            
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def get_abs_image_path_robust(docling_pipeline_output_dir: str, path_in_parquet: str) -> str:
    if not path_in_parquet:
        logger.warning("Empty path_in_parquet provided")
        return ""
    
    logger.debug(f"Converting image path: '{path_in_parquet}' with output_dir: '{docling_pipeline_output_dir}'")
    
    if os.path.isabs(path_in_parquet):
        if os.path.exists(path_in_parquet):
            logger.debug(f"Using absolute path: '{path_in_parquet}'")
            return path_in_parquet
        else:
            logger.warning(f"Absolute path does not exist: '{path_in_parquet}'")
    
    possible_paths = [
        os.path.join(docling_pipeline_output_dir, path_in_parquet),
        os.path.join(docling_pipeline_output_dir, os.path.basename(path_in_parquet)),
        os.path.join(os.getcwd(), path_in_parquet),
        path_in_parquet,
    ]
    
    for i, path in enumerate(possible_paths):
        normalized_path = os.path.normpath(path)
        logger.debug(f"Trying path option {i+1}: '{normalized_path}'")
        
        if os.path.exists(normalized_path):
            logger.debug(f"Found valid path: '{normalized_path}'")
            return normalized_path
    
    logger.error(f"Image file not found in any of the expected paths: {possible_paths}")
    logger.debug(f"Current working directory: {os.getcwd()}")
    
    if os.path.exists(docling_pipeline_output_dir):
        try:
            files = os.listdir(docling_pipeline_output_dir)
            logger.debug(f"Contents of output directory '{docling_pipeline_output_dir}': {files}")
        except Exception as e:
            logger.debug(f"Could not list output directory: {e}")
    
    return ""

def get_abs_image_path(docling_pipeline_output_dir: str, path_in_parquet: str) -> str:

    return get_abs_image_path_robust(docling_pipeline_output_dir, path_in_parquet)

def html_table_to_markdown(html_content: str) -> str:
    if not html_content:
        return ""
    try:
        return md(html_content, heading_style='atx', bullets='-')
    except Exception as e:
        logger.warning(f"Could not convert HTML table to Markdown: {e}")
        return f"```html\n{html_content}\n```"

def create_markdown_for_element(text: str, category: str, alt_text: Optional[str] = None, image_path_relative: Optional[str] = None, table_html_content: Optional[str] = None) -> str:
    if category.startswith("heading"):
        level = 1
        try:
            level = int(category.replace("heading", "")[0])
            if level < 1: level = 1
        except (ValueError, IndexError): pass
        return f"{'#' * level} {text}"
    elif category == "list":
        return "".join(f"- {line}\n" for line in text.splitlines() if line.strip())
    elif category == "figure":
        path_to_use = image_path_relative if image_path_relative else "/placeholder_image.png"
        return f"![{alt_text if alt_text else 'figure'}]({path_to_use})"
    elif category == "table":
        if table_html_content:
            return html_table_to_markdown(table_html_content)
        elif text.strip():
             return f"Table Data (OCR Text):\n{text}"
        return "| Header | ... |\n|---|---|\n| Data | ... |"
    elif category == "equation":
        return ocr_to_latex_simple(text)
    return text

def create_html_for_element(text: str, category: str, element_id: int, data_coord_str: str = "", table_html_seq: Optional[str] = None, alt_text: Optional[str] = None) -> str:
    html_tag = "p"
    additional_attrs = ""
    content_to_render = text

    if category.startswith("heading"):
        level = 1
        try:
            level = int(category.replace("heading", "")[0])
            if level < 1: level = 1
        except (ValueError, IndexError): pass
        html_tag = f"h{level}"
    elif category == "list":
        html_tag = "ul"
        content_to_render = "".join(f"<li>{line}</li>" for line in text.splitlines() if line.strip())
    elif category == "figure":
        html_tag = "figure"
        content_to_render = f"<img alt=\"{alt_text if alt_text else 'figure'}\" data-coord=\"{data_coord_str}\" />"
    elif category == "table":
        html_tag = "table"
        if table_html_seq:
            content_to_render = table_html_seq
        else:
            content_to_render = f"<caption>Table from segment text (ID: {element_id})</caption><thead><tr><th>Content</th></tr></thead><tbody><tr><td>{text if text.strip() else 'N/A'}</td></tr></tbody>"
    elif category == "equation":
        additional_attrs = " data-category='equation'"
        content_to_render = ocr_to_latex_simple(text)
    elif category == "caption": html_tag = "caption"
    elif category == "header": html_tag = "header"
    elif category == "footer": html_tag = "footer"
    elif category not in ["paragraph"]:
        additional_attrs = f" data-category='{category}'"
    
    return f"<{html_tag} id='{element_id}'{additional_attrs}>{content_to_render}</{html_tag}>"

def convert_docling_result(docling_parquet_path: str, local_output_dir: str):
    logger.info(f"Starting docling result conversion from: {docling_parquet_path}")
    logger.info(f"Output directory: {local_output_dir}")
    
    if not os.path.exists(docling_parquet_path):
        logger.error(f"Input parquet file not found: {docling_parquet_path}")
        raise FileNotFoundError(f"Input file not found: {docling_parquet_path}")

    try:
        df = pd.read_parquet(docling_parquet_path)
        logger.info(f"Successfully loaded parquet with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading parquet file {docling_parquet_path}: {e}")
        raise

    docling_pipeline_output_dir = os.path.dirname(docling_parquet_path)
    logger.debug(f"Using pipeline output directory: {docling_pipeline_output_dir}")

    json_cols_to_process = ['segments_json', 'image_paths_json', 'extra_json', 'tables_json', 'figures_json']
    
    for col in json_cols_to_process:
        if col in df.columns:
            is_list_type = col in ['segments_json', 'tables_json', 'figures_json']
            default_empty = [] if is_list_type else {}
            
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else (x if isinstance(x, (dict, list)) else default_empty)
            )
        else:
            logger.warning(f"Column {col} not found, creating default empty values")
            df[col] = [([] if col in ['segments_json', 'tables_json', 'figures_json'] else {}) for _ in range(len(df))]
    
    all_upstage_elements_temp: List[Dict[str, Any]] = []
    
    total_figure_count = 0
    total_table_count = 0
    total_base64_success_count = 0
    total_base64_fail_count = 0

    for page_idx, page_row in df.iterrows():
        logger.debug(f"Processing page {page_idx + 1}/{len(df)}")
        
        page_extra_json = page_row.get('extra_json') or {}
        page_no_upstage = page_extra_json.get('page_no', page_idx + 1)
        page_width_pts = page_extra_json.get('width_in_points', 595.0)
        page_height_pts = page_extra_json.get('height_in_points', 842.0)
        page_dpi = page_extra_json.get('dpi', 150)

        segments = page_row.get('segments_json', [])
        image_paths_data = page_row.get('image_paths_json', {})
        
        page_figures_from_json = page_row.get('figures_json', [])
        page_tables_from_json = page_row.get('tables_json', [])

        logger.debug(f"Page {page_idx}: {len(segments)} segments, {len(page_figures_from_json)} figures, {len(page_tables_from_json)} tables")

        used_figure_paths_from_segments = set()
        used_table_paths_from_segments = set()
        processed_figure_ids = set()
        processed_table_ids = set()
        
        temp_list_items_for_grouping = []
        
        seg_figure_idx = 0
        seg_table_idx = 0

        for segment_idx, segment in enumerate(segments):
            docling_label = segment.get("label", "text")
            segment_text = segment.get("text", "").strip()
            segment_bbox = segment.get("bbox")
            segment_data_field = segment.get("data") 

            upstage_category_mapped = CATEGORY_MAPPING_SEGMENTS.get(docling_label, "paragraph")

            if docling_label == "list_item":
                temp_list_items_for_grouping.append(segment)
                continue
            
            if temp_list_items_for_grouping:
                list_texts = [item.get("text","").strip() for item in temp_list_items_for_grouping]
                combined_list_text = "\n".join(filter(None, list_texts))
                list_bboxes_norm = [item.get("bbox") for item in temp_list_items_for_grouping if item.get("bbox")]
                list_overall_bbox_norm = None
                if list_bboxes_norm:
                    min_x1 = min(b[0] for b in list_bboxes_norm); min_y1 = min(b[1] for b in list_bboxes_norm)
                    max_x2 = max(b[2] for b in list_bboxes_norm); max_y2 = max(b[3] for b in list_bboxes_norm)
                    list_overall_bbox_norm = [min_x1, min_y1, max_x2, max_y2]

                if combined_list_text or list_overall_bbox_norm:
                    all_upstage_elements_temp.append({
                        "category": "list", "text_content": combined_list_text, "bbox_norm": list_overall_bbox_norm,
                        "page_no_upstage": page_no_upstage, "page_width_pts": page_width_pts,
                        "page_height_pts": page_height_pts, "page_dpi": page_dpi,
                        "alt_text": "list", "image_path_relative": None,
                        "table_html_from_segment_data": None,
                        "source_parser": "docling"
                    })
                temp_list_items_for_grouping = []

            base64_img_data = None
            table_html_from_segment_data = None
            current_image_path_relative = None
            alt_text_for_element = segment_text if segment_text else docling_label

            if upstage_category_mapped == "figure":
                total_figure_count += 1
                figure_images_list = image_paths_data.get("figure_images", [])
                if seg_figure_idx < len(figure_images_list):
                    current_image_path_relative = figure_images_list[seg_figure_idx]
                    logger.debug(f"Processing figure image: '{current_image_path_relative}' (segment {segment_idx})")
                    
                    abs_img_path = get_abs_image_path_robust(docling_pipeline_output_dir, current_image_path_relative)
                    
                    if abs_img_path:
                        base64_img_data = encode_image_to_base64(abs_img_path)
                        if base64_img_data:
                            logger.debug(f"Successfully encoded figure image: '{current_image_path_relative}'")
                            total_base64_success_count += 1
                        else:
                            logger.warning(f"Failed to encode figure image: '{current_image_path_relative}'")
                            total_base64_fail_count += 1
                    else:
                        logger.warning(f"Could not find figure image file: '{current_image_path_relative}'")
                        total_base64_fail_count += 1
                    
                    used_figure_paths_from_segments.add(current_image_path_relative)
                    if current_image_path_relative:
                        processed_figure_ids.add(current_image_path_relative)
                    seg_figure_idx += 1
                else:
                    logger.debug(f"No more figure images available for segment {segment_idx}")
                    if segment_text:
                        matching_fig = next((f_item for f_item in page_figures_from_json 
                                           if f_item.get("id") and segment_text in str(f_item.get("id", ""))), None)
                        if matching_fig and matching_fig.get("image_paths"):
                            current_image_path_relative = matching_fig["image_paths"][0]
                            abs_img_path = get_abs_image_path_robust(docling_pipeline_output_dir, current_image_path_relative)
                            if abs_img_path:
                                base64_img_data = encode_image_to_base64(abs_img_path)
                                if base64_img_data:
                                    total_base64_success_count += 1
                                    logger.debug(f"Found matching figure in figures_json: {current_image_path_relative}")
                                else:
                                    total_base64_fail_count += 1
                            else:
                                total_base64_fail_count += 1
                    
                alt_text_for_element = f"Figure (from segment, label: {docling_label})"
            
            elif upstage_category_mapped == "table":
                total_table_count += 1
                table_images_list = image_paths_data.get("table_images", [])
                if seg_table_idx < len(table_images_list):
                    current_image_path_relative = table_images_list[seg_table_idx]
                    logger.debug(f"Processing table image: '{current_image_path_relative}' (segment {segment_idx})")
                    
                    abs_img_path = get_abs_image_path_robust(docling_pipeline_output_dir, current_image_path_relative)
                    
                    if abs_img_path:
                        base64_img_data = encode_image_to_base64(abs_img_path)
                        if base64_img_data:
                            logger.debug(f"Successfully encoded table image: '{current_image_path_relative}'")
                            total_base64_success_count += 1
                        else:
                            logger.warning(f"Failed to encode table image: '{current_image_path_relative}'")
                            total_base64_fail_count += 1
                    else:
                        logger.warning(f"Could not find table image file: '{current_image_path_relative}'")
                        total_base64_fail_count += 1
                    
                    used_table_paths_from_segments.add(current_image_path_relative)
                    if current_image_path_relative:
                        processed_table_ids.add(current_image_path_relative)
                    seg_table_idx += 1
                else:
                    logger.debug(f"No more table images available for segment {segment_idx}")
                    if segment_text:
                        matching_tbl = next((t_item for t_item in page_tables_from_json 
                                           if t_item.get("id") and segment_text in str(t_item.get("id", ""))), None)
                        if matching_tbl and matching_tbl.get("image_paths"):
                            current_image_path_relative = matching_tbl["image_paths"][0]
                            abs_img_path = get_abs_image_path_robust(docling_pipeline_output_dir, current_image_path_relative)
                            if abs_img_path:
                                base64_img_data = encode_image_to_base64(abs_img_path)
                                if base64_img_data:
                                    total_base64_success_count += 1
                                    logger.debug(f"Found matching table in tables_json: {current_image_path_relative}")
                                else:
                                    total_base64_fail_count += 1
                            else:
                                total_base64_fail_count += 1
                
                if isinstance(segment_data_field, list) and segment_data_field and isinstance(segment_data_field[0], dict):
                    table_html_from_segment_data = segment_data_field[0].get('html_seq')
                alt_text_for_element = f"Table (from segment, label: {docling_label})"
            
            if not segment_text and not base64_img_data and upstage_category_mapped not in ["figure", "table", "list"]:
                continue

            all_upstage_elements_temp.append({
                "category": upstage_category_mapped, "text_content": segment_text, "bbox_norm": segment_bbox,
                "page_no_upstage": page_no_upstage, "base64_img_data": base64_img_data,
                "table_html_from_segment_data": table_html_from_segment_data,
                "page_width_pts": page_width_pts, "page_height_pts": page_height_pts, "page_dpi": page_dpi,
                "alt_text": alt_text_for_element, "image_path_relative": current_image_path_relative,
                "source_parser": "docling"
            })

        if temp_list_items_for_grouping:
            list_texts = [item.get("text","").strip() for item in temp_list_items_for_grouping]
            combined_list_text = "\n".join(filter(None, list_texts))
            list_bboxes_norm = [item.get("bbox") for item in temp_list_items_for_grouping if item.get("bbox")]
            list_overall_bbox_norm = None
            if list_bboxes_norm:
                min_x1 = min(b[0] for b in list_bboxes_norm); min_y1 = min(b[1] for b in list_bboxes_norm)
                max_x2 = max(b[2] for b in list_bboxes_norm); max_y2 = max(b[3] for b in list_bboxes_norm)
                list_overall_bbox_norm = [min_x1, min_y1, max_x2, max_y2]
            if combined_list_text or list_overall_bbox_norm:
                all_upstage_elements_temp.append({
                    "category": "list", "text_content": combined_list_text, "bbox_norm": list_overall_bbox_norm,
                    "page_no_upstage": page_no_upstage, "page_width_pts": page_width_pts,
                    "page_height_pts": page_height_pts, "page_dpi": page_dpi,
                    "alt_text": "list", "image_path_relative": None,
                    "table_html_from_segment_data": None,
                    "source_parser": "docling"
                })
        
        figure_paths_in_page = image_paths_data.get("figure_images", [])
        missing_figure_count = 0
        for fig_path_relative in figure_paths_in_page:
            if fig_path_relative not in used_figure_paths_from_segments:
                found_fig_data = next((f_item for f_item in page_figures_from_json if f_item.get("image_paths") and fig_path_relative in f_item.get("image_paths")), None)
                if found_fig_data:
                    fig_id = found_fig_data.get('id', fig_path_relative)
                    if fig_id not in processed_figure_ids:
                        missing_figure_count += 1
                        processed_figure_ids.add(fig_id)
                        logger.debug(f"Processing missing figure: '{fig_path_relative}' (ID: {fig_id})")
                        
                        fig_bbox_norm = found_fig_data.get("bbox")
                        predicted_class = found_fig_data.get('predicted_class', 'image')
                        
                        abs_img_path = get_abs_image_path_robust(docling_pipeline_output_dir, fig_path_relative)
                        base64_data = None
                        if abs_img_path:
                            base64_data = encode_image_to_base64(abs_img_path)
                            if base64_data:
                                total_base64_success_count += 1
                            else:
                                total_base64_fail_count += 1
                        else:
                            total_base64_fail_count += 1
                        
                        all_upstage_elements_temp.append({
                            "category": "chart" if predicted_class == "chart" else "figure", "text_content": "", "bbox_norm": fig_bbox_norm,
                            "page_no_upstage": page_no_upstage, "base64_img_data": base64_data,
                            "page_width_pts": page_width_pts, "page_height_pts": page_height_pts, "page_dpi": page_dpi,
                            "alt_text": f"{predicted_class.capitalize()} (from figures_json)", "image_path_relative": fig_path_relative,
                            "table_html_from_segment_data": None,
                            "source_parser": "docling"
                        })
                    else:
                        logger.debug(f"Skipping duplicate figure ID: {fig_id} (path: {fig_path_relative})")
                    
        if missing_figure_count > 0:
            logger.debug(f"Added {missing_figure_count} missing figures from page {page_idx}")

        table_paths_in_page = image_paths_data.get("table_images", [])
        missing_table_count = 0
        for tbl_path_relative in table_paths_in_page:
            if tbl_path_relative not in used_table_paths_from_segments:
                found_tbl_data = next((t_item for t_item in page_tables_from_json if t_item.get("image_paths") and tbl_path_relative in t_item.get("image_paths")), None)
                if found_tbl_data:
                    tbl_id = found_tbl_data.get('id', tbl_path_relative)
                    if tbl_id not in processed_table_ids:
                        missing_table_count += 1
                        processed_table_ids.add(tbl_id)
                        logger.debug(f"Processing missing table: '{tbl_path_relative}' (ID: {tbl_id})")
                        
                        tbl_bbox_norm = found_tbl_data.get("bbox")
                        
                        abs_img_path = get_abs_image_path_robust(docling_pipeline_output_dir, tbl_path_relative)
                        base64_data = None
                        if abs_img_path:
                            base64_data = encode_image_to_base64(abs_img_path)
                            if base64_data:
                                total_base64_success_count += 1
                            else:
                                total_base64_fail_count += 1
                        else:
                            total_base64_fail_count += 1
                        
                        table_text = f"Table (from tables_json, image: {os.path.basename(tbl_path_relative)})"
                        all_upstage_elements_temp.append({
                            "category": "table", "text_content": table_text, "bbox_norm": tbl_bbox_norm,
                            "page_no_upstage": page_no_upstage, "base64_img_data": base64_data,
                            "page_width_pts": page_width_pts, "page_height_pts": page_height_pts, "page_dpi": page_dpi,
                            "alt_text": "Table (from tables_json)", "image_path_relative": tbl_path_relative,
                            "table_html_from_segment_data": None,
                            "source_parser": "docling"
                        })
                    else:
                        logger.debug(f"Skipping duplicate table ID: {tbl_id} (path: {tbl_path_relative})")
                    
        if missing_table_count > 0:
            logger.debug(f"Added {missing_table_count} missing tables from page {page_idx}")

    logger.info(f"Total elements collected: {len(all_upstage_elements_temp)}")
    logger.info(f"Image processing statistics:")
    logger.info(f"  - Total figures: {total_figure_count}")
    logger.info(f"  - Total tables: {total_table_count}")
    logger.info(f"  - Base64 encoding success: {total_base64_success_count}")
    logger.info(f"  - Base64 encoding failed: {total_base64_fail_count}")
    
    if total_base64_fail_count > 0:
        logger.warning(f"Base64 encoding failed for {total_base64_fail_count} images. Check logs for details.")

    all_upstage_elements_temp.sort(key=lambda el: (
        el['page_no_upstage'], 
        el['bbox_norm'][1] if el.get('bbox_norm') else float('inf'),
        el['bbox_norm'][0] if el.get('bbox_norm') else float('inf')
    ))

    final_upstage_elements = []
    for i, temp_el in enumerate(all_upstage_elements_temp):
        data_coord_str = ""
        if temp_el.get("bbox_norm") and temp_el["category"] in ["figure", "chart"]:
            data_coord_str = generate_pixel_data_coord(temp_el["bbox_norm"], temp_el["page_width_pts"], temp_el["page_height_pts"], temp_el["page_dpi"])

        html_content = create_html_for_element(
            temp_el["text_content"], temp_el["category"], i, data_coord_str,
            temp_el.get("table_html_from_segment_data"), temp_el.get("alt_text")
        )
        markdown_content = create_markdown_for_element(
            temp_el["text_content"], temp_el["category"], temp_el.get("alt_text"),
            temp_el.get("image_path_relative"), temp_el.get("table_html_from_segment_data")
        )
        
        element = {
            "category": temp_el["category"],
            "content": {"text": temp_el["text_content"], "markdown": markdown_content, "html": html_content},
            "coordinates": convert_docling_bbox_to_upstage_coords(temp_el["bbox_norm"]) if temp_el.get("bbox_norm") else [],
            "id": i,
            "page": temp_el["page_no_upstage"],
            "source_parser": "docling"
        }
        if temp_el.get("base64_img_data"):
            element["base64_encoding"] = temp_el["base64_img_data"]
        final_upstage_elements.append(element)

    full_html_content = [elem["content"]["html"] for elem in final_upstage_elements]
    full_markdown_content = [elem["content"]["markdown"] for elem in final_upstage_elements]
    full_text_content = [elem["content"]["text"] for elem in final_upstage_elements]

    final_html = "\n<br>".join(full_html_content)
    final_markdown = "\n\n".join(full_markdown_content)
    final_text = "\n".join(full_text_content)

    upstage_result = {
        "api": "2.0",
        "model": "docling-segments-combined-v1.3", 
        "usage": {"pages": len(df)},
        "content": {"html": final_html, "markdown": final_markdown, "text": final_text},
        "elements": final_upstage_elements,
    }

    logger.info(f"Conversion completed successfully")
    logger.info(f"Final result summary:")
    logger.info(f"  - Total pages: {len(df)}")
    logger.info(f"  - Total elements: {len(final_upstage_elements)}")
    logger.info(f"  - Elements with base64 encoding: {len([e for e in final_upstage_elements if 'base64_encoding' in e])}")

    return upstage_result 