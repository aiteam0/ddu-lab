import json
import os
import base64
import html
import re
import logging

from .logging_config import get_logger

logger = get_logger(__name__)
logger.info("[DOCYOLO_INIT] convert_docyolo_v3.py logger initialized - testing logging system")

def get_image_filename(img_path):
    if not img_path:
        return None
    return os.path.basename(img_path)

def normalize_bbox_and_to_coordinates(bbox, page_width, page_height):
    if not bbox or len(bbox) != 4:
        return None
    
    x_min, y_min, x_max, y_max = bbox

    norm_x_min = x_min / page_width
    norm_y_min = y_min / page_height
    norm_x_max = x_max / page_width
    norm_y_max = y_max / page_height

    coordinates = [
        {"x": norm_x_min, "y": norm_y_min},
        {"x": norm_x_max, "y": norm_y_min},
        {"x": norm_x_max, "y": norm_y_max},
        {"x": norm_x_min, "y": norm_y_max},
    ]
    return coordinates

def encode_image_to_base64(image_path):
    try:
        if not os.path.exists(image_path):
            logger.warning(f"[DOCYOLO_B64] file check: {image_path} exists=False")
            return None
        
        logger.info(f"[DOCYOLO_B64] file check: {image_path} exists=True")
        
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            file_size = len(image_data)
            b64_data = base64.b64encode(image_data).decode('utf-8')
            
            logger.info(f"[DOCYOLO_B64] encoding: {os.path.basename(image_path)} -> {len(b64_data)} chars (from {file_size} bytes)")
            return b64_data
            
    except FileNotFoundError:
        logger.warning(f"[DOCYOLO_B64] error: Image file not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"[DOCYOLO_B64] error: Error encoding image {image_path}: {e}")
        return None

def extract_text_from_html(html_string):
    if not html_string:
        return ""
    text = re.sub(r'<(style|script).*?>.*?</\1>', '', html_string, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return html.unescape(text)

def analyze_text_content_pattern(text_content):
    if not text_content:
        return "paragraph"
    
    text_stripped = text_content.strip()
    
    if re.match(r'^[-•*]\s+', text_stripped) or re.match(r'^\d+\.\s+', text_stripped):
        return "list"
    
    # 수식 패턴  
    # if re.search(r'\$.*\$|\\[a-zA-Z]+|∫|∑|∂|α|β|γ|θ|π', text_stripped):
    #     return "equation"
    
    # 각주 패턴
    if re.match(r'^\d+\)\s+|^\[\d+\]\s+|^※\s+', text_stripped):
        return "footnote"
    
    # 페이지 헤더/푸터 패턴
    if len(text_stripped) < 50 and re.search(r'\d+|page|페이지', text_stripped.lower()):
        return "header"  # 위치에 따라 footer로 변경될 수 있음
    
    return "paragraph"

def determine_enhanced_category(content_item, middle_block):
    
    content_type = content_item.get("type", "")
    text_content = content_item.get("text", "")
    text_level = content_item.get("text_level")
    
    if content_type == "title":
        return "heading1"
    elif content_type == "reference":
        return "reference"
    elif content_type == "figure":
        return "figure"
    elif content_type == "image":
        return "figure"
    elif content_type == "figure_caption":
        return "caption"
    elif content_type == "table":
        return "table"
    elif content_type == "table_caption":
        return "caption"
    elif content_type == "text":
        if text_level == 1:
            return "heading1"
        elif text_level == 2:
            return "heading2"
        elif text_level == 3:
            return "heading3"
        else:
            return analyze_text_content_pattern(text_content)
    
    return "paragraph"

def extract_image_info_with_path(content_item, middle_block, doc_filename, type_counters):
    
    img_path = content_item.get("img_path")
    if not img_path:
        return None
    
    file_extension = '.png'
    
    page_no = content_item.get("page_idx", 0) + 1
    original_type = content_item.get("type", "image")
    
    if original_type == "table":
        category_name = "table"
        type_counters["table"] = type_counters.get("table", 0) + 1
        counter = type_counters["table"]
    elif original_type in ["image", "figure", "chart"]:
        category_name = "picture"
        type_counters["picture"] = type_counters.get("picture", 0) + 1
        counter = type_counters["picture"]
    else:
        category_name = "text"
        type_counters["text"] = type_counters.get("text", 0) + 1
        counter = type_counters["text"]
    
    new_filename = f"{doc_filename}-page-{page_no}-{category_name}-{counter}{file_extension}"
    
    return {
        "original_path": img_path,
        "new_filename": new_filename,
        "image_path": new_filename,
        "needs_rename": True,
        "category_name": category_name
    }

def get_page_size_for_content(content_item, middle_data):
    page_idx = content_item.get("page_idx", 0)
    
    if page_idx < len(middle_data["pdf_info"]):
        return middle_data["pdf_info"][page_idx]["page_size"]
    
    if len(middle_data["pdf_info"]) > 0:
        return middle_data["pdf_info"][0]["page_size"]
    
    return [595.0, 842.0]

def create_perfect_coordinates_mapping_v2(content_list_data, middle_data, doc_filename):
    
    coordinates_map = {}
    type_counters = {}
    
    if not middle_data or "pdf_info" not in middle_data:
        logger.warning("Middle.json data is missing. Creating coordinates map without bbox info.")
        return coordinates_map
    
    preproc_blocks = []
    for page_info in middle_data["pdf_info"]:
        preproc_blocks.extend(page_info.get("preproc_blocks", []))
    
    logger.info(f"Perfect matching: {len(content_list_data)} content items with {len(preproc_blocks)} preproc_blocks")
    
    for i, content_item in enumerate(content_list_data):
        coordinates = None
        middle_block = None
        
        if i < len(preproc_blocks):
            middle_block = preproc_blocks[i]
            bbox = middle_block.get("bbox")
            
            if bbox and len(bbox) == 4:
                element_page_width, element_page_height = get_page_size_for_content(content_item, middle_data)
                coordinates = normalize_bbox_and_to_coordinates(bbox, element_page_width, element_page_height)
                logger.debug(f"Perfect match {i}: bbox {bbox} → coordinates {coordinates} (page_size: {element_page_width}x{element_page_height})")
            else:
                logger.warning(f"Perfect match {i}: invalid bbox {bbox}")
        else:
            logger.warning(f"Perfect match {i}: no corresponding preproc_block (index out of range)")
        
        category = determine_enhanced_category(content_item, middle_block)
        
        image_info = extract_image_info_with_path(content_item, middle_block, doc_filename, type_counters)
        
        coordinates_map[i] = {
            "coordinates": coordinates,
            "category": category,
            "content_item": content_item,
            "middle_block": middle_block,
            "image_info": image_info
        }
        
        logger.debug(f"Perfect mapping {i}: {content_item.get('type')} → {category}")
    
    logger.info(f"Perfect coordinates mapping completed: {len(coordinates_map)} items mapped")
    logger.info(f"Type counters: {type_counters}")
    return coordinates_map

def convert_content_list(content_list_data, middle_data, local_image_dir_path, content_list_file_path=None):

    upstage_result = {
        "api": "2.0",
        "model": "converted-from-content-list-v3-enhanced-fixed",
        "usage": {},
        "content": {"html": "", "markdown": "", "text": ""},
        "elements": []
    }
    
    doc_filename = "document"
    if content_list_file_path:
        base_filename = os.path.basename(content_list_file_path)
        if base_filename.endswith('_content_list.json'):
            doc_filename = base_filename.replace('_content_list.json', '').replace('.pdf', '')
            logger.info(f"Extracted doc_filename from file path: '{doc_filename}'")
    elif local_image_dir_path:
        parent_dir = os.path.dirname(local_image_dir_path)
        if parent_dir:
            for file in os.listdir(parent_dir):
                if file.endswith('_content_list.json'):
                    doc_filename = file.replace('_content_list.json', '').replace('.pdf', '')
                    logger.warning(f"Using fallback doc_filename extraction: '{doc_filename}'")
                    break

    total_pages = 0
    page_width, page_height = 595.0, 842.0
    
    if middle_data and "pdf_info" in middle_data:
        total_pages = len(middle_data["pdf_info"])
        logger.info(f"Processing {total_pages} pages from middle.json (Enhanced Version)")
        
        if total_pages > 0:
            page_info = middle_data["pdf_info"][0]
            page_size = page_info.get("page_size")
            if page_size and len(page_size) == 2:
                page_width, page_height = page_size
                logger.info(f"Using page size: {page_width} x {page_height}")
            else:
                logger.warning(f"Missing or invalid page_size. Using default: {page_width} x {page_height}")
    else:
        logger.warning("Middle.json data is missing. Using default page size and estimating pages.")
        if content_list_data:
            max_page_idx = 0
            for item in content_list_data:
                if "page_idx" in item and item["page_idx"] > max_page_idx:
                    max_page_idx = item["page_idx"]
            total_pages = max_page_idx + 1
    
    upstage_result["usage"]["pages"] = total_pages
    
    coordinates_map = create_perfect_coordinates_mapping_v2(
        content_list_data, middle_data, doc_filename
    )

    all_elements_html = []
    all_elements_markdown = []
    all_elements_text = []

    for idx, content_item in enumerate(content_list_data):
        mapping_info = coordinates_map.get(idx, {})
        
        element_upstage = {
            "id": idx,
            "page": content_item.get("page_idx", 0) + 1,
            "category": mapping_info.get("category", "paragraph"),
            "content": {"html": "", "markdown": "", "text": ""},
            "source_parser": "docyolo"
        }
        
        coordinates = mapping_info.get("coordinates")
        if coordinates:
            element_upstage["coordinates"] = coordinates
        
        image_info = mapping_info.get("image_info")
        if image_info:
            element_upstage["image_path"] = image_info["image_path"]
            
            original_path = image_info["original_path"]
            if original_path:
                filename_for_b64 = get_image_filename(original_path)
                if filename_for_b64:
                    actual_image_path = os.path.join(local_image_dir_path, filename_for_b64)
                    b64_data = encode_image_to_base64(actual_image_path)
                    if b64_data:
                        element_upstage["base64_encoding"] = b64_data
                        logger.debug(f"Base64 encoded for element {idx}: {filename_for_b64}")
                        
                        new_filename = image_info["new_filename"]
                        new_image_path = os.path.join(local_image_dir_path, new_filename)
                        try:
                            import shutil
                            shutil.copy2(actual_image_path, new_image_path)
                            logger.info(f"[FILE_SAVE] Saved image: {filename_for_b64} → {new_filename}")
                        except Exception as e:
                            logger.warning(f"[FILE_SAVE] Failed to save {new_filename}: {e}")

        item_type = content_item.get("type")
        text_content = content_item.get("text", "")
        text_level = content_item.get("text_level")
        img_path = content_item.get("img_path")
        
        md_content = ""
        html_content = ""
        plain_text_content = ""

        escaped_text = html.escape(text_content)
        category = element_upstage["category"]

        if category.startswith("heading"):
            level = int(category.replace("heading", ""))
            md_content = f"{'#' * level} {text_content}"
            html_content = f"<h{level} id='{idx}'>{escaped_text}</h{level}>"
            plain_text_content = text_content
        elif category == "list":
            md_content = text_content
            html_content = f"<ul id='{idx}'><li>{escaped_text.replace(chr(10), '</li><li>')}</li></ul>"
            plain_text_content = text_content
        elif category == "paragraph":
            md_content = text_content
            html_content = f"<p id='{idx}'>{escaped_text}</p>"
            plain_text_content = text_content
        elif category == "reference":
            md_content = text_content
            html_content = f"<p id='{idx}' data-category='reference'>{escaped_text}</p>"
            plain_text_content = text_content
        elif category == "caption":
            md_content = f"*{text_content}*"
            html_content = f"<figcaption id='{idx}'>{escaped_text}</figcaption>"
            plain_text_content = text_content
        elif category == "footnote":
            md_content = text_content
            html_content = f"<p id='{idx}' data-category='footnote'>{escaped_text}</p>"
            plain_text_content = text_content
        elif category == "header":
            md_content = text_content
            html_content = f"<header id='{idx}'>{escaped_text}</header>"
            plain_text_content = text_content
        elif category == "footer":
            md_content = text_content
            html_content = f"<footer id='{idx}'>{escaped_text}</footer>"
            plain_text_content = text_content
        elif category == "table":
            table_body_html = content_item.get("table_body", "<table></table>")
            table_caption_list = content_item.get("table_caption", [])
            caption_text = " ".join(table_caption_list)
            
            md_content = table_body_html
            
            if caption_text:
                match = re.search(r"<table.*?>", table_body_html, re.IGNORECASE)
                if match:
                    table_tag_end = match.end()
                    html_content = f"{table_body_html[:table_tag_end]}<caption id='cap_{idx}'>{html.escape(caption_text)}</caption>{table_body_html[table_tag_end:]}"
                    html_content = re.sub(r"(<table)", rf"\1 id='{idx}'", html_content, 1, flags=re.IGNORECASE)
                else:
                    html_content = re.sub(r"(<table)", rf"\1 id='{idx}'", table_body_html, 1, flags=re.IGNORECASE) if table_body_html else f"<table id='{idx}'></table>"
            else:
                html_content = re.sub(r"(<table)", rf"\1 id='{idx}'", table_body_html, 1, flags=re.IGNORECASE) if table_body_html else f"<table id='{idx}'></table>"

            plain_text_content = extract_text_from_html(table_body_html)
            if caption_text:
                plain_text_content = caption_text + "\n" + plain_text_content
        elif category in ["figure", "chart"]:
            img_caption_list = content_item.get("img_caption", [])
            caption_text = " ".join(img_caption_list) if img_caption_list else ""
            alt_text = html.escape(caption_text)
            
            display_path = element_upstage.get("image_path", img_path or "placeholder.jpg")
            md_content = f"![{alt_text}]({display_path})"
            
            if caption_text:
                html_content = f"<figure id='{idx}' data-category='{category}'><img alt=\"{alt_text}\" src=\"{html.escape(display_path)}\"/><figcaption>{html.escape(caption_text)}</figcaption></figure>"
            else:
                html_content = f"<figure id='{idx}' data-category='{category}'><img alt=\"{alt_text}\" src=\"{html.escape(display_path)}\"/></figure>"
            
            plain_text_content = caption_text
        elif category == "equation":
            latex_content = text_content
            md_content = f"$${latex_content}$$"
            html_content = f"<p id='{idx}' data-category='equation'>$${html.escape(latex_content)}$$</p>"
            plain_text_content = latex_content
        else:
            md_content = text_content
            html_content = f"<p id='{idx}' data-category='{category}'>{escaped_text}</p>"
            plain_text_content = text_content

        element_upstage["content"]["markdown"] = md_content
        element_upstage["content"]["html"] = html_content
        element_upstage["content"]["text"] = plain_text_content.strip()
        
        if category in ["figure", "chart"]:
            img_caption_list = content_item.get("img_caption", [])
            if img_caption_list:
                element_upstage["content"]["caption"] = " ".join(img_caption_list)
        elif category == "table":
            table_caption_list = content_item.get("table_caption", [])
            if table_caption_list:
                element_upstage["content"]["caption"] = " ".join(table_caption_list)

        upstage_result["elements"].append(element_upstage)

        all_elements_html.append(html_content)
        all_elements_markdown.append(md_content)
        all_elements_text.append(plain_text_content.strip())

    upstage_result["content"]["html"] = "<br>".join(all_elements_html)
    upstage_result["content"]["markdown"] = "\n\n".join(all_elements_markdown)
    upstage_result["content"]["text"] = "\n".join(all_elements_text)
    
    removed_b64_count = 0
    removed_coord_count = 0
    final_b64_count = 0
    
    for elem in upstage_result["elements"]:
        if "base64_encoding" in elem and elem["base64_encoding"] is None:
            del elem["base64_encoding"]
            removed_b64_count += 1
        elif "base64_encoding" in elem:
            final_b64_count += 1
            
        if "coordinates" in elem and elem["coordinates"] is None:
            del elem["coordinates"]
            removed_coord_count += 1
    
    logger.info(f"[DOCYOLO_B64] cleanup: removed {removed_b64_count} None base64_encoding fields")
    logger.info(f"[DOCYOLO_B64] cleanup: removed {removed_coord_count} None coordinates fields")
    logger.info(f"[DOCYOLO_B64] final: {final_b64_count} elements with base64 data")


    return upstage_result

if __name__ == "__main__":
    from .logging_config import setup_logging
    setup_logging(log_level="INFO")
    
    logger.info("[DOCYOLO_B64] Starting convert_docyolo_v3.py in standalone mode")
    
    local_image_dir = r"E:\MyProject2\H-DDU\docyolo_output\images"
    local_dir = r"E:\MyProject2\H-DDU\docyolo_output"

    content_list_file = os.path.join(local_dir,"content_list.json")
    middle_file = os.path.join(local_dir,"middle.json")

    output_file_name = os.path.basename(content_list_file).replace("_content_list.json", "_docyolo_to_result.json")
    output_file = os.path.join(local_dir,output_file_name)

    try:
        with open(content_list_file, 'r', encoding='utf-8') as f_cl:
            content_list_data = json.load(f_cl)
        logger.info(f"Successfully loaded content_list.json: {content_list_file}")
    except Exception as e:
        logger.error(f"Error loading content_list.json: {e}")
        content_list_data = []

    try:
        with open(middle_file, 'r', encoding='utf-8') as f_mid:
            middle_data = json.load(f_mid)
        logger.info(f"Successfully loaded middle.json: {middle_file}")
    except Exception as e:
        logger.error(f"Error loading middle.json: {e}")
        middle_data = {}

    if content_list_data:
        logger.info(f"[DOCYOLO_B64] Starting conversion with {len(content_list_data)} content items")
        converted_result = convert_content_list(content_list_data, middle_data, local_image_dir)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(converted_result, f_out, ensure_ascii=False, indent=4)
            logger.info(f"Successfully converted and saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving converted result: {e}")
    else:
        logger.warning("No data from content_list.json to process.")