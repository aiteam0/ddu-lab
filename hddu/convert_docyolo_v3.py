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
        {"x": round(norm_x_min, 4), "y": round(norm_y_min, 4)},
        {"x": round(norm_x_max, 4), "y": round(norm_y_min, 4)},
        {"x": round(norm_x_max, 4), "y": round(norm_y_max, 4)},
        {"x": round(norm_x_min, 4), "y": round(norm_y_max, 4)},
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

def convert_content_list(content_list_data, middle_data, local_image_dir_path):
    upstage_result = {
        "api": "2.0",
        "model": "converted-from-content-list-v2",
        "usage": {},
        "content": {"html": "", "markdown": "", "text": ""},
        "elements": []
    }

    coordinates_map = {}
    total_pages = 0
    if middle_data and "pdf_info" in middle_data:
        total_pages = len(middle_data["pdf_info"])
        logger.info(f"Processing {total_pages} pages from middle.json")
        for page_info in middle_data["pdf_info"]:
            page_idx = page_info.get("page_idx", 0)
            page_size = page_info.get("page_size")
            if not page_size or len(page_size) != 2:
                logger.warning(f"Page {page_idx}: Missing or invalid page_size. Coordinates may be incorrect.")
                page_width, page_height = 1.0, 1.0
            else:
                page_width, page_height = page_size

            for item_type in ["images", "tables"]:
                items = page_info.get(item_type, [])
                logger.debug(f"Page {page_idx}: Processing {len(items)} {item_type}")
                for item in items:
                    bbox = item.get("bbox")
                    logger.debug(f"Item on page {page_idx}: type={item.get('type')}, bbox={bbox}")
                    img_path_in_middle = None
                    
                    try:
                        if item.get("blocks"):
                            for block in item["blocks"]:
                                if block.get("lines"):
                                    for line in block["lines"]:
                                        if line.get("spans"):
                                            for span in line["spans"]:
                                                if span.get("type") == "image" or span.get("type") == "table":
                                                    img_path_in_middle = span.get("image_path")
                                                    logger.debug(f"Found image_path in middle.json: {img_path_in_middle} on page {page_idx}")
                                                    break
                                            if img_path_in_middle: break
                                        if img_path_in_middle: break
                                if img_path_in_middle: break
                    except Exception as e:
                        logger.error(f"Error while searching for image_path on page {page_idx}: {e}")
                        logger.debug(f"Item structure that caused error: {item}")
                    
                    logger.debug(f"Condition check: img_path_in_middle={img_path_in_middle is not None}, bbox={bbox is not None}")
                    if img_path_in_middle and bbox:
                        logger.debug(f"Processing image: {img_path_in_middle}, bbox: {bbox}")
                        filename = get_image_filename(img_path_in_middle)
                        logger.debug(f"Extracted filename: {filename}")
                        if filename:
                            logger.debug(f"Normalizing bbox {bbox} with page_size ({page_width}, {page_height})")
                            coords = normalize_bbox_and_to_coordinates(bbox, page_width, page_height)
                            logger.debug(f"Normalized coords: {coords}")
                            if coords:
                                if filename in coordinates_map:
                                    logger.warning(f"Duplicate filename found: {filename}. Previous: page {coordinates_map[filename]['page_idx_middle']}, Current: page {page_idx}")
                                    unique_key = f"{filename}_page_{page_idx}"
                                    coordinates_map[unique_key] = {
                                        "coordinates": coords,
                                        "page_idx_middle": page_idx,
                                        "original_filename": filename
                                    }
                                    logger.info(f"Added coordinates for {unique_key} (bbox: {bbox})")
                                else:
                                    coordinates_map[filename] = {
                                        "coordinates": coords,
                                        "page_idx_middle": page_idx
                                    }
                                    logger.info(f"Added coordinates for {filename} on page {page_idx} (bbox: {bbox})")
                            else:
                                logger.warning(f"Failed to normalize bbox {bbox} for {img_path_in_middle}")
                        else:
                            logger.warning(f"Failed to extract filename from {img_path_in_middle}")
                    else:
                        if not img_path_in_middle:
                            logger.debug(f"No image_path found in {item_type} item on page {page_idx}")
                        if not bbox:
                            logger.debug(f"No bbox found in {item_type} item on page {page_idx}: {item}")
                            logger.debug(f"Item structure: {item}")
        
        logger.info(f"Total coordinates_map entries: {len(coordinates_map)}")
        logger.debug(f"Coordinates map keys: {list(coordinates_map.keys())}")
    else:
        logger.warning("Middle.json data is missing or invalid. Coordinates and page count might be inaccurate.")
        if content_list_data:
            max_page_idx = 0
            for item in content_list_data:
                if "page_idx" in item and item["page_idx"] > max_page_idx:
                    max_page_idx = item["page_idx"]
            total_pages = max_page_idx + 1


    upstage_result["usage"]["pages"] = total_pages

    all_elements_html = []
    all_elements_markdown = []
    all_elements_text = []

    for idx, content_item in enumerate(content_list_data):
        element_upstage = {
            "id": idx,
            "page": content_item.get("page_idx", 0) + 1,
            "category": "",
            "content": {"html": "", "markdown": "", "text": ""},
            "coordinates": None,
        }

        item_type = content_item.get("type")
        text_content = content_item.get("text", "")
        text_level = content_item.get("text_level")
        img_path = content_item.get("img_path")
        
        if item_type == "text":
            if text_level:
                element_upstage["category"] = f"heading{text_level}"
            elif text_content.strip().startswith(("- ", "* ", "• ")) or re.match(r"^\d+\.\s", text_content.strip()):
                element_upstage["category"] = "list"
            else:
                element_upstage["category"] = "paragraph"
        elif item_type == "table":
            element_upstage["category"] = "table"
        elif item_type == "image":
            element_upstage["category"] = "figure"
        elif item_type == "equation":
            element_upstage["category"] = "equation"
        else:
            element_upstage["category"] = "unknown"
            logger.warning(f"Unknown item type: {item_type} for item id {idx}")

        element_upstage["source_parser"] = "docyolo"

        if element_upstage["category"] in ["figure", "table"] and img_path:
            filename_cl = get_image_filename(img_path)
            if filename_cl:
                logger.debug(f"Looking for coordinates for {filename_cl} (from img_path: {img_path}) on page {content_item.get('page_idx', 0)}")
                
                coords_found = False
                if filename_cl in coordinates_map:
                    if coordinates_map[filename_cl]["page_idx_middle"] == content_item.get("page_idx", 0):
                        element_upstage["coordinates"] = coordinates_map[filename_cl]["coordinates"]
                        coords_found = True
                        logger.debug(f"Found coordinates for {filename_cl} using exact match")
                    else:
                        logger.warning(f"Page index mismatch for image {filename_cl}: content_list has {content_item.get('page_idx', 0)}, middle.json has {coordinates_map[filename_cl]['page_idx_middle']}. Trying alternative matching.")
                
                if not coords_found:
                    page_specific_key = f"{filename_cl}_page_{content_item.get('page_idx', 0)}"
                    if page_specific_key in coordinates_map:
                        element_upstage["coordinates"] = coordinates_map[page_specific_key]["coordinates"]
                        coords_found = True
                        logger.debug(f"Found coordinates for {filename_cl} using page-specific key: {page_specific_key}")
                
                if not coords_found:
                    for key, coord_info in coordinates_map.items():
                        if (key.startswith(filename_cl) and 
                            "_page_" in key and 
                            coord_info.get("original_filename") == filename_cl):
                            element_upstage["coordinates"] = coord_info["coordinates"]
                            coords_found = True
                            logger.warning(f"Found coordinates for {filename_cl} using fallback match with key: {key} (page mismatch)")
                            break
                
                if not coords_found:
                    logger.warning(f"Coordinates not found in middle.json for image: {filename_cl} (from content_list img_path: {img_path})")
                    logger.debug(f"Available coordinates_map keys: {list(coordinates_map.keys())}")
            else:
                logger.warning(f"Could not extract filename from img_path: {img_path}")


        if element_upstage["category"] in ["figure", "table"] and img_path:
            logger.info(f"[DOCYOLO_B64] processing: {element_upstage['category']} element {idx} with img_path: {img_path}")
            
            filename_for_b64 = get_image_filename(img_path)
            if filename_for_b64:
                actual_image_path = os.path.join(local_image_dir_path, filename_for_b64)
                logger.info(f"[DOCYOLO_B64] path: {actual_image_path}")
                
                b64_data = encode_image_to_base64(actual_image_path)
                if b64_data:
                    element_upstage["base64_encoding"] = b64_data
                    logger.info(f"[DOCYOLO_B64] result: success for {filename_for_b64}")
                else:
                    element_upstage["base64_encoding"] = None 
                    logger.warning(f"[DOCYOLO_B64] result: failure for {filename_for_b64} (encoding returned None)")
            else:
                element_upstage["base64_encoding"] = None
                logger.warning(f"[DOCYOLO_B64] result: failure - filename extraction failed from img_path: {img_path}")
        elif element_upstage["category"] in ["figure", "table"]:
            logger.info(f"[DOCYOLO_B64] skip: {element_upstage['category']} element {idx} has no img_path")


        md_content = ""
        html_content = ""
        plain_text_content = ""

        escaped_text = html.escape(text_content)

        if element_upstage["category"].startswith("heading"):
            level = int(element_upstage["category"].replace("heading", ""))
            md_content = f"{'#' * level} {text_content}"
            html_content = f"<h{level} id='{idx}'>{escaped_text}</h{level}>"
            plain_text_content = text_content
        elif element_upstage["category"] == "list":
            md_content = text_content
            html_content = f"<p id='{idx}' data-category='list'>{escaped_text}</p>"
            plain_text_content = text_content
        elif element_upstage["category"] == "paragraph":
            md_content = text_content
            html_content = f"<p id='{idx}' data-category='paragraph'>{escaped_text}</p>"
            plain_text_content = text_content
        elif element_upstage["category"] == "table":
            table_body_html = content_item.get("table_body", "<table></table>")
            table_caption_list = content_item.get("table_caption", [])
            caption_text = " ".join(table_caption_list)
            
            md_content = table_body_html
            
            if caption_text:
                if "<table>" in table_body_html.lower():
                    table_body_with_caption = table_body_html.lower().replace("<table>", f"<table><caption id='cap_{idx}'>{html.escape(caption_text)}</caption>", 1)
                    match = re.search(r"<table.*?>", table_body_html, re.IGNORECASE)
                    if match:
                        table_tag_end = match.end()
                        html_content = f"{table_body_html[:table_tag_end]}<caption id='cap_{idx}'>{html.escape(caption_text)}</caption>{table_body_html[table_tag_end:]}"
                        html_content = re.sub(r"(<table)", rf"\1 id='{idx}'", html_content, 1, flags=re.IGNORECASE)
                    else:
                         html_content = re.sub(r"(<table)", rf"\1 id='{idx}'", table_body_html, 1, flags=re.IGNORECASE) if table_body_html else f"<table id='{idx}'></table>"

                else:
                    html_content = f"<table id='{idx}'><caption id='cap_{idx}'>{html.escape(caption_text)}</caption>{table_body_html}</table>"
            else:
                html_content = re.sub(r"(<table)", rf"\1 id='{idx}'", table_body_html, 1, flags=re.IGNORECASE) if table_body_html else f"<table id='{idx}'></table>"

            plain_text_content = extract_text_from_html(table_body_html)
            if caption_text:
                plain_text_content = caption_text + "\n" + plain_text_content

        elif element_upstage["category"] == "figure":
            img_caption_list = content_item.get("img_caption", [])
            caption_text = " ".join(img_caption_list) if img_caption_list else ""
            alt_text = html.escape(caption_text)
            md_img_path = img_path if img_path else "placeholder.jpg"
            md_content = f"![{alt_text}]({md_img_path})"
            
            if caption_text:
                html_content = f"<figure id='{idx}'><img alt=\"{alt_text}\" src=\"{html.escape(img_path if img_path else '')}\"/><figcaption>{html.escape(caption_text)}</figcaption></figure>"
            else:
                html_content = f"<figure id='{idx}'><img alt=\"{alt_text}\" src=\"{html.escape(img_path if img_path else '')}\"/></figure>"
            
            plain_text_content = caption_text
        elif element_upstage["category"] == "equation":
            latex_content = text_content
            md_content = f"$${latex_content}$$"
            html_content = f"<p id='{idx}' data-category='equation'>$${html.escape(latex_content)}$$</p>"
            plain_text_content = latex_content

        element_upstage["content"]["markdown"] = md_content
        element_upstage["content"]["html"] = html_content
        element_upstage["content"]["text"] = plain_text_content.strip()
        
        if element_upstage["category"] == "figure":
            img_caption_list = content_item.get("img_caption", [])
            if img_caption_list:
                element_upstage["content"]["caption"] = " ".join(img_caption_list)
        elif element_upstage["category"] == "table":
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
    output_file_name = os.path.basename(content_list_file).replace("_content_list.json", "_docyolo_to_upstage_result.json")
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