import requests
import json
import os
import time
import datetime
import logging
import pandas as pd
from pathlib import Path
from .base import BaseNode
from .state import ParseState

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

from docling.datamodel.base_models import InputFormat, DocItemLabel, OcrCell
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.export import generate_multimodal_pages
from docling.utils.utils import create_hash
from docling_core.types.doc import PictureItem, TableItem, SectionHeaderItem

from .convert_docyolo_v3 import convert_content_list
from .convert_docling import convert_docling_result

from .assembly.main_assembler import DocumentAssembler
from .assembly import config as assembly_config 


class DocumentParseNode(BaseNode):
    def __init__(self, lang="auto", verbose=False, **kwargs):
        """
        DocumentParse 클래스의 생성자

        :lang: 파싱 언어 설정 (ko, en, cn, auto)
        """
        super().__init__(verbose=verbose, **kwargs)
        self.lang = lang
        
        try:
            self.assembler = DocumentAssembler()
            self.assembly_enabled = True
            self.log("DocumentAssembler initialized successfully. Assembly feature is ON.")
        except Exception as e:
            self.assembler = None
            self.assembly_enabled = False
            self.log(f"Failed to initialize DocumentAssembler: {e}. Assembly feature is OFF.", level=logging.ERROR)

    def _layout_analysis(self, input_file):

        output_dir = "intermediate"
        os.makedirs(output_dir, exist_ok=True)
        
        input_filename = os.path.basename(input_file)
        base_filename = os.path.splitext(input_filename)[0]
        
        docyolo_response = None
        docling_response = None
        docyolo_output_file = None
        docling_output_file = None
        
        try:
            self.log("DocYOLO 분석 시작...")
            docyolo_response = self._analyze_with_docyolo(input_file)
            
            docyolo_output_file = os.path.join(output_dir, base_filename + "_docyolo.json")
            with open(docyolo_output_file, "w", encoding="utf-8") as f:
                json.dump(docyolo_response, f, ensure_ascii=False, indent=2)
            self.log(f"DocYOLO 결과 저장: {docyolo_output_file}")
            
        except Exception as e:
            self.log(f"DocYOLO 분석 실패: {str(e)}")

        try:
            self.log("Docling 분석 시작...")
            docling_response = self._analyze_with_docling(input_file)
            
            docling_output_file = os.path.join(output_dir, base_filename + "_docling.json")
            with open(docling_output_file, "w", encoding="utf-8") as f:
                json.dump(docling_response, f, ensure_ascii=False, indent=2)
            self.log(f"Docling 결과 저장: {docling_output_file}")
            
        except Exception as e:
            self.log(f"Docling 분석 실패: {str(e)}")

        if docling_output_file and docyolo_output_file and self.assembly_enabled:
            try:
                self.log("Docling과 DocYOLO 분석 모두 성공. 결과 병합 시작...")
                assembled_output_file = os.path.join(output_dir, base_filename + "_assembled.json")
                
                self.assembler.run(
                    docling_path=docling_output_file,
                    docyolo_path=docyolo_output_file,
                    output_path=assembled_output_file,
                    use_async=True
                )
                
                self.log(f"최종 결과 저장 (Assembled): {assembled_output_file}")
                return assembled_output_file
            except Exception as e:
                self.log(f"결과 병합(Assembly) 실패: {e}. Fallback 로직으로 전환합니다.", level=logging.ERROR)
                
        if docling_output_file:
            self.log(f"최종 결과로 Docling 결과 사용: {docling_output_file}")
            return docling_output_file
            
        elif docyolo_output_file:
            self.log(f"최종 결과로 DocYOLO 결과 사용 (Docling 실패): {docyolo_output_file}")
            return docyolo_output_file
            
        else:
            raise ValueError(f"DocYOLO와 Docling 분석 모두 실패")

    def _analyze_with_docyolo(self, input_file):
        try:
            name_without_suff = os.path.basename(input_file)
            
            base_dir = "docyolo_output"
            local_image_dir = os.path.join(base_dir, "images")
            local_md_dir = base_dir
            
            os.makedirs(local_image_dir, exist_ok=True)
            
            image_dir = "images"
            
            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(local_md_dir)
            
            reader = FileBasedDataReader("")
            pdf_bytes = reader.read(input_file)
            
            ds = PymuDocDataset(pdf_bytes, lang=self.lang)
            
            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)
            
            content_list_content = pipe_result.get_content_list(image_dir)
            middle_json_content = pipe_result.get_middle_json()
            
            pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

            pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

            infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

            pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

            pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

            pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

            with open(os.path.join(local_md_dir, f"{name_without_suff}_content_list.json"), "r", encoding="utf-8") as f:
                content_list_content = json.load(f)

            with open(os.path.join(local_md_dir, f"{name_without_suff}_middle.json"), "r", encoding="utf-8") as f:
                middle_json_content = json.load(f)

            converted_result = convert_content_list(
                content_list_content, 
                middle_json_content, 
                local_image_dir
            )
            
            self.log(f"DocYOLO 분석 완료: {input_file}")
            return converted_result
            
        except Exception as e:
            self.log(f"DocYOLO 분석 중 오류 발생: {str(e)}")
            raise ValueError(f"DocYOLO 분석 실패: {str(e)}")

    def _analyze_with_docling(self, input_file):
        try:
            name_without_suff = os.path.basename(input_file)
            doc_filename = os.path.splitext(name_without_suff)[0]
            
            base_dir = "docling_output"
            output_dir = Path(base_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            IMAGE_RESOLUTION_SCALE = 2.0
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.ocr_options.lang = ["ko", "en"]
            pipeline_options.ocr_options.use_gpu = False
            
            self.log(f"Docling 분석 시작: {input_file}")
            self.log(f"파이프라인 옵션: 이미지 스케일={pipeline_options.images_scale}, OCR 언어={pipeline_options.ocr_options.lang}")
            
            conv_res = None
            
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
            from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
            
            backends_to_try = [
                #("PyPdfium2Backend", PyPdfiumDocumentBackend),
                #("DoclingParseV2Backend", DoclingParseV2DocumentBackend),
                ("DoclingParseBackend", DoclingParseDocumentBackend),
                ("Default", None),
            ]
            
            for backend_name, backend_class in backends_to_try:
                try:
                    self.log(f"Docling 백엔드 시도: {backend_name}")
                    
                    # 백엔드별 포맷 옵션 설정
                    if backend_name == "PyPdfium2Backend":
                        format_option = PdfFormatOption(
                            pipeline_options=pipeline_options,
                            backend=backend_class
                        )

                    elif backend_name == "DoclingParseV2Backend":
                        format_option = PdfFormatOption(
                            pipeline_options=pipeline_options,
                            backend=backend_class
                        )

                    elif backend_name == "DoclingParseBackend":
                        format_option = PdfFormatOption(
                            pipeline_options=pipeline_options,
                            backend=backend_class
                        )

                    else:
                        format_option = PdfFormatOption(pipeline_options=pipeline_options)
                    
                    doc_converter = DocumentConverter(
                        format_options={
                            InputFormat.PDF: format_option
                        }
                    )
                    
                    conv_res = doc_converter.convert(input_file)
                    self.log(f"Docling 백엔드 변환 성공: {backend_name}")
                    break
                    
                except Exception as e:
                    self.log(f"Docling 백엔드 {backend_name} 실패: {str(e)}")
                    continue
                
                if conv_res is None:
                    raise ValueError("모든 백엔드에서 변환 실패")
                
            self.log(f"문서 변환 완료: 총 {len(conv_res.pages)}개 페이지")
            
            page_image_paths = {}
            table_items_by_page = {}
            picture_items_by_page = {}
            
            for page_no, page in enumerate(conv_res.pages):
                if page.image:
                    page_image_filename = output_dir / f"{doc_filename}-page-{page_no+1}.png"
                    page.image.save(page_image_filename, format="PNG")
                    page_image_paths[page_no] = str(page_image_filename)
                
                table_items_by_page[page_no] = []
                picture_items_by_page[page_no] = []
            
            total_table_count = 0
            total_picture_count = 0
            
            for element, _level in conv_res.document.iterate_items():
                try:
                    if isinstance(element, TableItem):
                        total_table_count += 1
                        page_no = self._extract_page_number(element)
                        if page_no is not None and page_no in table_items_by_page:
                            table_items_by_page[page_no].append(element)
                            
                    elif isinstance(element, PictureItem):
                        total_picture_count += 1
                        page_no = self._extract_page_number(element)
                        if page_no is not None and page_no in picture_items_by_page:
                            picture_items_by_page[page_no].append(element)
                except Exception as e:
                    self.log(f"Error processing element: {str(e)}")
            
            self.log(f"문서에서 발견된 총 테이블 수: {total_table_count}, 총 그림 수: {total_picture_count}")
            
            table_info_by_page = {}
            picture_info_by_page = {}
            
            table_counter = 0
            for page_no, tables in table_items_by_page.items():
                if page_no not in table_info_by_page:
                    table_info_by_page[page_no] = []
                
                for table_idx, table in enumerate(tables):
                    table_counter += 1
                    try:
                        table_meta = {
                            'id': getattr(table, 'id', f"table_{page_no}_{table_idx}"),
                            'page_no': page_no,
                            'image_paths': [],
                        }
                        
                        bbox = self._extract_bbox(table)
                        if bbox:
                            table_meta['bbox'] = list(bbox)
                        
                        if hasattr(table, 'num_rows'):
                            table_meta['num_rows'] = table.num_rows
                        if hasattr(table, 'num_cols'):
                            table_meta['num_cols'] = table.num_cols
                        
                        element_image_filename = output_dir / f"{doc_filename}-page-{page_no}-table-{table_counter}.png"
                        image_path = self._extract_and_save_image(table, conv_res.document, element_image_filename)
                        if image_path:
                            table_meta['image_paths'].append(element_image_filename.name)
                        
                        table_info_by_page[page_no].append(table_meta)
                        
                    except Exception as e:
                        self.log(f"Error processing table: {str(e)}")
            
            picture_counter = 0
            for page_no, pictures in picture_items_by_page.items():
                if page_no not in picture_info_by_page:
                    picture_info_by_page[page_no] = []
                
                for pic_idx, picture in enumerate(pictures):
                    picture_counter += 1
                    try:
                        picture_meta = {
                            'id': getattr(picture, 'id', f"picture_{page_no}_{pic_idx}"),
                            'page_no': page_no,
                            'image_paths': [],
                        }
                        
                        bbox = self._extract_bbox(picture)
                        if bbox:
                            picture_meta['bbox'] = list(bbox)
                        
                        if hasattr(picture, 'predicted_class'):
                            picture_meta['predicted_class'] = picture.predicted_class
                        if hasattr(picture, 'confidence'):
                            picture_meta['confidence'] = picture.confidence
                        
                        element_image_filename = output_dir / f"{doc_filename}-page-{page_no}-picture-{picture_counter}.png"
                        image_path = self._extract_and_save_image(picture, conv_res.document, element_image_filename)
                        if image_path:
                            picture_meta['image_paths'].append(element_image_filename.name)
                        
                        picture_info_by_page[page_no].append(picture_meta)
                        
                    except Exception as e:
                        self.log(f"Error processing picture: {str(e)}")
            
            self.log(f"Saved {table_counter} table images and {picture_counter} picture images")
            
            original_rows = []
            
            for (
                content_text,
                content_md,
                content_dt,
                page_cells,
                page_segments,
                page,
            ) in generate_multimodal_pages(conv_res):
                
                current_page_no = page.page_no + 1
                if current_page_no > len(conv_res.pages):
                    current_page_no = len(conv_res.pages)
                
                dpi = page._default_image_scale * 72
                
                ocr_cells = [c for c in page.cells if isinstance(c, OcrCell)]
                ocr_confidence_cells = [c for c in page.cells if hasattr(c, "confidence")]
                avg_confidence = 0
                if ocr_confidence_cells:
                    avg_confidence = sum(c.confidence for c in ocr_confidence_cells) / len(ocr_confidence_cells)
                
                layout_clusters = []
                if page.predictions and hasattr(page.predictions, 'layout') and hasattr(page.predictions.layout, 'clusters'):
                    for cluster in page.predictions.layout.clusters:
                        layout_clusters.append({
                            "id": cluster.id,
                            "label": cluster.label,
                            "bbox": cluster.bbox.as_tuple() if hasattr(cluster.bbox, "as_tuple") else None,
                            "confidence": cluster.confidence,
                            "cell_count": len(cluster.cells)
                        })
                
                equations = []
                if page.predictions and hasattr(page.predictions, 'equations_prediction') and hasattr(page.predictions.equations_prediction, 'equation_map'):
                    for eq in page.predictions.equations_prediction.equation_map.values():
                        equations.append({
                            "id": eq.id,
                            "bbox": eq.cluster.bbox.as_tuple() if hasattr(eq.cluster.bbox, "as_tuple") else None,
                            "text": eq.text
                        })
                
                assembled_elements = {
                    "body": [],
                    "headers": []
                }
                if page.assembled:
                    assembled_elements["body"] = [
                        {
                            "id": elem.id, 
                            "label": elem.label, 
                            "text_length": len(elem.text) if hasattr(elem, "text") and elem.text else 0
                        } 
                        for elem in page.assembled.body
                    ]
                    assembled_elements["headers"] = [
                        {
                            "id": elem.id, 
                            "label": elem.label, 
                            "text": elem.text if hasattr(elem, "text") else ""
                        } 
                        for elem in page.assembled.headers
                    ]
                
                aspect_ratio = None
                if page.size and page.size.height and page.size.width:
                    aspect_ratio = page.size.width / page.size.height
                
                page_relations = {
                    "is_first": current_page_no == 0,
                    "is_last": current_page_no == len(conv_res.pages) - 1,
                    "section_continuation": False
                }
                if page.assembled:
                    page_relations["section_continuation"] = any(
                        elem.label == DocItemLabel.SECTION_HEADER for elem in page.assembled.body
                    )
                
                current_page_table_paths = []
                current_page_figure_paths = []
                
                if current_page_no in table_info_by_page:
                    for table in table_info_by_page[current_page_no]:
                        current_page_table_paths.extend(table.get('image_paths', []))
                        
                if current_page_no in picture_info_by_page:
                    for figure in picture_info_by_page[current_page_no]:
                        current_page_figure_paths.extend(figure.get('image_paths', []))
                
                row = {
                    "document": conv_res.input.file.name,
                    "hash": conv_res.input.document_hash,
                    "page_hash": create_hash(
                        conv_res.input.document_hash + ":" + str(current_page_no)
                    ),
                    "contents": content_text,
                    "contents_md": content_md,
                    "contents_dt": content_dt,
                    
                    "image": {
                        "width": page.image.width if page.image else None,
                        "height": page.image.height if page.image else None,
                        "format": "RGB",
                        "scale": page._default_image_scale,
                        "path": f"{doc_filename}-page-{current_page_no}.png"
                    },
                    
                    "cells": page_cells,
                    "segments": page_segments,
                    
                    "layout": {
                        "clusters": layout_clusters
                    },
                    
                    "tables": table_info_by_page.get(current_page_no, []),
                    "figures": picture_info_by_page.get(current_page_no, []),
                    "equations": equations,
                    
                    "ocr_stats": {
                        "cell_count": len(ocr_cells),
                        "avg_confidence": avg_confidence
                    },
                    
                    "metadata": {
                        "format": str(conv_res.input.format),
                        "filesize": conv_res.input.filesize,
                        "total_pages": conv_res.input.page_count,
                        "current_page": current_page_no
                    },
                    
                    "assembled_elements": assembled_elements,
                    "page_relations": page_relations,
                    
                    "image_paths": {
                        "page_image": f"{doc_filename}-page-{current_page_no}.png",
                        "table_images": current_page_table_paths,
                        "figure_images": current_page_figure_paths,
                        "page_no_zero_based": current_page_no - 1
                    },
                    
                    "extra": {
                        "page_no": current_page_no,
                        "page_num": current_page_no,
                        "width_in_points": page.size.width if page.size else None,
                        "height_in_points": page.size.height if page.size else None,
                        "dpi": dpi,
                        "aspect_ratio": aspect_ratio,
                        "rotation": 0,
                    },
                }
                
                original_rows.append(row)
            
            df = pd.DataFrame(original_rows)
            self.log(f"Created dataframe with {len(df)} rows")
            
            df_for_parquet = df.copy()
            
            complex_columns = [
                'tables', 'figures', 'image_paths', 'cells', 'segments', 'layout',
                'equations', 'assembled_elements', 'page_relations', 'extra', 'image',
                'ocr_stats', 'metadata'
            ]
            
            for col in complex_columns:
                if col in df_for_parquet.columns:
                    json_col = f"{col}_json"
                    df_for_parquet[json_col] = df_for_parquet[col].apply(
                        lambda x: json.dumps(x, ensure_ascii=False, indent=None)
                    )
                    df_for_parquet = df_for_parquet.drop(columns=[col])
            
            parquet_path = output_dir / f"{doc_filename}.parquet"
            df_for_parquet.to_parquet(parquet_path)
            self.log(f"Saved dataframe to {parquet_path}")
            
            converted_result = convert_docling_result(str(parquet_path), str(output_dir))
            
            self.log(f"Docling 분석 완료: {input_file}")
            return converted_result
            
        except Exception as e:
            self.log(f"Docling 분석 중 오류 발생: {str(e)}")
            raise ValueError(f"Docling 분석 실패: {str(e)}")
    
    def _extract_page_number(self, element):
        page_no = None
        
        if hasattr(element, 'page_no'):
            page_no = element.page_no
        elif hasattr(element, 'bbox') and hasattr(element.bbox, 'page'):
            page_no = element.bbox.page
        elif hasattr(element, 'prov') and element.prov:
            try:
                if isinstance(element.prov, list) and len(element.prov) > 0:
                    if hasattr(element.prov[0], 'page_no'):
                        page_no = element.prov[0].page_no
            except Exception:
                pass
        
        if page_no is None:
            return None
        
        return page_no
    
    def _extract_bbox(self, element):
        try:
            if hasattr(element, 'bbox'):
                if hasattr(element.bbox, 'as_tuple'):
                    return element.bbox.as_tuple()
                
                bbox_attrs = ('l', 't', 'r', 'b')
                if all(hasattr(element.bbox, attr) for attr in bbox_attrs):
                    return (
                        getattr(element.bbox, 'l'),
                        getattr(element.bbox, 't'),
                        getattr(element.bbox, 'r'),
                        getattr(element.bbox, 'b')
                    )
                    
                bbox_attrs = ('x0', 'y0', 'x1', 'y1')
                if all(hasattr(element.bbox, attr) for attr in bbox_attrs):
                    return (
                        getattr(element.bbox, 'x0'),
                        getattr(element.bbox, 'y0'),
                        getattr(element.bbox, 'x1'),
                        getattr(element.bbox, 'y1')
                    )
            
            if hasattr(element, 'prov') and isinstance(element.prov, list) and len(element.prov) > 0:
                if hasattr(element.prov[0], 'bbox'):
                    bbox = element.prov[0].bbox
                    if hasattr(bbox, 'as_tuple'):
                        return bbox.as_tuple()
                    
                    bbox_attrs = ('l', 't', 'r', 'b')
                    if all(hasattr(bbox, attr) for attr in bbox_attrs):
                        return (
                            getattr(bbox, 'l'),
                            getattr(bbox, 't'),
                            getattr(bbox, 'r'),
                            getattr(bbox, 'b')
                        )
        except Exception as e:
            self.log(f"경계 상자 추출 실패: {str(e)}")
        
        return None
    
    def _extract_and_save_image(self, element, document, output_path):
        try:
            image = None
            if hasattr(element, 'get_image'):
                image = element.get_image(document)
            
            if image is None and hasattr(element, 'image'):
                image = element.image
                
            if image is None and hasattr(element, 'image') and hasattr(element.image, 'uri'):
                try:
                    from PIL import Image
                    import base64
                    import io
                    
                    uri = element.image.uri
                    if uri.startswith('data:image/'):
                        img_data = uri.split(',')[1]
                        img_binary = base64.b64decode(img_data)
                        image = Image.open(io.BytesIO(img_binary))
                except Exception as e:
                    self.log(f"Base64 변환 실패: {str(e)}")
            
            if image:
                image.save(output_path, "PNG")
                return str(output_path)
            else:
                self.log(f"이미지가 없거나 추출 실패: {output_path}")
                return None
                
        except Exception as e:
            self.log(f"이미지 추출 및 저장 실패: {str(e)}")
            return None

    def parse_start_end_page(self, filepath):
        filename = os.path.basename(filepath)
        name_without_ext = filename.rsplit(".", 1)[0]

        try:
            if len(name_without_ext) < 9:
                return (-1, -1)

            page_numbers = name_without_ext[-9:]

            if not (
                page_numbers[4] == "_"
                and page_numbers[:4].isdigit()
                and page_numbers[5:].isdigit()
            ):
                return (-1, -1)

            start_page = int(page_numbers[:4])
            end_page = int(page_numbers[5:])

            if start_page > end_page:
                return (-1, -1)

            return (start_page, end_page)

        except (IndexError, ValueError):
            return (-1, -1)

    def execute(self, state: ParseState):
        start_time = time.time()
        self.log(f"Start Parsing: {state['working_filepath']}")
        
        filepath = state["working_filepath"]
        parsed_json_path = self._layout_analysis(filepath)

        start_page, _ = self.parse_start_end_page(filepath)
        page_offset = start_page if start_page != -1 else 0

        with open(parsed_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for element in data["elements"]:
                element["page"] = element.get("page", 0) + page_offset

        metadata = {
            "api": data.pop("api", "unknown"),
            "model": data.pop("model", "unknown"),
            "usage": data.pop("usage", {}),
        }

        duration = time.time() - start_time
        self.log(f"Finished Parsing in {duration:.2f} seconds")

        return {"metadata": [metadata], "raw_elements": [data["elements"]]}


class PostDocumentParseNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        elements_list = state["raw_elements"]
        id_counter = 0
        post_processed_elements = []

        for elements in elements_list:
            for element in elements:
                elem = element.copy()
                elem["id"] = id_counter
                id_counter += 1

                post_processed_elements.append(elem)

        self.log(f"Total Post-processed Elements: {id_counter}")

        pages_count = 0
        metadata = state["metadata"]

        for meta in metadata:
            for k, v in meta.items():
                if k == "usage":
                    pages_count += int(v["pages"])

        return {
            "elements_from_parser": post_processed_elements,
        }


class WorkingQueueNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        working_filepath = state.get("working_filepath", None)
        if (
            "working_filepath" not in state
            or state["working_filepath"] is None
            or state["working_filepath"] == ""
        ):
            if len(state["split_filepaths"]) > 0:
                working_filepath = state["split_filepaths"][0]
            else:
                working_filepath = "<<FINISHED>>"
        else:
            if working_filepath == "<<FINISHED>>":
                return {"working_filepath": "<<FINISHED>>"}

            current_index = state["split_filepaths"].index(working_filepath)
            if current_index + 1 < len(state["split_filepaths"]):
                working_filepath = state["split_filepaths"][current_index + 1]
            else:
                working_filepath = "<<FINISHED>>"
        return {"working_filepath": working_filepath}


class SaveStateNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def _get_page_range_from_filepaths(self, split_filepaths):
        min_page = float('inf')
        max_page = 0
        
        for filepath in split_filepaths:
            start_page, end_page = self._parse_start_end_page(filepath)
            if start_page != -1 and end_page != -1:
                min_page = min(min_page, start_page)
                max_page = max(max_page, end_page)
        
        if min_page == float('inf'):
            return None, None
        
        return min_page, max_page

    def _parse_start_end_page(self, filepath):
        filename = os.path.basename(filepath)
        name_without_ext = filename.rsplit(".", 1)[0]

        try:
            if len(name_without_ext) < 9:
                return (-1, -1)

            page_numbers = name_without_ext[-9:]

            if not (
                page_numbers[4] == "_"
                and page_numbers[:4].isdigit()
                and page_numbers[5:].isdigit()
            ):
                return (-1, -1)

            start_page = int(page_numbers[:4])
            end_page = int(page_numbers[5:])

            if start_page > end_page:
                return (-1, -1)

            return (start_page, end_page)

        except (IndexError, ValueError):
            return (-1, -1)

    def execute(self, state: ParseState):
        if state["split_filepaths"]:
            first_filepath = state["split_filepaths"][0]
            base_filename = os.path.basename(first_filepath)
            
            if len(base_filename) > 13:
                potential_page_part = base_filename[-13:-4]
                if (len(potential_page_part) == 9 and 
                    potential_page_part[4] == "_" and 
                    potential_page_part[:4].isdigit() and 
                    potential_page_part[5:].isdigit()):
                    original_name = base_filename[:-14]
                else:
                    original_name = os.path.splitext(base_filename)[0]
            else:
                original_name = os.path.splitext(base_filename)[0]
        else:
            original_name = "unknown"

        min_page, max_page = self._get_page_range_from_filepaths(state["split_filepaths"])
        
        if min_page is not None and max_page is not None:
            save_filename = f"{original_name}_{min_page:04d}_{max_page:04d}_assembled_result.json"
        else:
            save_filename = f"{original_name}_assembled_result.json"

        output_dir = "intermediate"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, save_filename)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        self.log(f"Final state saved to: {save_path}")
        
        return {"save_path": save_path, "save_filename": save_filename}


def continue_parse(state: ParseState):
    if state["working_filepath"] == "<<FINISHED>>":
        return False
    else:
        return True
