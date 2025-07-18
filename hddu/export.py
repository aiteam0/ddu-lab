import re
import os
import io
import base64
from PIL import Image
from io import StringIO
import pandas as pd
from bs4 import BeautifulSoup
from .base import BaseNode
from .state import ParseState

class ExportImage(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def save_to_png(self, base64_encoding, dirname, basename, category, page, index):
        image_data = base64.b64decode(base64_encoding)

        image = Image.open(io.BytesIO(image_data))

        image_dir = os.path.join("export", "images", category)
        os.makedirs(image_dir, exist_ok=True)

        base_prefix = os.path.splitext(basename)[0]
        image_filename = (
            f"{base_prefix.upper()}_{category.upper()}_Page_{page}_Index_{index}.png"
        )
        image_path = os.path.join(image_dir, image_filename)
        abs_image_path = os.path.abspath(image_path)

        image.save(abs_image_path)
        return abs_image_path

    def execute(self, state: ParseState):
        filepath = state["filepath"]
        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        for elem in state["elements_from_parser"]:
            if elem["category"] in ["figure", "chart", "table"]:
                base64_encoding = elem.get("base64_encoding")
                if base64_encoding is not None: 
                    print(base64_encoding[:30])
                    image_path = self.save_to_png(
                        base64_encoding,
                        dirname,
                        basename,
                        elem["category"],
                        elem["page"],
                        elem["id"],
                    )
                    elem["png_filepath"] = image_path

        return {"elements_from_parser": state["elements_from_parser"]}


class ExportHTML(BaseNode):
    def __init__(self, ignore_new_line_in_text=False, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.ignore_new_line_in_text = ignore_new_line_in_text

    def _add_base64_src_to_html(self, html, base64_encoding):
        if not base64_encoding:
            return html

        pattern = r"<img([^>]*)>"
        replacement = f'<img\\1 src="data:image/png;base64,{base64_encoding}">'
        return re.sub(pattern, replacement, html)
    
    def _add_base64_src_to_html_docling(self, html, base64_encoding):
        if not base64_encoding:
            return html

        image_type = "image/png"
        base64_image_data_uri =  f"data:{image_type};base64,{base64_encoding}"
        image_tag = f'<img src="{base64_image_data_uri}" alt="Embedded Image">'

        return image_tag + "\n" + html


    def execute(self, state: ParseState):
        filepath = state["filepath"]
        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        html_basename = os.path.splitext(basename)[0] + ".html"
        model = state["metadata"][0]["model"]
        
        # export 폴더 생성
        export_dir = "export"
        os.makedirs(export_dir, exist_ok=True)
        
        html_filepath = os.path.join(export_dir, html_basename)

        with open(html_filepath, "w", encoding="utf-8") as f:
            for elem in state["elements_from_parser"]:
                if elem["category"] in ["header1", "header2", "header3", "footer", "footnote"]:
                    continue

                if elem["category"] in ["figure", "chart"]:
                    base64_encoding = elem.get("base64_encoding")

                    if "content-list" in model:
                        modified_html = self._add_base64_src_to_html(
                            elem["content"]["html"], base64_encoding
                        )
                    else:
                        modified_html = self._add_base64_src_to_html_docling(
                            elem["content"]["html"], base64_encoding
                        )
                    f.write(modified_html)
                    continue

                if elem["category"] == "table":
                    base64_encoding = elem.get("base64_encoding")
                    table_image = f'<img src="data:image/png;base64,{base64_encoding}">'
                    f.write(table_image)
                    f.write(elem["content"]["html"])
                    
                else:
                    if self.ignore_new_line_in_text:
                        f.write(elem["content"]["html"].replace("<br>", " "))
                    else:
                        f.write(elem["content"]["html"])

        self.log(f"HTML file created successfully: {html_filepath}")

        return {"export": [html_filepath]}


class ExportMarkdown(BaseNode):
    def __init__(
        self,
        ignore_new_line_in_text=False,
        show_image=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(verbose=verbose, **kwargs)
        self.ignore_new_line_in_text = ignore_new_line_in_text
        self.show_image = show_image
        self.separator = "\n\n"

    def _add_src_to_markdown(self, png_filepath, md_dir):
        if not png_filepath:
            return ""
        
        try:
            rel_path = os.path.relpath(png_filepath, md_dir)
            rel_path = rel_path.replace("\\", "/")
            return f"![]({rel_path})"
        except ValueError:
            return f"![]({png_filepath})"

    def execute(self, state: ParseState):
        filepath = state["filepath"]
        dirname = os.path.abspath(os.path.dirname(filepath))
        basename = os.path.basename(filepath)
        md_basename = os.path.splitext(basename)[0] + ".md"
        
        export_dir = os.path.join("export")
        os.makedirs(export_dir, exist_ok=True)
        
        md_filepath = os.path.join(export_dir, md_basename)

        with open(md_filepath, "w", encoding="utf-8") as f:
            for elem in state["elements_from_parser"]:
                if elem["category"] in ["header", "footer", "footnote"]:
                    continue

                if elem["category"] in ["figure", "chart"]:
                    if self.show_image:
                        png_filepath = elem.get("png_filepath")
                        if png_filepath:
                            print(f"png_filepath: {png_filepath}")
                            self.log(f"png_filepath: {png_filepath}")
                        else:
                            print("png_filepath가 없습니다.")
                            self.log("png_filepath가 없습니다.")
                        modified_md = self._add_src_to_markdown(png_filepath, export_dir)
                        f.write(modified_md + self.separator)

                elif elem["category"] in ["table"]:
                    if self.show_image:
                        png_filepath = elem.get("png_filepath")
                        if png_filepath:
                            print(f"png_filepath: {png_filepath}")
                            self.log(f"png_filepath: {png_filepath}")
                        else:
                            print("png_filepath가 없습니다.")
                            self.log("png_filepath가 없습니다.")
                        modified_md = self._add_src_to_markdown(png_filepath, export_dir)
                        f.write(modified_md + self.separator)
                    f.write(elem["content"]["markdown"] + self.separator)

                elif elem["category"] in ["paragraph"]:
                    if self.ignore_new_line_in_text:
                        f.write(
                            elem["content"]["markdown"].replace("\n", " ")
                            + self.separator
                        )
                    else:
                        f.write(elem["content"]["markdown"] + self.separator)
                else:
                    f.write(elem["content"]["markdown"] + self.separator)

        self.log(f"Markdown file created successfully: {md_filepath}")

        return {"export": [md_filepath]}


class ExportTableCSV(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        filepath = state["filepath"]

        dirname = os.path.dirname(filepath)
        table_dir = os.path.join("export", "tables")
        os.makedirs(table_dir, exist_ok=True)

        basename = os.path.basename(filepath)
        base_without_ext = os.path.splitext(basename)[0]

        csv_filepaths = []

        for elem in state["elements_from_parser"]:
            if elem["category"] == "table":
                soup = BeautifulSoup(elem["content"]["html"], "html.parser")

                for td in soup.find_all("td"):
                    td.string = (
                        td.get_text(strip=True).replace("\\t", " ").replace("\t", " ")
                    )

                cleaned_html = str(soup)
                cleaned_html_io = StringIO(cleaned_html)

                try:
                    parsed_tables = pd.read_html(cleaned_html_io)
                    for table in parsed_tables:
                        csv_filename = f"{base_without_ext.upper()}_TABLE_Page_{elem['page']}_Index_{elem['id']}.csv"
                        csv_filepath = os.path.join(table_dir, csv_filename)
                        absolute_path = os.path.abspath(csv_filepath)

                        table.to_csv(absolute_path, index=False, encoding="utf-8-sig")
                        csv_filepaths.append(absolute_path)
                        elem["csv_filepath"] = absolute_path
                        self.log(
                            f"CSV file created successfully: {absolute_path}"
                        )
                except Exception as e:
                    self.log(f"Error parsing table: {str(e)}")
                    continue

        if csv_filepaths:
            return {
                "elements_from_parser": state["elements_from_parser"],
                "export": csv_filepaths,
            }
        else:
            self.log("No tables to convert.")
            return {"elements_from_parser": state["elements_from_parser"], "export": []}
