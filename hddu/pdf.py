from .base import BaseNode
import pymupdf
import os
from .state import ParseState


class SplitPDFFilesNode(BaseNode):

    def __init__(self, batch_size=10, test_page=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "SplitPDFNode"
        self.batch_size = batch_size
        self.test_page = test_page

    def execute(self, state: ParseState) -> ParseState:
        filepath = state["filepath"]

        input_pdf = pymupdf.open(filepath)
        num_pages = len(input_pdf)
        print(f"총 페이지 수: {num_pages}")

        if self.test_page is not None:
            if self.test_page < num_pages:
                num_pages = self.test_page

        ret = []
        for start_page in range(0, num_pages, self.batch_size):
            end_page = min(start_page + self.batch_size, num_pages) - 1

            input_file_basename = os.path.splitext(filepath)[0]
            output_file = f"{input_file_basename}_{start_page:04d}_{end_page:04d}.pdf"
            print(f"분할 PDF 생성: {output_file}")

            with pymupdf.open() as output_pdf:
                output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
                output_pdf.save(output_file)
                ret.append(output_file)

        input_pdf.close()

        return ParseState(split_filepaths=ret)
