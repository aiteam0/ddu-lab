from typing import TypedDict, Annotated, List, Dict
import operator
from .element import Element
from langchain_core.documents import Document


class ParseState(TypedDict):
    filepath: Annotated[str, "filepath"] 
    filetype: Annotated[
        str, "filetype"
    ]
    split_filepaths: Annotated[List[str], "split_filepaths"]
    working_filepath: Annotated[str, "working_filepath"]

    metadata: Annotated[
        List[Dict], operator.add
    ]

    total_cost: Annotated[float, "total_cost"]

    raw_elements: Annotated[List[Dict], operator.add]
    elements_from_parser: Annotated[
        List[Dict], "elements_from_parser"
    ]

    translated_elements: Annotated[
        List[Dict], "translated_elements"
    ]

    image_paths: Annotated[Dict[str, str], "image_paths"]

    elements: Annotated[List[Element], "elements"]
    reconstructed_elements: Annotated[
        List[Dict], "reconstructed_elements"
    ]

    export: Annotated[List, operator.add]

    texts_by_page: Annotated[Dict[int, str], "texts_by_page"]
    images_by_page: Annotated[
        Dict[int, List[Element]], "images_by_page"
    ]

    tables_by_page: Annotated[
        Dict[int, List[Element]], "tables_by_page"
    ]

    extracted_image_entities: Annotated[
        List[Element], "extracted_image_entities"
    ]

    extracted_table_entities: Annotated[
        List[Element], "extracted_table_entities"
    ]

    documents: Annotated[List[Document], "documents"]

    language: Annotated[str, "language"]
