from typing import Dict, List, Any
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class Element:
    category: str  # heading1, heading2, heading3, paragraph, list, table, figure, chart, equation, caption, footnote, header, footer, reference
    content: str = ""
    html: str = ""
    markdown: str = ""
    base64_encoding: str = None
    image_filename: str = None
    page: int = None
    id: int = None
    coordinates: List[List[float]] = field(default_factory=list)
    entity: Dict[str, Any] = field(default_factory=dict)

    caption: str = ""
    translation_text: str = ""
    translation_html: str = ""
    translation_markdown: str = ""
    contextualize_text: str = ""
    
    processing_type: str = ""
    processing_status: str = ""
    source_parser: str = ""
    raw_output: str = ""

    def copy(self):
        return deepcopy(self)
