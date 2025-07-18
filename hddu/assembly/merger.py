import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from pydantic import BaseModel, Field
except ImportError:
    from pydantic.v1 import BaseModel, Field

from . import config
from .config import MatchedPair, Element
from ..logging_config import get_logger

logger = get_logger(__name__)
logger.info("[ASSEMBLY_INIT] merger.py logger initialized - testing logging system")
    
class MergedElement(BaseModel):
    category: str = Field(description="The category of the element (e.g., 'paragraph', 'heading1').")
    text: str = Field(description="The merged and corrected text content.")
    is_valid: bool = Field(description="Whether this element is valid and should be included.", default=True)



class MergedContent(BaseModel):
    text: str = Field(description="The most accurate plain text representation or caption for the figure.")
    markdown: str = Field(description="A clean, well-formatted Markdown representation for the figure (e.g., ![caption](path)).")
    html: str = Field(description="A clean, well-formatted HTML representation for the figure (e.g., <figure><img>...</figure>).")


class LLMMerger:
    def __init__(self, text_llm: BaseChatModel, vision_llm: BaseChatModel):
        self.text_llm_structured = text_llm.with_structured_output(MergedElement)
        self.vision_llm_content = vision_llm.with_structured_output(MergedContent)

    def _create_text_merge_prompts(self, pairs: List[MatchedPair], d_only: List[Element], y_only: List[Element], all_d_elements: List[Element] = None, all_y_elements: List[Element] = None) -> List[Dict[str, Any]]:
        prompts = []
        
        for d_elem, y_elem in pairs:
            context_info = self._build_context_info(d_elem, y_elem, all_d_elements, all_y_elements)
            
            enhanced_system_prompt = """You are an expert Korean document analyst specializing in high-quality document processing. Your task is to merge two text elements from different parsers, selecting the highest quality result based on Korean document standards.

**QUALITY DECISION FRAMEWORK:**

1. **FORMATTING STANDARDS (Highest Priority):**
   - DATES: Use Korean standard dot notation (YYYY.MM.DD.) for official documents
   - NUMBERS: Use proper separators (1,000,000 not 1000000)
   - PUNCTUATION: Follow Korean typography rules
   - SPACING: Proper Korean spacing between words and punctuation

2. **PARSER CHARACTERISTICS:**
   - Parser_A (Docling): Superior formatting, punctuation, structure
   - Parser_B (DocYOLO): Superior text recognition, completeness

3. **COMMON DECISION PATTERNS:**
   - Dates: "2019.10.29." (formatted) > "2019 10 29" (unformatted)
   - Numbers: "1,234.56" > "1234.56" 
   - Titles: Proper category (heading1) > wrong category (paragraph)
   - Korean text: Proper spacing > poor spacing
   - Punctuation: Complete punctuation > missing punctuation

4. **CONTEXT CONSISTENCY:**
   - Maintain formatting consistency with surrounding elements
   - Consider document type and style patterns
   - Preserve logical document structure

**DECISION PRIORITY:**
1. Standard Korean formatting conventions
2. Text completeness and accuracy  
3. Proper categorization
4. Consistency with document context

OUTPUT LANGUAGE REQUIREMENT: Korean"""

            enhanced_human_prompt = f"""Analyze and merge these two elements using Korean document quality standards:

**Parser_A (Docling - Structure & Formatting Focused):**
{d_elem}

**Parser_B (DocYOLO - Text Recognition Focused):**
{y_elem}

**DOCUMENT CONTEXT:**
{context_info}

**ANALYSIS CHECKLIST:**
□ Compare formatting quality (dates, numbers, punctuation)
□ Evaluate text completeness and accuracy
□ Check category appropriateness (heading vs paragraph)
□ Consider consistency with surrounding elements
□ Apply Korean document formatting standards

**SPECIFIC GUIDELINES:**
- For DATES: Prefer dot notation (YYYY.MM.DD.) for Korean official documents
- For NUMBERS: Prefer properly formatted versions with separators
- For HEADINGS: Ensure correct category assignment (heading1, heading2, etc.)
- For TEXT: Choose version with better Korean grammar and spacing

Return the optimal merged element following Korean document standards."""

            prompts.append({
                "messages": [
                    SystemMessage(content=enhanced_system_prompt),
                    HumanMessage(content=enhanced_human_prompt)
                ],
                "original_element": d_elem
            })

        for elem in d_only:
            prompts.append({
                "messages": [
                    SystemMessage(content="You are an expert document analyst. The following element was found only by a highly structured parser (Parser_A). Review its content. If it is valid, return it as a JSON object. If it looks like noise, set 'is_valid' to false. OUTPUT LANGUAGE REQUIREMENT: Korean"),
                    HumanMessage(content=f"Review the following element:\n\nParser_A (Docling):\n{elem}")
                ],
                "original_element": elem
            })
            
        for elem in y_only:
             prompts.append({
                "messages": [
                    SystemMessage(content="You are an expert document analyst. The following element was found only by a text-aggregation-focused parser (Parser_B). Review its content. It might be a valid missing piece or noise. Determine its correct category and text. If it is valid, return it as a JSON object. If it's noise, set 'is_valid' to false. OUTPUT LANGUAGE REQUIREMENT: Korean"),
                    HumanMessage(content=f"Review the following element and determine its final structure:\n\nParser_B (DocYOLO):\n{elem}")
                ],
                "original_element": elem
            })
        return prompts
    
    def _build_context_info(self, d_elem: Element, y_elem: Element, all_d_elements: List[Element] = None, all_y_elements: List[Element] = None) -> str:
        context_parts = []
        
        d_total = len(all_d_elements) if all_d_elements else 0
        y_total = len(all_y_elements) if all_y_elements else 0
        logger.debug(f"[CONTEXT_BUILD] Input elements: docling={d_total}, docyolo={y_total}")
        
        if all_d_elements:
            current_id = d_elem.get('id', 0)
            
            prev_d = [elem for elem in all_d_elements if elem.get('id', 999) < current_id]
            next_d = [elem for elem in all_d_elements if elem.get('id', -1) > current_id]
            
            if prev_d:
                prev_elem = sorted(prev_d, key=lambda x: x.get('id', 0))[-1]
                prev_text = prev_elem.get('content', {}).get('text', '')[:30]
                context_parts.append(f"**DOCLING PREVIOUS:** [{prev_elem.get('category', 'unknown')}] {prev_text}...")
            
            if next_d:
                next_elem = sorted(next_d, key=lambda x: x.get('id', 0))[0]
                next_text = next_elem.get('content', {}).get('text', '')[:30]
                context_parts.append(f"**DOCLING NEXT:** [{next_elem.get('category', 'unknown')}] {next_text}...")
        
        if all_y_elements:
            current_id = y_elem.get('id', 0)
            
            prev_y = [elem for elem in all_y_elements if elem.get('id', 999) < current_id]
            next_y = [elem for elem in all_y_elements if elem.get('id', -1) > current_id]
            
            if prev_y:
                prev_elem = sorted(prev_y, key=lambda x: x.get('id', 0))[-1]
                prev_text = prev_elem.get('content', {}).get('text', '')[:30]
                context_parts.append(f"**DOCYOLO PREVIOUS:** [{prev_elem.get('category', 'unknown')}] {prev_text}...")
            
            if next_y:
                next_elem = sorted(next_y, key=lambda x: x.get('id', 0))[0]
                next_text = next_elem.get('content', {}).get('text', '')[:30]
                context_parts.append(f"**DOCYOLO NEXT:** [{next_elem.get('category', 'unknown')}] {next_text}...")
        
        result_context = '\n'.join(context_parts) if context_parts else "No surrounding context available"
        logger.debug(f"[CONTEXT_BUILD] Generated context: {len(context_parts)} parts, "
                    f"total_length={len(result_context)}")
        return result_context
        






        
