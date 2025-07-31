from hddu.config import (
    create_text_model,
    create_vision_model,
    INTERPRETER_TEMPERATURE,
    INTERPRETER_MAX_TOKENS,
    INTERPRETER_BATCH_SIZE,
    INTERPRETER_MAX_RETRIES,
    INTERPRETER_RETRY_DELAY,
    INTERPRETER_BATCH_REDUCTION_FACTOR
)


import base64
import os
import json
from io import BytesIO
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from PIL import Image
import requests
from datetime import datetime
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import re
import asyncio

from .logging_config import get_logger
from .rate_limit_handler import (
    TokenUsageTracker, 
    ExponentialBackoffHandler, 
    RateLimitAwareAPIClient
)

logger = get_logger(__name__)


class ImageContextExtractor:


    def __init__(
        self,
        model_type: str = "vision",  # 하위 호환성용
        model_name: Optional[str] = None,  # 하위 호환성용
        api_key: Optional[str] = None,  # 하위 호환성용
        ollama_base_url: str = "http://localhost:11434",  # 하위 호환성용
        temperature: float = INTERPRETER_TEMPERATURE,
        max_tokens: int = INTERPRETER_MAX_TOKENS,
    ):

        if model_name is not None or api_key is not None:
            print("⚠️ model_name, api_key 파라미터는 더 이상 사용되지 않습니다. .env 파일에서 VISION_LLM_PROVIDER를 설정하세요.")
        
        self.model_type = model_type
        self.model_name = model_name or "auto"
        
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.model = create_vision_model(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        self.output_parser = StrOutputParser()

    def encode_image_base64(self, image_input: Union[str, Path, Image.Image]) -> str:

        if isinstance(image_input, (str, Path)):
            image_path = str(image_input)
            
            if self._is_likely_base64(image_path):
                return image_path
            
            elif image_path.startswith(("http://", "https://")):
                response = requests.get(image_path)
                response.raise_for_status()
                image_bytes = response.content
            else:
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                    
        elif isinstance(image_input, Image.Image):
            buffered = BytesIO()
            image_input.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
        else:
            raise TypeError("지원하지 않는 이미지 입력 타입")

        return base64.b64encode(image_bytes).decode("utf-8")

    def _is_likely_base64(self, s: str) -> bool:

        try:
            if len(s) < 10 or not s.replace('+', '').replace('/', '').replace('=', '').isalnum():
                return False
            
            base64.b64decode(s, validate=True)
            return True
            
        except Exception:
            return False

    def _create_context_extraction_prompt(self) -> ChatPromptTemplate:

        system_message = """You are an expert image analyst. 
Analyze the given image and extract comprehensive, structured context information 
using Markdown formatting that can be utilized in a RAG (Retrieval-Augmented Generation) system.

Format your response using proper Markdown syntax with headers, lists, tables, etc. 
DO NOT wrap your entire response in code blocks (```markdown or ```). 
Provide the content directly using Markdown formatting.

Analyze the following aspects in detail:

## Main Objects and Elements
- All important objects, people, animals, buildings, etc. in the image
- Features, colors, sizes, and positions of each object

## Text Information
- All text in the image (signs, labels, document content, etc.)
- Language, font, and style of the text
- **If the image contains a table, extract it as a properly formatted Markdown table with all rows and columns**

## Spatial Context
- Location, environment, background description
- Indoor/outdoor classification, geographical features

## Actions and Situations
- People's actions and interactions
- Ongoing activities or situations

## Visual Characteristics
- Color scheme, lighting, composition, style
- Overall atmosphere of the image

## Metadata and Additional Information
- Estimated time period, purpose of the image
- Cultural, historical context (if applicable)

## Keywords
- Related keywords that could be useful for search (as a bullet list)

**IMPORTANT**: 
1. If the image contains a table or tabular data, extract it as a properly formatted Markdown table with clear headers and all data preserved. 
2. DO NOT use code block syntax (```) to wrap your response.
3. Provide content directly using Markdown formatting (headers with #, lists with -, tables with |, etc.).
4. Use Korean language."""

        user_message = """Analyze this image and extract comprehensive context information using Markdown formatting following the guidelines above. 
Provide detailed and accurate information so that this data can be used in a RAG system 
to search for related images and answer questions. 

Remember: Do not wrap your response in code blocks. Use Markdown formatting directly (headers, lists, tables, etc.).
If there are tables in the image, make sure to extract them as properly formatted Markdown tables.
Use Korean language."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_data}"}}
            ])
        ])

    def _create_summary_extraction_prompt(self) -> ChatPromptTemplate:

        system_message = """You are an expert image summarizer. 
Analyze the given image and provide a concise and accurate summary using Markdown formatting optimized for RAG search.

DO NOT wrap your response in code blocks (```markdown or ```). 
Provide the content directly using Markdown syntax (headers, lists, etc.).

If the image contains tables, include a brief description of the table structure and key data points.
Use Korean language."""

        user_message = """Summarize the key content of this image using Markdown formatting with 2-3 sentences or bullet points. 
Include main keywords and features so that this summary can be used to find related images during search.

Remember: Do not wrap your response in code blocks. Use Markdown formatting directly.
If there are tables in the image, briefly describe the table structure and highlight important data.
Use Korean language."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_data}"}}
            ])
        ])

    def extract_comprehensive_context(
        self,
        image_input: Union[str, Path, Image.Image],
        custom_prompt: Optional[str] = None,
        return_json: bool = False,
    ) -> Dict[str, Any]:

        try:
            image_base64 = self.encode_image_base64(image_input)
            
            if custom_prompt:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert image analyst."),
                    ("user", [
                        {"type": "text", "text": custom_prompt},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_data}"}}
                    ])
                ])
            else:
                prompt = self._create_context_extraction_prompt()

            chain = prompt | self.model | self.output_parser
            
            response = chain.invoke({"image_data": image_base64})
            
            if return_json:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return {"raw_text": response, "parsing_error": True}
            else:
                return {"markdown_content": response}
                
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }

    def extract_summary(
        self,
        image_input: Union[str, Path, Image.Image]
    ) -> str:

        try:
            image_base64 = self.encode_image_base64(image_input)
            prompt = self._create_summary_extraction_prompt()
            
            chain = prompt | self.model | self.output_parser
            return chain.invoke({"image_data": image_base64})
            
        except Exception as e:
            return f"Error occurred during summary extraction: {str(e)}"

    def extract_text_content(
        self,
        image_input: Union[str, Path, Image.Image]
    ) -> str:

        custom_prompt = """Extract all text content from this image accurately using Markdown formatting. 
Include all text such as signs, labels, document content, handwriting, etc. 

DO NOT wrap your response in code blocks (```markdown or ```). 
Provide the content directly using Markdown syntax.

**IMPORTANT**: If the image contains tables or tabular data, extract them as properly formatted Markdown tables with:
- Clear column headers
- Proper table syntax using pipes (|) and hyphens (-)
- All rows and columns preserved
- Accurate data transcription

For non-tabular text, format appropriately using Markdown syntax (headers, lists, paragraphs, etc.).
If there is no text in the image, please respond with 'No text found'.
Use Korean language."""
        
        try:
            image_base64 = self.encode_image_base64(image_input)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in extracting text from images with special focus on converting tables to Markdown format. Always provide content directly using Markdown syntax without wrapping in code blocks."),
                ("user", [
                    {"type": "text", "text": custom_prompt},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_data}"}}
                ])
            ])
            
            chain = prompt | self.model | self.output_parser
            return chain.invoke({"image_data": image_base64})
            
        except Exception as e:
            return f"Error occurred during text extraction: {str(e)}"

    def batch_extract_context(
        self,
        image_inputs: List[Union[str, Path, Image.Image]],
        custom_prompts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:

        results = []
        
        for i, image_input in enumerate(image_inputs):
            custom_prompt = (
                custom_prompts[i] if custom_prompts and i < len(custom_prompts) 
                else None
            )
            
            result = self.extract_comprehensive_context(
                image_input, 
                custom_prompt=custom_prompt
            )
            results.append(result)
            
        return results

    def extract_for_rag_indexing(
        self,
        image_input: Union[str, Path, Image.Image],
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        context_result = self.extract_comprehensive_context(image_input, return_json=False)
        context = context_result.get("markdown_content", "")
        
        summary = self.extract_summary(image_input)
        
        text_content = self.extract_text_content(image_input)
        
        rag_data = {
            "image_id": image_id or str(hash(str(image_input))),
            "summary": summary,
            "text_content": text_content,
            "detailed_context": context_result,
            "extraction_model": f"{self.model_type}:{self.model_name}",
            "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        try:
            rag_data["keywords"] = self._extract_keywords_from_markdown(context, summary, text_content)
        except:
            rag_data["keywords"] = []
        
        return rag_data

    def _extract_keywords_from_markdown(self, context: str, summary: str, text_content: str) -> List[str]:
        import re
        
        all_text = f"{context} {summary} {text_content}"
        
        korean_words = re.findall(r'[가-힣]{2,}', all_text)
        english_words = re.findall(r'\b[A-Za-z]{3,}\b', all_text)
        
        stopwords = {
            '것이', '그것', '이것', '있는', '하는', '되는', '있다', '하다', '되다',
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'does', 'let', 'man', 'new', 'put', 'say', 'she', 'too', 'use'
        }
        keywords = [word for word in korean_words + english_words 
                   if word.lower() not in stopwords and len(word) > 2]
        
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(15)]


def extract_image_context(
    image_input: Union[str, Path, Image.Image],
    model_type: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:

    extractor = ImageContextExtractor(
        model_type=model_type,
        model_name=model_name,
        **kwargs
    )
    return extractor.extract_comprehensive_context(image_input)


def extract_image_summary(
    image_input: Union[str, Path, Image.Image],
    model_type: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> str:

    extractor = ImageContextExtractor(
        model_type=model_type,
        model_name=model_name,
        **kwargs
    )
    return extractor.extract_summary(image_input)


def prepare_for_rag(
    image_input: Union[str, Path, Image.Image],
    image_id: Optional[str] = None,
    model_type: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:

    extractor = ImageContextExtractor(
        model_type=model_type,
        model_name=model_name,
        **kwargs
    )
    return extractor.extract_for_rag_indexing(image_input, image_id)

from hddu.state import ParseState
from langchain_core.prompts import PromptTemplate
from pydantic import Field, BaseModel
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser


DEFAULT_BATCH_SIZE = INTERPRETER_BATCH_SIZE
DEFAULT_MAX_TOKENS = INTERPRETER_MAX_TOKENS

PROCESSABLE_CATEGORIES = [
    "paragraph", "heading1", "heading2", "heading3", 
    "header", "footer", "caption", "list", "footnote", "table", "figure", "chart", "equation", "reference"
]

IMAGE_CATEGORIES = ["table", "figure", "chart", "equation"]


class ContextualizedText(BaseModel):
    contextualized_text: str = Field(
        description="The more contextualized text from the given text"
    )

output_parser = PydanticOutputParser(pydantic_object=ContextualizedText)

prompt = PromptTemplate.from_template(
    """당신은 RAG 시스템을 위한 문서 요소 최적화 전문가입니다.
주어진 텍스트를 배경 정보를 활용하여 검색과 이해가 쉬운 형태로 개선하세요.

**핵심 원칙 (우선순위 순):**

1. **문서 구조적 맥락**:
   - 페이지 위치: (예) 제공된 문맥화 대상 element의 페이지를 참고하여, "페이지 X에서..."
   - 요소 유형: (예) 제공된 대상 문맥화 대상 element의 카테고리를 참조하여, "이 제목/문단/표/그림에서는..."
   - 구성 관계: (예) 제공된 문맥화 대상 element의 백그라운드 정보를 참조하여,"OOO을 설명하는 문단/표/그림과 함께..."

2. **지시대명사 참조 가이드 : 불확실한 지시대명사 사용 지양**:
   - "이 문단","이 그림", "이 표", "해당 이미지" 등을 구체적 설명으로 반드시 대체
   - (예) "이 문단" → 텍스트 요약을 활용하여 "OOO을 설명하는 문단"로 사용하여 지시대명사 반드시 대체
   - (예) "이 그림" → 이미지 요약을 활용하여 "OOO을 설명하는 이미지"로 사용하여 지시대명사 반드시 대체

3. **검색 효율성 향상**:
   - 모호한 표현을 구체적이고 검색 가능한 용어로 변환
   - 핵심 키워드와 개념을 명확히 포함
   - RAG 시스템에서 관련 정보를 찾기 쉽도록 최적화

4. **내용 중심 맥락화**:
   - 관련 내용과의 의미적 연결 강화
   - 주제적 일관성과 연관성 명시
   - 실질적 정보 가치 향상

**지양할 요소들**:
- 불필요한 메타정보: 요소 ID, 추출 시간, 기술적 세부사항
- 형식적 수사: "이를 통해...", "따라서..." 등의 불필요한 연결어

**출력 요구사항**:
1. **자연스러운 통합**: 원본 텍스트와 추가 맥락이 하나의 자연스러운 문장/문단이 되도록 작성
2. **과도한 기술적 세부사항 지양**: 요소 ID, 추출 타임스탬프, 모델명 등은 포함하지 않음
3. **원본 의도 보존**: 핵심 메시지와 사실 정보 유지
4. **자연스러운 한국어**: 문법적으로 올바르고 읽기 쉬운 문장
5. **간결성**: 불필요한 확장보다는 명확하고 간결한 표현
**CRITICAL**: 반드시 배경정보의 지시대명사 참조 가이드를 활용하여 정확히 대체"

**배경 정보**:
{background_information}

**문맥화 대상 텍스트**:
{text}

**개선된 결과를 출력하세요**:"""
)

llm = create_text_model(temperature=0).with_structured_output(
    ContextualizedText
)

chain = prompt | llm

openai_extractor = ImageContextExtractor(
    max_tokens=DEFAULT_MAX_TOKENS
)



def _process_image_element(element: dict, extractor: ImageContextExtractor) -> str:

    if "base64_encoding" not in element:
        return ""
    
    try:
        logger.debug("Base64 encoding found")
        image_context = extractor.extract_for_rag_indexing(
            element["base64_encoding"], 
            f"{element['category']}_{element['page']}_{element['id']}"
        )
        
        image_info = [
            f"이미지 ID: {image_context['image_id']}",
            f"요약: {image_context['summary']}",
            f"텍스트 내용: {image_context['text_content']}",
            f"상세 컨텍스트: {image_context['detailed_context']}",
            f"키워드: {', '.join(image_context['keywords'])}"
        ]
        
        if "caption" in element:
            element["caption"] += "\n\nSummary:\n\n" + image_context["summary"]
        else:
            element["caption"] = "Summary:\n\n" + image_context["summary"]
        
        background_info = " ".join(image_info) + " "
        logger.debug(f"background_image: {background_info}")
        return background_info
        
    except Exception as e:
        logger.error(f"Error processing image element: {e}")
        return f"Error processing image: {str(e)}"


def _group_elements_by_page(elements: List[dict]) -> Dict[int, List[dict]]:

    elements_by_page = {}
    for element in elements:
        if element["category"]:
            page = element["page"]
            if page not in elements_by_page:
                elements_by_page[page] = []
            elements_by_page[page].append(element)
    
    return elements_by_page


def _create_background_information(
    batch: List[dict], 
    page: int, 
    extractor: ImageContextExtractor
) -> str:

    sections = []

    sections.append(f"**페이지**: {page}")

    category_counts = {}
    for elem in batch:
        category = elem.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    
    category_summary = ", ".join([f"{cat}({count}개)" for cat, count in category_counts.items()])
    sections.append(f"**페이지 구성**: {category_summary}")

    content_by_category = {}
    for elem in batch:
        category = elem.get("category", "unknown")
        content = elem.get("content", {}).get("markdown", "")
        if content and len(content.strip()) > 0 and category not in IMAGE_CATEGORIES:
            if category not in content_by_category:
                content_by_category[category] = []
            content_by_category[category].append(content)
    
    for category, contents in content_by_category.items():
        if len(contents) == 1:
            sections.append(f"**{category} 텍스트 내용**: {contents[0]}")
        else:
            for i, content in enumerate(contents, 1):
                content_clean = content.strip()
                if content_clean.startswith('#'):

                    preview = content_clean.split('\n')[0].replace('#', '').strip()[:100]
                else:
                    preview = content_clean[:40].replace('\n', ' ').strip()

                sections.append(f"**{category} - 텍스트 {preview}**: {content}")
    
    visual_analyses = []
    for elem in batch:
        if elem["category"] in IMAGE_CATEGORIES and "base64_encoding" in elem:
            try:
                analysis = _process_image_element(elem, extractor)
                if analysis:
                    category_korean_map = {
                        'table': '표',
                        'figure': '이미지', 
                        'chart': '차트',
                        'equation': '수식'
                    }
                    
                    category_korean = category_korean_map.get(elem['category'], elem['category'])
                    identifier = f"관련 {category_korean}"
                    
                    visual_analyses.append(f"**{identifier}**: {analysis}")
            except Exception as e:
                logger.debug(f"이미지 분석 실패: {e}")
                continue
    
    if visual_analyses:
        sections.append("**시각적 요소 상세 분석**:")
        sections.extend(visual_analyses)
    
    total_elements = len(batch)
    total_chars = sum(len(elem.get("content", {}).get("markdown", "")) for elem in batch)
    sections.append(f"**전체 정보**: {total_elements}개 요소, 약 {total_chars}자 분량")
    
    return "\n\n".join(sections) 


def _process_batch(
    batch: List[dict], 
    background_information: str, 
    chain
) -> List[ContextualizedText]:

    batch_data = [
        {
            "text": elem["content"]["markdown"],
            "background_information": background_information,
        }
        for elem in batch
    ]
    
    contextualized_results = chain.batch(batch_data)
    
    for elem, cr in zip(batch, contextualized_results):
        print("[Original Text]")
        print(elem["content"]["markdown"])
        print("-" * 100)
        print("[Contextualized Text]")
        print(cr.contextualized_text)
        print("-" * 100)
    
    return contextualized_results


def _process_batch_with_retry(
    batch: List[dict], 
    background_information: str, 
    chain,
    verbose: bool = True
) -> List[ContextualizedText]:

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return _process_batch_sync_fallback(batch, background_information, chain, verbose)
        else:
            return loop.run_until_complete(_process_batch_with_retry_async(batch, background_information, chain, verbose))
    except RuntimeError:
        return asyncio.run(_process_batch_with_retry_async(batch, background_information, chain, verbose))

async def _process_batch_with_retry_async(
    batch: List[dict], 
    background_information: str, 
    chain,
    verbose: bool = True
) -> List[ContextualizedText]:

    if not hasattr(_process_batch_with_retry_async, 'api_client'):
        token_tracker = TokenUsageTracker(tpm_limit=200000, model="gpt-4o-mini")
        backoff_handler = ExponentialBackoffHandler(base_delay=1.0, max_delay=300.0, max_retries=7)
        _process_batch_with_retry_async.api_client = RateLimitAwareAPIClient(token_tracker, backoff_handler)
    
    api_client = _process_batch_with_retry_async.api_client
    
    if verbose:
        usage_summary = api_client.get_usage_summary()
        logger.info(f"Interpreter: 배치 처리 시작 - {usage_summary}")
        logger.info(f"Interpreter: 배치 크기 {len(batch)}")
    
    try:
        batch_texts = [elem["content"]["markdown"] for elem in batch]
        total_text = background_information + " ".join(batch_texts)
        estimated_tokens = api_client.token_tracker.estimate_batch_tokens(
            batch_texts,
            prompt_overhead=len(background_information) // 4 + 500
        )
        
        can_proceed, wait_time = api_client.token_tracker.can_make_request(estimated_tokens)
        if not can_proceed:
            if verbose:
                logger.info(f"Interpreter: 토큰 한도 근접으로 {wait_time}초 대기...")
            await asyncio.sleep(wait_time)
        
        async def batch_context_call():
            return _process_batch(batch, background_information, chain)
        
        results = await api_client.safe_api_call(
            batch_context_call,
            total_text,
            request_type="contextualization"
        )
        
        if verbose:
            logger.info(f"Interpreter: 배치 처리 완료 - {len(results)}개 결과")
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            logger.error(f"Interpreter: 배치 처리 실패: {error_msg}")
        
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            if len(batch) > 1:
                if verbose:
                    logger.info(f"Interpreter: Rate Limit로 인한 실패, 배치를 절반으로 분할하여 재시도")
                
                mid = len(batch) // 2
                first_half = batch[:mid]
                second_half = batch[mid:]
                
                first_results = await _process_batch_with_retry_async(first_half, background_information, chain, verbose)
                second_results = await _process_batch_with_retry_async(second_half, background_information, chain, verbose)
                
                return first_results + second_results
        
        if verbose:
            logger.info("Interpreter: 개별 처리로 fallback")
        return await _process_individual_fallback_async(batch, background_information, chain, verbose)


async def _process_individual_fallback_async(
    batch: List[dict], 
    background_information: str, 
    chain,
    verbose: bool = True
) -> List[ContextualizedText]:

    results = []
    
    if hasattr(_process_batch_with_retry_async, 'api_client'):
        api_client = _process_batch_with_retry_async.api_client
    else:
        token_tracker = TokenUsageTracker(tpm_limit=200000, model="gpt-4o-mini")
        backoff_handler = ExponentialBackoffHandler(base_delay=1.0, max_delay=300.0, max_retries=5)
        api_client = RateLimitAwareAPIClient(token_tracker, backoff_handler)
    
    for elem in batch:
        max_individual_retries = 3
        for retry in range(max_individual_retries):
            try:
                element_text = elem["content"]["markdown"]
                estimated_tokens = len(element_text + background_information) // 4 + 300
                can_proceed, wait_time = api_client.token_tracker.can_make_request(estimated_tokens)
                
                if not can_proceed:
                    if verbose:
                        logger.info(f"Interpreter: 개별 처리 토큰 대기 {wait_time:.1f}초")
                    await asyncio.sleep(wait_time)
                
                single_batch_data = [{
                    "text": element_text,
                    "background_information": background_information,
                }]
                
                async def single_context_call():
                    return chain.batch(single_batch_data)
                
                single_result = await api_client.safe_api_call(
                    single_context_call,
                    element_text + background_information,
                    request_type="individual_contextualization"
                )
                
                results.append(single_result[0])
                
                if verbose:
                    logger.info(f"  개별 요소 처리 완료")
                break
                
            except Exception as individual_error:
                error_str = str(individual_error).lower()
                is_rate_limit = ("rate_limit" in error_str or "429" in error_str or 
                                "quota" in error_str or "exceeded" in error_str)
                
                if retry < max_individual_retries - 1:
                    wait_time = (5 * (2 ** retry)) if is_rate_limit else (2 * (2 ** retry))
                    if verbose:
                        logger.warning(f"Interpreter: 개별 처리 재시도 {retry + 1}/{max_individual_retries}, {wait_time}초 대기")
                    await asyncio.sleep(wait_time)
                else:
                    if verbose:
                        logger.error(f"  개별 요소 처리 실패: {str(individual_error)}")
                    results.append(ContextualizedText(
                        contextualized_text=elem["content"]["markdown"]
                    ))
    
    return results

def _process_batch_sync_fallback(
    batch: List[dict], 
    background_information: str, 
    chain,
    verbose: bool = True
) -> List[ContextualizedText]:

    current_batch_size = len(batch)
    current_batch = batch.copy()
    
    for retry_count in range(INTERPRETER_MAX_RETRIES):
        try:
            if verbose:
                logger.info(f"Interpreter: 배치 크기 {current_batch_size}로 처리 시도 {retry_count + 1}/{INTERPRETER_MAX_RETRIES}")
            
            return _process_batch(current_batch, background_information, chain)
            
        except Exception as e:
            if verbose:
                logger.error(f"Interpreter: 배치 처리 실패 ({retry_count + 1}/{INTERPRETER_MAX_RETRIES}): {str(e)}")
            
            if retry_count < INTERPRETER_MAX_RETRIES - 1:
                new_batch_size = max(1, int(current_batch_size * INTERPRETER_BATCH_REDUCTION_FACTOR))
                if new_batch_size < current_batch_size:
                    current_batch = current_batch[:new_batch_size]
                    current_batch_size = new_batch_size
                    if verbose:
                        logger.info(f"Interpreter: 배치 크기를 {current_batch_size}로 감소")
                
                wait_time = INTERPRETER_RETRY_DELAY * (2 ** retry_count)
                if verbose:
                    logger.info(f"Interpreter: {wait_time}초 대기 후 재시도")
                time.sleep(wait_time)
            else:
                if verbose:
                    logger.info("Interpreter: 개별 처리로 fallback")
                return _process_individual_fallback(batch, background_information, chain, verbose)
    
    return _process_individual_fallback(batch, background_information, chain, verbose)


def _process_individual_fallback(
    batch: List[dict], 
    background_information: str, 
    chain,
    verbose: bool = True
) -> List[ContextualizedText]:

    results = []
    for elem in batch:
        try:
            single_batch_data = [{
                "text": elem["content"]["markdown"],
                "background_information": background_information,
            }]
            single_result = chain.batch(single_batch_data)
            results.append(single_result[0])
            
            if verbose:
                logger.info(f"  개별 요소 처리 완료")
                
        except Exception as individual_error:
            if verbose:
                logger.error(f"  개별 요소 처리 실패: {str(individual_error)}")
            results.append(ContextualizedText(
                contextualized_text=elem["content"]["markdown"]
            ))
    
    return results


def contextualize_text(
    state: ParseState,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    processable_categories: List[str] = None,
    verbose: bool = True
) -> dict:

    if processable_categories is None:
        processable_categories = PROCESSABLE_CATEGORIES
    
    elements_by_page = _group_elements_by_page(state["elements_from_parser"])
    
    if verbose:
        logger.info(f"문맥화 처리 시작 - 총 {len(elements_by_page)}개 페이지")
        logger.info(f"설정: batch_size={batch_size}, max_tokens={max_tokens}")
    
    for page_idx, (page, elements) in enumerate(elements_by_page.items()):
        if verbose:
            logger.info(f"페이지 {page} 처리 중 ({page_idx + 1}/{len(elements_by_page)}) - {len(elements)}개 요소")
        
        total_batches = (len(elements) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(elements), batch_size):
            batch = elements[batch_idx : batch_idx + batch_size]
            current_batch_num = (batch_idx // batch_size) + 1
            
            if verbose:
                logger.info(f"  배치 {current_batch_num}/{total_batches} 처리 중 - {len(batch)}개 요소")
            
            try:
                background_information = _create_background_information(
                    batch, page, openai_extractor
                )
                
                if verbose:
                    info_size = len(background_information)
                    estimated_tokens = info_size // 2.5
                    logger.debug(f"    배경정보 크기: {info_size}자 (~{estimated_tokens:.0f} 토큰)")
                
                contextualized_results = _process_batch_with_retry(
                    batch, background_information, chain, verbose
                )
                
                for elem, result in zip(batch, contextualized_results):
                    elem["content"]["contextualize_text"] = result.contextualized_text
                    
                if verbose:
                    logger.info(f"  배치 {current_batch_num}/{total_batches} 완료")
                    
            except Exception as e:
                if verbose:
                    logger.error(f"  배치 {current_batch_num}/{total_batches} 처리 중 오류: {str(e)}")
                    logger.info(f"  배치 크기를 1로 줄여서 재시도...")
                
                for elem in batch:
                    try:
                        single_batch = [elem]
                        background_information = _create_background_information(
                            single_batch, page, openai_extractor
                        )
                        contextualized_results = _process_batch_with_retry(
                            single_batch, background_information, chain, verbose
                        )
                        elem["content"]["contextualize_text"] = contextualized_results[0].contextualized_text
                        
                        if verbose:
                            logger.info(f"    개별 요소 처리 완료")
                            
                    except Exception as individual_error:
                        if verbose:
                            logger.error(f"    개별 요소 처리 실패: {str(individual_error)}")
                        elem["content"]["contextualize_text"] = elem["content"]["markdown"]
        
        if verbose:
            logger.info(f"페이지 {page} 처리 완료")

    if verbose:
        logger.info("전체 문맥화 처리 완료")

    return {"elements_from_parser": state["elements_from_parser"]}
