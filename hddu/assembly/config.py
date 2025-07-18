import os
import logging
from typing import List, Dict, Any, Tuple, Literal
from dotenv import load_dotenv

from ..config import (
    TEXT_LLM_PROVIDER, TEXT_OPENAI_MODEL, TEXT_AZURE_DEPLOYMENT, TEXT_OLLAMA_MODEL,
    VISION_LLM_PROVIDER, VISION_OPENAI_MODEL, VISION_AZURE_DEPLOYMENT, VISION_OLLAMA_MODEL,
    OPENAI_API_KEY, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION,
    OLLAMA_BASE_URL, ASSEMBLY_MODE, SIMILARITY_THRESHOLD, BATCH_SIZE, LLM_BASED_ID_ASSIGNMENT
)

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

Element = Dict[str, Any]
PageData = Dict[Literal['docling', 'docyolo'], List[Element]]
GroupedData = Dict[int, PageData]
MatchedPair = Tuple[Element, Element]
MatchResult = Tuple[List[MatchedPair], List[Element], List[Element]]

LLM_PROVIDER = TEXT_LLM_PROVIDER.upper()
TEXT_LLM_PROVIDER_UPPER = TEXT_LLM_PROVIDER.upper()
VISION_LLM_PROVIDER_UPPER = VISION_LLM_PROVIDER.upper()

OPENAI_TEXT_MODEL = TEXT_OPENAI_MODEL
OPENAI_VISION_MODEL = VISION_OPENAI_MODEL
OLLAMA_TEXT_MODEL = TEXT_OLLAMA_MODEL
OLLAMA_VISION_MODEL = VISION_OLLAMA_MODEL
AZURE_TEXT_DEPLOYMENT = TEXT_AZURE_DEPLOYMENT
AZURE_VISION_DEPLOYMENT = VISION_AZURE_DEPLOYMENT

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "OPENAI")).upper()
EMBEDDING_AZURE_DEPLOYMENT = os.getenv("EMBEDDING_AZURE_DEPLOYMENT", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))
EMBEDDING_OPENAI_MODEL = os.getenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-small")
EMBEDDING_OLLAMA_MODEL = os.getenv("EMBEDDING_OLLAMA_MODEL", "mxbai-embed-large")
AZURE_EMBEDDING_DEPLOYMENT = EMBEDDING_AZURE_DEPLOYMENT
OPENAI_EMBEDDING_MODEL = EMBEDDING_OPENAI_MODEL
OLLAMA_EMBEDDING_MODEL = EMBEDDING_OLLAMA_MODEL

def validate_model_config(provider: str, model_type: str = "") -> bool:
    if provider == "AZURE":
        return bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)
    elif provider == "OPENAI":
        return bool(OPENAI_API_KEY)
    elif provider == "OLLAMA":
        return True
    else:
        logger.error(f"Unsupported provider: {provider}")
        return False

def validate_config():
    validation_errors = []
    
    valid_modes = ["merge", "docling_only", "docyolo_only"]
    if ASSEMBLY_MODE not in valid_modes:
        validation_errors.append(f"Invalid ASSEMBLY_MODE: {ASSEMBLY_MODE}. Must be one of: {valid_modes}")
    
    if not validate_model_config(TEXT_LLM_PROVIDER_UPPER, "text"):
        validation_errors.append(f"Text LLM configuration invalid for provider: {TEXT_LLM_PROVIDER_UPPER}")
    
    if not validate_model_config(VISION_LLM_PROVIDER_UPPER, "vision"):
        validation_errors.append(f"Vision LLM configuration invalid for provider: {VISION_LLM_PROVIDER_UPPER}")
    
    if not validate_model_config(EMBEDDING_PROVIDER, "embedding"):
        validation_errors.append(f"Embedding model configuration invalid for provider: {EMBEDDING_PROVIDER}")
    
    if validation_errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(validation_errors))
    
    logger.info(f"Assembly mode: {ASSEMBLY_MODE}")
    logger.info(f"Text LLM: {TEXT_LLM_PROVIDER_UPPER}")
    logger.info(f"Vision LLM: {VISION_LLM_PROVIDER_UPPER}")
    logger.info(f"Embedding: {EMBEDDING_PROVIDER}")

validate_config()