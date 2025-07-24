import os
import logging
from typing import Optional, Dict, Any, Literal
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ModelType = Literal["text", "vision"]
ProviderType = Literal["openai", "azure", "ollama", "anthropic", "google"]


TEXT_LLM_PROVIDER = os.getenv("TEXT_LLM_PROVIDER", "openai").lower()
TEXT_OPENAI_MODEL = os.getenv("TEXT_OPENAI_MODEL", "gpt-4o-mini")
TEXT_AZURE_DEPLOYMENT = os.getenv("TEXT_AZURE_DEPLOYMENT", "gpt-4o-mini")
TEXT_OLLAMA_MODEL = os.getenv("TEXT_OLLAMA_MODEL", "llama3.1:8b")
TEXT_ANTHROPIC_MODEL = os.getenv("TEXT_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
TEXT_GOOGLE_MODEL = os.getenv("TEXT_GOOGLE_MODEL", "gemini-1.5-flash")


VISION_LLM_PROVIDER = os.getenv("VISION_LLM_PROVIDER", "openai").lower()
VISION_OPENAI_MODEL = os.getenv("VISION_OPENAI_MODEL", "gpt-4o-mini")
VISION_AZURE_DEPLOYMENT = os.getenv("VISION_AZURE_DEPLOYMENT", "gpt-4o-mini")
VISION_OLLAMA_MODEL = os.getenv("VISION_OLLAMA_MODEL", "llava:13b")
VISION_ANTHROPIC_MODEL = os.getenv("VISION_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
VISION_GOOGLE_MODEL = os.getenv("VISION_GOOGLE_MODEL", "gemini-1.5-flash")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TEXT_OLLAMA_BASE_URL = os.getenv("TEXT_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
VISION_OLLAMA_BASE_URL = os.getenv("VISION_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
EMBEDDING_OLLAMA_BASE_URL = os.getenv("EMBEDDING_OLLAMA_BASE_URL", OLLAMA_BASE_URL)


TRANSLATION_TEMPERATURE = float(os.getenv("TRANSLATION_TEMPERATURE", "0.1"))
TRANSLATION_BATCH_SIZE = int(os.getenv("TRANSLATION_BATCH_SIZE", "20"))
TRANSLATION_MAX_RETRIES = int(os.getenv("TRANSLATION_MAX_RETRIES", "3"))
TRANSLATION_RETRY_DELAY = float(os.getenv("TRANSLATION_RETRY_DELAY", "1.0"))
TRANSLATION_BATCH_REDUCTION_FACTOR = float(os.getenv("TRANSLATION_BATCH_REDUCTION_FACTOR", "0.5"))
TRANSLATION_TARGET_LANGUAGE = os.getenv("TRANSLATION_TARGET_LANGUAGE", "auto")

INTERPRETER_TEMPERATURE = float(os.getenv("INTERPRETER_TEMPERATURE", "0.1"))
INTERPRETER_MAX_TOKENS = int(os.getenv("INTERPRETER_MAX_TOKENS", "12000"))
INTERPRETER_BATCH_SIZE = int(os.getenv("INTERPRETER_BATCH_SIZE", "3"))
INTERPRETER_MAX_RETRIES = int(os.getenv("INTERPRETER_MAX_RETRIES", "3"))
INTERPRETER_RETRY_DELAY = float(os.getenv("INTERPRETER_RETRY_DELAY", "1.0"))
INTERPRETER_BATCH_REDUCTION_FACTOR = float(os.getenv("INTERPRETER_BATCH_REDUCTION_FACTOR", "0.5"))

EXTRACTOR_TEMPERATURE = float(os.getenv("EXTRACTOR_TEMPERATURE", "0"))
EXTRACTOR_BATCH_SIZE = int(os.getenv("EXTRACTOR_BATCH_SIZE", "10"))
EXTRACTOR_MAX_TOKENS = int(os.getenv("EXTRACTOR_MAX_TOKENS", "8000"))
EXTRACTOR_MAX_RETRIES = int(os.getenv("EXTRACTOR_MAX_RETRIES", "3"))
EXTRACTOR_RETRY_DELAY = float(os.getenv("EXTRACTOR_RETRY_DELAY", "1.0"))
EXTRACTOR_BATCH_REDUCTION_FACTOR = float(os.getenv("EXTRACTOR_BATCH_REDUCTION_FACTOR", "0.5"))

# 전처리 모듈 설정 (RefineContentNode가 정규식 기반으로 변경되어 더 이상 사용되지 않음)
# PREPROCESSING_TEMPERATURE = float(os.getenv("PREPROCESSING_TEMPERATURE", "0.1"))
# PREPROCESSING_MAX_TOKENS = int(os.getenv("PREPROCESSING_MAX_TOKENS", "12000"))
# PREPROCESSING_BATCH_SIZE = int(os.getenv("PREPROCESSING_BATCH_SIZE", "10"))
# PREPROCESSING_MAX_RETRIES = int(os.getenv("PREPROCESSING_MAX_RETRIES", "3"))
# PREPROCESSING_RETRY_DELAY = float(os.getenv("PREPROCESSING_RETRY_DELAY", "1.0"))
# PREPROCESSING_BATCH_REDUCTION_FACTOR = float(os.getenv("PREPROCESSING_BATCH_REDUCTION_FACTOR", "0.5"))

ASSEMBLY_MODE = os.getenv("ASSEMBLY_MODE", "merge").lower()
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
LLM_BASED_ID_ASSIGNMENT = os.getenv("LLM_BASED_ID_ASSIGNMENT", "disabled").lower()


def get_model_config(model_type: ModelType) -> Dict[str, Any]:
    if model_type == "text":
        return {
            "provider": TEXT_LLM_PROVIDER,
            "models": {
                "openai": TEXT_OPENAI_MODEL,
                "azure": TEXT_AZURE_DEPLOYMENT,
                "ollama": TEXT_OLLAMA_MODEL,
                "anthropic": TEXT_ANTHROPIC_MODEL,
                "google": TEXT_GOOGLE_MODEL
            }
        }
    elif model_type == "vision":
        return {
            "provider": VISION_LLM_PROVIDER,
            "models": {
                "openai": VISION_OPENAI_MODEL,
                "azure": VISION_AZURE_DEPLOYMENT,
                "ollama": VISION_OLLAMA_MODEL,
                "anthropic": VISION_ANTHROPIC_MODEL,
                "google": VISION_GOOGLE_MODEL
            }
        }
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")


def validate_provider_config(provider: ProviderType) -> bool:
    if provider == "openai":
        return bool(OPENAI_API_KEY)
    elif provider == "azure":
        return bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)
    elif provider == "anthropic":
        return bool(ANTHROPIC_API_KEY)
    elif provider == "google":
        return bool(GOOGLE_API_KEY)
    elif provider == "ollama":
        return True
    else:
        logger.warning(f"알 수 없는 provider: {provider}")
        return False


def create_text_model(temperature: float = 0.1, max_tokens: int = 16000, **kwargs):
    config = get_model_config("text")
    provider = config["provider"]
    model_name = config["models"][provider]
    
    if not validate_provider_config(provider):
        raise ValueError(f"Provider '{provider}' 설정이 유효하지 않습니다. API 키를 확인해주세요.")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model_name,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            **kwargs
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            base_url=TEXT_OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=max_tokens,
            **kwargs
        )
    else:
        raise ValueError(f"지원하지 않는 텍스트 모델 provider: {provider}")


def create_vision_model(temperature: float = 0.1, max_tokens: int = 8000, **kwargs):
    config = get_model_config("vision")
    provider = config["provider"]
    model_name = config["models"][provider]
    
    if not validate_provider_config(provider):
        raise ValueError(f"Provider '{provider}' 설정이 유효하지 않습니다. API 키를 확인해주세요.")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model_name,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            **kwargs
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            base_url=VISION_OLLAMA_BASE_URL,
            temperature=temperature,
            num_predict=max_tokens,
            **kwargs
        )
    else:
        raise ValueError(f"지원하지 않는 비전 모델 provider: {provider}")


def create_embedding_model(**kwargs):

    provider = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()
    
    if not validate_provider_config(provider):
        raise ValueError(f"Provider '{provider}' 설정이 유효하지 않습니다. API 키를 확인해주세요.")
    
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("EMBEDDING_OPENAI_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(
            model=model,
            api_key=OPENAI_API_KEY,
            **kwargs
        )
    elif provider == "azure":
        from langchain_openai import AzureOpenAIEmbeddings
        deployment = os.getenv("EMBEDDING_AZURE_DEPLOYMENT", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))
        return AzureOpenAIEmbeddings(
            azure_deployment=deployment,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            **kwargs
        )
    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        model = os.getenv("EMBEDDING_OLLAMA_MODEL", "mxbai-embed-large")
        return OllamaEmbeddings(
            model=model,
            base_url=EMBEDDING_OLLAMA_BASE_URL,
            **kwargs
        )
    else:
        raise ValueError(f"지원하지 않는 임베딩 모델 provider: {provider}")


def get_provider_info() -> Dict[str, Any]:
    return {
        "text_provider": TEXT_LLM_PROVIDER,
        "vision_provider": VISION_LLM_PROVIDER,
        "text_model": get_model_config("text")["models"][TEXT_LLM_PROVIDER],
        "vision_model": get_model_config("vision")["models"][VISION_LLM_PROVIDER],
    }



def validate_assembly_config() -> bool:

    valid_modes = ["merge", "docling_only", "docyolo_only"]
    if ASSEMBLY_MODE not in valid_modes:
        logger.error(f"Invalid ASSEMBLY_MODE: {ASSEMBLY_MODE}. Must be one of: {valid_modes}")
        return False
    
    valid_llm_modes = ["disabled", "simple", "advanced"]
    if LLM_BASED_ID_ASSIGNMENT not in valid_llm_modes:
        logger.error(f"Invalid LLM_BASED_ID_ASSIGNMENT: {LLM_BASED_ID_ASSIGNMENT}. Must be one of: {valid_llm_modes}")
        return False
    
    return True


def validate_all_configs():

    errors = []
    
    if not validate_provider_config(TEXT_LLM_PROVIDER):
        errors.append(f"Text LLM provider '{TEXT_LLM_PROVIDER}' 설정 오류")
    
    if not validate_provider_config(VISION_LLM_PROVIDER):
        errors.append(f"Vision LLM provider '{VISION_LLM_PROVIDER}' 설정 오류")
    
    if not validate_assembly_config():
        errors.append(f"Assembly configuration invalid")
    
    if errors:
        logger.error("설정 검증 실패:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError("설정이 올바르지 않습니다. .env 파일을 확인해주세요.")
    
    info = get_provider_info()
    logger.info("LLM 설정 로드 완료:")
    logger.info(f"Text: {info['text_provider']} ({info['text_model']})")
    logger.info(f"Vision: {info['vision_provider']} ({info['vision_model']})")
    
    logger.info(f"Assembly mode: {ASSEMBLY_MODE}")
    
    logger.info("RefineContentNode: 정규식 기반 텍스트 정제 (LLM 불필요)")


try:
    validate_all_configs()
except Exception as e:
    logger.warning(f"설정 검증 실패, 기본값 사용: {e}")
    logger.info("RefineContentNode는 정규식 기반으로 작동하므로 LLM 설정 없이도 동작합니다.") 