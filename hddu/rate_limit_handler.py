import time
import random
import logging
import asyncio
from typing import List, Dict, Optional, Tuple, Any
import re
from dataclasses import dataclass

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.getLogger(__name__).warning("tiktoken을 사용할 수 없습니다. 토큰 수 추정이 부정확할 수 있습니다.")


@dataclass
class UsageRecord:
    """토큰 사용 기록"""
    timestamp: float
    tokens_used: int
    request_type: str = "unknown"


class TokenUsageTracker:
    """토큰 사용량 추적 및 Rate Limit 예측 클래스"""
    
    def __init__(self, tpm_limit: int = 200000, model: str = "gpt-4o-mini"):
        """
        Args:
            tpm_limit: 분당 토큰 한도 (Tokens Per Minute)
            model: 사용하는 OpenAI 모델명
        """
        self.tpm_limit = tpm_limit
        self.model = model
        self.usage_history: List[UsageRecord] = []
        self.logger = logging.getLogger(__name__)
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None
            
    def estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 예측"""
        if not text:
            return 0
            
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                self.logger.debug(f"tiktoken 인코딩 실패: {e}")
                
        korean_chars = len([c for c in text if ord(c) > 127])
        english_chars = len(text) - korean_chars
        return int(korean_chars / 2 + english_chars / 4)
        
    def estimate_batch_tokens(self, texts: List[str], prompt_overhead: int = 500) -> int:
        """배치 처리 시 총 토큰 수 예측"""
        content_tokens = sum(self.estimate_tokens(text) for text in texts)
        total_overhead = prompt_overhead + len(texts) * 50
        return content_tokens + total_overhead
        
    def get_current_minute_usage(self) -> int:
        """현재 1분간 사용된 토큰 수 반환"""
        now = time.time()
        minute_ago = now - 60
        
        self.usage_history = [record for record in self.usage_history if record.timestamp > minute_ago]
        
        current_usage = sum(record.tokens_used for record in self.usage_history)
        return current_usage
        
    def can_make_request(self, estimated_tokens: int) -> Tuple[bool, int]:
        """
        요청 가능 여부와 대기 시간 반환
        
        Returns:
            (가능 여부, 대기 시간(초))
        """
        current_usage = self.get_current_minute_usage()
        remaining = self.tpm_limit - current_usage
        
        if estimated_tokens <= remaining:
            return True, 0
        else:
            if self.usage_history:
                oldest_record = min(self.usage_history, key=lambda r: r.timestamp)
                wait_time = max(0, 65 - (time.time() - oldest_record.timestamp))
            else:
                wait_time = 65
                
            self.logger.info(f"토큰 부족: 현재 {current_usage}/{self.tpm_limit}, 필요 {estimated_tokens}, {wait_time:.1f}초 대기")
            return False, int(wait_time)
            
    def record_usage(self, tokens_used: int, request_type: str = "api_call"):
        """실제 토큰 사용량 기록"""
        record = UsageRecord(
            timestamp=time.time(),
            tokens_used=tokens_used,
            request_type=request_type
        )
        self.usage_history.append(record)
        
        current_total = self.get_current_minute_usage()
        self.logger.debug(f"토큰 사용 기록: +{tokens_used} ({request_type}), 현재 총 {current_total}/{self.tpm_limit}")
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """현재 사용량 통계 반환"""
        current_usage = self.get_current_minute_usage()
        usage_rate = (current_usage / self.tpm_limit) * 100
        
        return {
            "current_usage": current_usage,
            "limit": self.tpm_limit,
            "usage_percentage": usage_rate,
            "remaining": self.tpm_limit - current_usage,
            "records_count": len(self.usage_history)
        }


class ExponentialBackoffHandler:
    """지수 백오프 재시도 처리 클래스"""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 300.0, max_retries: int = 7):
        """
        Args:
            base_delay: 기본 대기 시간 (초)
            max_delay: 최대 대기 시간 (초)
            max_retries: 최대 재시도 횟수
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
    def calculate_delay(self, attempt: int, is_rate_limit: bool = False) -> float:
        """
        재시도 대기 시간 계산
        
        Args:
            attempt: 현재 시도 횟수 (0부터 시작)
            is_rate_limit: Rate Limit 오류인지 여부
            
        Returns:
            대기 시간 (초)
        """
        if is_rate_limit:
            base_wait = max(60, self.base_delay * (2 ** attempt))
        else:
            base_wait = self.base_delay * (2 ** attempt)
            
        delay = min(base_wait, self.max_delay)
        
        jitter = delay * 0.2 * (random.random() - 0.5)
        final_delay = max(0, delay + jitter)
        
        return final_delay
        
    def parse_rate_limit_message(self, error_message: str) -> Optional[float]:
        """
        Rate Limit 오류 메시지에서 제안된 대기 시간 파싱
        
        Args:
            error_message: 오류 메시지
            
        Returns:
            제안된 대기 시간 (초), 파싱 실패 시 None
        """
        patterns = [
            r"try again in (\d+)ms",
            r"try again in (\d+(?:\.\d+)?)s",
            r"try again in (\d+) ms",
            r"try again in (\d+(?:\.\d+)?) seconds?"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                value = float(match.group(1))
                if "ms" in pattern:
                    return value / 1000
                else:
                    return value
                    
        return None
        
    async def retry_with_backoff(self, 
                                func,
                                *args,
                                rate_limit_exceptions: tuple = None,
                                **kwargs) -> Any:
        """
        지수 백오프로 함수 재시도 실행
        
        Args:
            func: 실행할 함수
            *args, **kwargs: 함수 인자들
            rate_limit_exceptions: Rate Limit 오류로 간주할 예외 타입들
            
        Returns:
            함수 실행 결과
            
        Raises:
            마지막 시도의 예외
        """
        if rate_limit_exceptions is None:
            rate_limit_exceptions = ()
            
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    is_rate_limit = isinstance(last_exception, rate_limit_exceptions)
                    delay = self.calculate_delay(attempt - 1, is_rate_limit)
                    
                    if is_rate_limit and hasattr(last_exception, 'message'):
                        suggested_delay = self.parse_rate_limit_message(str(last_exception.message))
                        if suggested_delay:
                            delay = max(delay, suggested_delay + 5)
                            
                    self.logger.info(f"재시도 {attempt}/{self.max_retries}: {delay:.1f}초 대기 중...")
                    await asyncio.sleep(delay)
                    
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                if attempt > 0:
                    self.logger.info(f"재시도 성공 (시도 {attempt + 1}/{self.max_retries + 1})")
                    
                return result
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                if attempt == self.max_retries:
                    self.logger.error(f"최대 재시도 횟수 초과: {error_type}: {str(e)}")
                    raise e
                else:
                    is_rate_limit = isinstance(e, rate_limit_exceptions)
                    log_level = logging.INFO if is_rate_limit else logging.WARNING
                    self.logger.log(log_level, f"시도 {attempt + 1} 실패 ({error_type}): {str(e)}")
                    
        raise last_exception


class RateLimitAwareAPIClient:
    """Rate Limit을 인식하는 API 클라이언트 래퍼"""
    
    def __init__(self, 
                 token_tracker: Optional[TokenUsageTracker] = None,
                 backoff_handler: Optional[ExponentialBackoffHandler] = None):
        """
        Args:
            token_tracker: 토큰 사용량 추적기
            backoff_handler: 백오프 재시도 처리기
        """
        self.token_tracker = token_tracker or TokenUsageTracker()
        self.backoff_handler = backoff_handler or ExponentialBackoffHandler()
        self.logger = logging.getLogger(__name__)
        
    async def safe_api_call(self,
                           api_func,
                           prompt_text: str,
                           request_type: str = "api_call",
                           *args,
                           **kwargs) -> Any:
        """
        안전한 API 호출 (토큰 모니터링 + 재시도)
        
        Args:
            api_func: 호출할 API 함수
            prompt_text: 프롬프트 텍스트 (토큰 수 계산용)
            request_type: 요청 타입 (로깅용)
            *args, **kwargs: API 함수 인자들
            
        Returns:
            API 응답 결과
        """
        estimated_tokens = self.token_tracker.estimate_tokens(prompt_text)
        
        can_proceed, wait_time = self.token_tracker.can_make_request(estimated_tokens)
        if not can_proceed:
            self.logger.info(f"토큰 한도 근접으로 {wait_time}초 대기 중...")
            await asyncio.sleep(wait_time)
            
        async def api_call_wrapper():
            response = await api_func(*args, **kwargs) if asyncio.iscoroutinefunction(api_func) else api_func(*args, **kwargs)
            
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                actual_tokens = response.usage.total_tokens
                self.token_tracker.record_usage(actual_tokens, request_type)
            else:
                self.token_tracker.record_usage(estimated_tokens, request_type)
                
            return response
            
        rate_limit_exceptions = (Exception,)
        
        try:
            result = await self.backoff_handler.retry_with_backoff(
                api_call_wrapper,
                rate_limit_exceptions=rate_limit_exceptions
            )
            return result
            
        except Exception as e:
            self.logger.error(f"API 호출 최종 실패 ({request_type}): {type(e).__name__}: {str(e)}")
            raise
            
    def get_usage_summary(self) -> str:
        """현재 토큰 사용량 요약 문자열 반환"""
        stats = self.token_tracker.get_usage_stats()
        return (f"토큰 사용량: {stats['current_usage']:,}/{stats['limit']:,} "
                f"({stats['usage_percentage']:.1f}%) "
                f"남은 토큰: {stats['remaining']:,}")


def create_default_tracker(tpm_limit: int = 200000, model: str = "gpt-4o-mini") -> TokenUsageTracker:
    """기본 설정으로 토큰 추적기 생성"""
    return TokenUsageTracker(tpm_limit=tpm_limit, model=model)


def create_default_client(tpm_limit: int = 200000, model: str = "gpt-4o-mini") -> RateLimitAwareAPIClient:
    """기본 설정으로 Rate Limit 인식 클라이언트 생성"""
    tracker = create_default_tracker(tpm_limit, model)
    backoff = ExponentialBackoffHandler()
    return RateLimitAwareAPIClient(tracker, backoff)


default_token_tracker = None
default_api_client = None


def get_default_tracker() -> TokenUsageTracker:
    """모듈 레벨 기본 토큰 추적기 반환 (lazy 초기화)"""
    global default_token_tracker
    if default_token_tracker is None:
        default_token_tracker = create_default_tracker()
    return default_token_tracker


def get_default_client() -> RateLimitAwareAPIClient:
    """모듈 레벨 기본 API 클라이언트 반환 (lazy 초기화)"""
    global default_api_client
    if default_api_client is None:
        default_api_client = create_default_client()
    return default_api_client