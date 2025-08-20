# GraphRecursionError 해결 가이드

## 📌 문제 진단

### 에러 메시지
```
GraphRecursionError: Recursion limit of 25 reached without hitting a stop condition.
You can increase the limit by setting the `recursion_limit` config key.
```

### 발생 원인
- **LangGraph 기본 재귀 제한**: 25회
- **H-DDU 워크플로우 복잡도**: 18개 이상의 노드가 연결된 복잡한 그래프
- **처리 단계**: PDF 파싱 → DocYOLO/Docling 분석 → Assembly 병합 → 이미지/테이블 추출 → 번역 → 문맥화 → 엔티티 병합
- **실행 시간**: 약 81분(4873초) 후 에러 발생

## ✅ 해결 방법

### 1. 환경변수 기반 설정 (구현 완료)

#### **1.1 .env 파일 수정**
```bash
# LangGraph 워크플로우 설정 (hddu/complete_workflow.py)
# =============================================================================

# LangGraph 재귀 제한 설정 (기본값: 25)
# 복잡한 워크플로우에서 GraphRecursionError 방지를 위해 증가
# 권장값: 간단(25-50), 중간(50-100), 복잡(100-150), 매우 복잡(150-200)
LANGGRAPH_RECURSION_LIMIT=150
```

#### **1.2 hddu/config.py 수정**
```python
# LangGraph 워크플로우 설정
# =============================================================================

# LangGraph 재귀 제한 설정 (기본값: 150, 범위: 25-300)
# GraphRecursionError 방지를 위해 복잡한 워크플로우에서는 높은 값 필요
LANGGRAPH_RECURSION_LIMIT = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "150"))
```

#### **1.3 hddu/complete_workflow.py 수정**
```python
# Import 추가
from hddu.config import LANGGRAPH_RECURSION_LIMIT
from langgraph.errors import GraphRecursionError

# workflow.invoke 호출 시 recursion_limit 적용
config = {
    "configurable": {"thread_id": "assembled_workflow_thread"},
    "recursion_limit": LANGGRAPH_RECURSION_LIMIT
}
final_state = workflow.invoke(initial_state, config=config)
```

### 2. 모니터링 추가 (구현 완료)

#### **워크플로우 실행 전 로깅**
```python
logger.info("📊 워크플로우 구성 정보:")
logger.info(f"  - 노드 수: {len(workflow.nodes) if hasattr(workflow, 'nodes') else 'N/A'}")
logger.info(f"  - 재귀 제한: {LANGGRAPH_RECURSION_LIMIT}")
logger.info(f"  - 환경: Assembled 기반 워크플로우")
```

### 3. 에러 핸들링 개선 (구현 완료)

#### **GraphRecursionError 명시적 처리**
```python
except GraphRecursionError as e:
    total_time = time.time() - start_time
    logger.error(f"❌ GraphRecursionError: 재귀 제한 초과 (실행 시간: {total_time:.2f}초)")
    logger.error(f"현재 재귀 제한: {LANGGRAPH_RECURSION_LIMIT}")
    logger.error("해결 방법:")
    logger.error("  1. .env 파일에서 LANGGRAPH_RECURSION_LIMIT 값을 증가시키세요 (예: 200)")
    logger.error("  2. 워크플로우 복잡도를 줄이거나 배치 크기를 줄여보세요")
    logger.error(f"상세 에러: {e}")
    raise
```

## 📊 권장 설정 값

| 워크플로우 복잡도 | 노드 수 | 권장 recursion_limit | 용도 |
|-----------------|--------|---------------------|------|
| 간단 | 1-5 | 25-50 | 테스트, 단순 파싱 |
| 중간 | 6-10 | 50-100 | 일반 문서 처리 |
| 복잡 | 11-20 | 100-150 | H-DDU 표준 워크플로우 |
| 매우 복잡 | 20+ | 150-200 | 대용량/복잡한 문서 |

## 🔄 변경 파일 목록

1. **.env** - LANGGRAPH_RECURSION_LIMIT=150 추가
2. **hddu/config.py** - LANGGRAPH_RECURSION_LIMIT 환경변수 읽기 추가
3. **hddu/complete_workflow.py**
   - config import 추가
   - GraphRecursionError import 추가
   - workflow.invoke config에 recursion_limit 추가 (2곳)
   - 모니터링 로깅 추가
   - GraphRecursionError 전용 에러 핸들링 추가

## 🧪 테스트 방법

### 1. 환경변수 확인
```bash
grep LANGGRAPH_RECURSION_LIMIT .env
# 출력: LANGGRAPH_RECURSION_LIMIT=150
```

### 2. 워크플로우 실행
```bash
uv run test_complete_workflow.py
```

### 3. 로그 확인
```bash
# 실행 시작 시 다음과 같은 로그가 출력되어야 함:
# 📊 워크플로우 구성 정보:
#   - 노드 수: 18
#   - 재귀 제한: 150
#   - 환경: Complete 워크플로우
```

### 4. 에러가 계속 발생할 경우
1. `.env` 파일에서 `LANGGRAPH_RECURSION_LIMIT` 값을 200으로 증가
2. 배치 크기를 줄여서 테스트 (예: batch_size=1)
3. 테스트 페이지를 제한하여 실행 (예: test_page=5)

## 📝 추가 최적화 제안

### 1. 동적 재귀 제한 계산
```python
def calculate_recursion_limit(node_count, edge_count):
    """워크플로우 복잡도에 따른 동적 재귀 제한 계산"""
    base_limit = 25
    node_factor = node_count * 3
    edge_factor = edge_count * 2
    
    calculated = base_limit + node_factor + edge_factor
    
    # 최소 50, 최대 300으로 제한
    return min(max(calculated, 50), 300)
```

### 2. 워크플로우 단순화
- 병렬 처리 가능한 노드들을 식별하여 순차 실행 대신 병렬 실행
- 불필요한 중간 단계 제거
- 조건부 실행으로 필요한 노드만 실행

### 3. 체크포인트 활용
```python
# 중간 상태 저장으로 실패 시 재시작 가능
checkpointer = MemorySaver()
workflow.compile(checkpointer=checkpointer)
```

## 📅 변경 이력

- **2025-08-20**: GraphRecursionError 해결을 위한 환경변수 기반 설정 구현
- **작성자**: Claude Code Assistant
- **검토**: H-DDU 프로젝트 팀

## 📚 참고 자료

- [LangGraph Troubleshooting - GRAPH_RECURSION_LIMIT](https://python.langchain.com/docs/troubleshooting/errors/GRAPH_RECURSION_LIMIT)
- [LangGraph Configuration Guide](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/graph-api.md)
- H-DDU 프로젝트 문서