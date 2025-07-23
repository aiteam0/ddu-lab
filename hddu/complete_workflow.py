from hddu.utils import SplitPDFFilesNode
from hddu.parser import (
    DocumentParseNode,
    PostDocumentParseNode,
    WorkingQueueNode,
    SaveStateNode,
    continue_parse,
)

from hddu.logging_config import get_logger, setup_verbose_logging
import time
import os

logger = get_logger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil 모듈을 찾을 수 없습니다. 시스템 리소스 모니터링이 비활성화됩니다.")

from langgraph.graph import StateGraph, END
from hddu.state import ParseState
from langchain_utils.graphs import visualize_graph
from langgraph.checkpoint.memory import MemorySaver

from hddu.preprocessing import (
    SaveImageNode,
    RefineContentNode,
    CreateElementsNode,
    MergeEntityNode,
    ReconstructElementsNode,
    GenerateComprehensiveMarkdownNode,
    LangChainDocumentNode,
    SaveFinalStateNode,
)

from hddu.translate import add_translation_module
from hddu.interpreter import contextualize_text
from hddu.extractor import (
    PageElementsExtractorNode,
    ImageEntityExtractorNode,
    TableEntityExtractorNode,
)


def create_complete_workflow(batch_size=1, test_page=None, verbose=True):

    if verbose:
        logger.info("워크플로우 구성 요소 초기화 시작...")
    
    parser_start = time.time()
    
    split_pdf_node = SplitPDFFilesNode(
        batch_size=batch_size, test_page=test_page, verbose=verbose
    )
    
    document_parse_node = DocumentParseNode(
        lang="auto", verbose=verbose
    )
    
    post_document_parse_node = PostDocumentParseNode(verbose=verbose)
    
    working_queue_node = WorkingQueueNode(verbose=verbose)
    
    parser_save_state_node = SaveStateNode(verbose=verbose)
    
    parser_time = time.time() - parser_start
    
    if verbose:
        logger.info(f"Document Parser 노드들 생성 완료 ({parser_time:.2f}초)")
        logger.info(f"설정: batch_size={batch_size}, test_page={test_page}")
    
    nodes_start = time.time()
    
    if verbose:
        logger.debug("워크플로우 노드들 생성 중...")
    
    save_image_node = SaveImageNode(verbose=verbose)
    refine_content_node = RefineContentNode(verbose=verbose)
    create_elements_node = CreateElementsNode(verbose=verbose)
    page_elements_extractor_node = PageElementsExtractorNode(verbose=verbose)
    image_entity_extractor_node = ImageEntityExtractorNode(verbose=verbose)
    table_entity_extractor_node = TableEntityExtractorNode(verbose=verbose)
    merge_entity_node = MergeEntityNode(verbose=verbose)
    reconstruct_elements_node = ReconstructElementsNode(verbose=verbose)
    generate_comprehensive_markdown_node = GenerateComprehensiveMarkdownNode(verbose=verbose)
    langchain_document_node = LangChainDocumentNode(verbose=verbose)
    save_final_state_node = SaveFinalStateNode(verbose=verbose)
    
    nodes_time = time.time() - nodes_start
    
    if verbose:
        logger.info(f"16개 워크플로우 노드 생성 완료 ({nodes_time:.2f}초)")

    graph_start = time.time()
    parent_workflow = StateGraph(ParseState)

    if verbose:
        logger.debug("워크플로우 그래프에 노드들 추가 중...")

    node_names = [
        ("split_pdf_node", split_pdf_node, "PDF 분할"),
        ("document_parse_node", document_parse_node, "문서 파싱"),
        ("post_document_parse_node", post_document_parse_node, "파싱 후처리"),
        ("working_queue_node", working_queue_node, "작업 큐 관리"),
        ("parser_save_state_node", parser_save_state_node, "파싱 상태 저장"),
        ("save_image_node", save_image_node, "이미지 저장"),
        ("refine_content_node", refine_content_node, "텍스트 정제"),
        ("add_translation", add_translation_module, "번역"),
        ("contextualize_text", contextualize_text, "문맥화"),
        ("create_elements_node", create_elements_node, "요소 생성"),
        ("page_elements_extractor", page_elements_extractor_node, "페이지별 요소 추출"),
        ("image_entity_extractor", image_entity_extractor_node, "이미지 엔티티 추출"),
        ("table_entity_extractor", table_entity_extractor_node, "테이블 엔티티 추출"),
        ("merge_entity_node", merge_entity_node, "엔티티 병합"),
        ("reconstruct_elements_node", reconstruct_elements_node, "요소 재구성"),
        ("generate_comprehensive_markdown", generate_comprehensive_markdown_node, "마크다운 생성"),
        ("langchain_document_node", langchain_document_node, "LangChain 문서 생성"),
        ("save_final_state", save_final_state_node, "최종 상태 저장")
    ]
    
    for node_name, node_instance, description in node_names:
        parent_workflow.add_node(node_name, node_instance)
        if verbose:
            logger.debug(f"  노드 추가: {node_name} ({description})")
    
    if verbose:
        logger.info(f"{len(node_names)}개 노드가 워크플로우에 추가되었습니다.")

    if verbose:
        logger.debug("노드 간 연결(엣지) 설정 중...")
    
    parent_workflow.add_edge("split_pdf_node", "working_queue_node")
    parent_workflow.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    parent_workflow.add_edge("document_parse_node", "working_queue_node")
    parent_workflow.add_edge("post_document_parse_node", "parser_save_state_node")
    
    if verbose:
        logger.debug("  파싱 단계 연결 완료 (조건부 루프 포함)")
    
    parent_workflow.add_edge("parser_save_state_node", "save_image_node")
    
    sequential_edges = [
        ("save_image_node", "refine_content_node", "이미지 저장 -> 텍스트 정제"),
        ("refine_content_node", "add_translation", "텍스트 정제 -> 번역"),
        ("add_translation", "contextualize_text", "번역 -> 문맥화"),
        ("contextualize_text", "create_elements_node", "문맥화 -> 요소 생성"),
        ("create_elements_node", "page_elements_extractor", "요소 생성 -> 페이지별 추출")
    ]
    
    for from_node, to_node, description in sequential_edges:
        parent_workflow.add_edge(from_node, to_node)
        if verbose:
            logger.debug(f"  순차 연결: {description}")
    
    if verbose:
        logger.debug("  병렬 처리 구간 설정:")
        logger.debug("    페이지별 추출 -> 이미지/테이블 엔티티 추출 (병렬)")
    
    parent_workflow.add_edge("page_elements_extractor", "image_entity_extractor")
    parent_workflow.add_edge("page_elements_extractor", "table_entity_extractor")
    
    parent_workflow.add_edge("image_entity_extractor", "merge_entity_node")
    parent_workflow.add_edge("table_entity_extractor", "merge_entity_node")
    
    if verbose:
        logger.debug("    이미지/테이블 엔티티 추출 -> 엔티티 병합")
    
    parent_workflow.add_edge("merge_entity_node", "reconstruct_elements_node")
    
    if verbose:
        logger.debug("    요소 재구성 -> 마크다운/문서 생성 (병렬)")
    
    parent_workflow.add_edge("reconstruct_elements_node", "generate_comprehensive_markdown")
    parent_workflow.add_edge("reconstruct_elements_node", "langchain_document_node")
    
    parent_workflow.add_edge("generate_comprehensive_markdown", "save_final_state")
    
    if verbose:
        logger.debug("    마크다운 생성 -> 최종 상태 저장")
        logger.debug("    문서 생성은 병렬로 완료")

    parent_workflow.set_entry_point("split_pdf_node")
    
    if verbose:
        logger.debug("워크플로우 시작점 설정: split_pdf_node")

    compile_start = time.time()
    parent_graph = parent_workflow.compile(checkpointer=MemorySaver())
    compile_time = time.time() - compile_start
    
    graph_total_time = time.time() - graph_start
    
    if verbose:
        logger.info(f"워크플로우 그래프 컴파일 완료 ({compile_time:.2f}초)")
        logger.info(f"전체 그래프 구성 시간: {graph_total_time:.2f}초")
    
    if verbose:
        setup_verbose_logging(verbose)
        
        total_creation_time = parser_time + nodes_time + graph_total_time
        logger.info("=" * 50)
        logger.info("완전한 통합 워크플로우 생성 완료")
        logger.info("=" * 50)
        logger.info(f"총 생성 시간: {total_creation_time:.2f}초")
        logger.info(f"  - Parser 노드들: {parser_time:.2f}초")
        logger.info(f"  - 기타 노드들: {nodes_time:.2f}초")
        logger.info(f"  - 그래프 구성: {graph_total_time:.2f}초")
        
        logger.info(f"노드 구성: {len(node_names)}개 노드")
        logger.info("통합 워크플로우 구조:")
        logger.info("  단계 1: PDF 파싱 (조건부 루프)")
        logger.info("    → PDF 분할 → 작업큐 ↔ 문서파싱 → 후처리 → 상태저장")
        logger.info("  단계 2: 이미지 및 텍스트 처리")
        logger.info("    → 이미지 저장 → 텍스트 정제")
        logger.info("  단계 3: 언어 처리")
        logger.info("    → 번역 → 문맥화 → 요소 생성")
        logger.info("  단계 4: 엔티티 추출 (병렬)")
        logger.info("    → 페이지별 추출 ┬→ 이미지 엔티티")
        logger.info("                    └→ 테이블 엔티티")
        logger.info("  단계 5: 최종 처리 (병렬)")
        logger.info("    → 엔티티 병합 → 요소 재구성 ┬→ 마크다운 생성")
        logger.info("                              └→ 문서 생성")
        logger.info("  단계 6: 저장")
        logger.info("    → 최종 상태 저장")
        
        logger.info("병렬 처리 구간: 2개")
        logger.info("  - 이미지/테이블 엔티티 추출")
        logger.info("  - 마크다운/문서 생성")
        logger.info("🔧 SubGraph 제거: metadata/raw_elements 중복 해결")
        logger.info("=" * 50)

    return parent_graph


def create_workflow_from_assembled(verbose=True):

    
    if verbose:
        logger.info("Assembled 기반 워크플로우 구성 시작...")
    
    nodes_start = time.time()
    
    save_image_node = SaveImageNode(verbose=verbose)
    refine_content_node = RefineContentNode(verbose=verbose)
    create_elements_node = CreateElementsNode(verbose=verbose)
    page_elements_extractor_node = PageElementsExtractorNode(verbose=verbose)
    image_entity_extractor_node = ImageEntityExtractorNode(verbose=verbose)
    table_entity_extractor_node = TableEntityExtractorNode(verbose=verbose)
    merge_entity_node = MergeEntityNode(verbose=verbose)
    reconstruct_elements_node = ReconstructElementsNode(verbose=verbose)
    generate_comprehensive_markdown_node = GenerateComprehensiveMarkdownNode(verbose=verbose)
    langchain_document_node = LangChainDocumentNode(verbose=verbose)
    save_final_state_node = SaveFinalStateNode(verbose=verbose)
    
    nodes_time = time.time() - nodes_start
    
    if verbose:
        logger.info(f"11개 워크플로우 노드 생성 완료 ({nodes_time:.2f}초)")
    
    graph_start = time.time()
    parent_workflow = StateGraph(ParseState)
    
    if verbose:
        logger.debug("워크플로우 그래프에 노드들 추가 중...")
    
    node_names = [
        ("save_image_node", save_image_node, "이미지 저장"),
        ("refine_content_node", refine_content_node, "텍스트 정제"),
        ("add_translation", add_translation_module, "번역"),
        ("contextualize_text", contextualize_text, "문맥화"),
        ("create_elements_node", create_elements_node, "요소 생성"),
        ("page_elements_extractor", page_elements_extractor_node, "페이지별 요소 추출"),
        ("image_entity_extractor", image_entity_extractor_node, "이미지 엔티티 추출"),
        ("table_entity_extractor", table_entity_extractor_node, "테이블 엔티티 추출"),
        ("merge_entity_node", merge_entity_node, "엔티티 병합"),
        ("reconstruct_elements_node", reconstruct_elements_node, "요소 재구성"),
        ("generate_comprehensive_markdown", generate_comprehensive_markdown_node, "마크다운 생성"),
        ("langchain_document_node", langchain_document_node, "LangChain 문서 생성"),
        ("save_final_state", save_final_state_node, "최종 상태 저장")
    ]
    
    for node_name, node_instance, description in node_names:
        parent_workflow.add_node(node_name, node_instance)
        if verbose:
            logger.debug(f"  노드 추가: {node_name} ({description})")
    
    if verbose:
        logger.info(f"{len(node_names)}개 노드가 워크플로우에 추가되었습니다.")

    if verbose:
        logger.debug("노드 간 연결(엣지) 설정 중...")
    
    sequential_edges = [
        ("save_image_node", "refine_content_node", "이미지 저장 -> 텍스트 정제"),
        ("refine_content_node", "add_translation", "텍스트 정제 -> 번역"),
        ("add_translation", "contextualize_text", "번역 -> 문맥화"),
        ("contextualize_text", "create_elements_node", "문맥화 -> 요소 생성"),
        ("create_elements_node", "page_elements_extractor", "요소 생성 -> 페이지별 추출")
    ]
    
    for from_node, to_node, description in sequential_edges:
        parent_workflow.add_edge(from_node, to_node)
        if verbose:
            logger.debug(f"  순차 연결: {description}")
    
    if verbose:
        logger.debug("  병렬 처리 구간 설정:")
        logger.debug("    페이지별 추출 -> 이미지/테이블 엔티티 추출 (병렬)")
    
    parent_workflow.add_edge("page_elements_extractor", "image_entity_extractor")
    parent_workflow.add_edge("page_elements_extractor", "table_entity_extractor")
    
    parent_workflow.add_edge("image_entity_extractor", "merge_entity_node")
    parent_workflow.add_edge("table_entity_extractor", "merge_entity_node")
    
    if verbose:
        logger.debug("    이미지/테이블 엔티티 추출 -> 엔티티 병합")
    
    parent_workflow.add_edge("merge_entity_node", "reconstruct_elements_node")
    
    if verbose:
        logger.debug("    요소 재구성 -> 마크다운/문서 생성 (병렬)")
    
    parent_workflow.add_edge("reconstruct_elements_node", "generate_comprehensive_markdown")
    parent_workflow.add_edge("reconstruct_elements_node", "langchain_document_node")
    
    parent_workflow.add_edge("generate_comprehensive_markdown", "save_final_state")
    
    if verbose:
        logger.debug("    마크다운 생성 -> 최종 상태 저장")
        logger.debug("    문서 생성은 병렬로 완료")

    parent_workflow.set_entry_point("save_image_node")
    
    if verbose:
        logger.debug("워크플로우 시작점 설정: save_image_node")

    compile_start = time.time()
    parent_graph = parent_workflow.compile(checkpointer=MemorySaver())
    compile_time = time.time() - compile_start
    
    graph_total_time = time.time() - graph_start
    
    if verbose:
        logger.info(f"Assembled 기반 워크플로우 컴파일 완료 ({compile_time:.2f}초)")
        logger.info(f"전체 그래프 구성 시간: {graph_total_time:.2f}초")
    
    if verbose:
        setup_verbose_logging(verbose)
        
        total_creation_time = nodes_time + graph_total_time
        logger.info("=" * 50)
        logger.info("Assembled 기반 워크플로우 생성 완료")
        logger.info("=" * 50)
        logger.info(f"총 생성 시간: {total_creation_time:.2f}초")
        logger.info(f"  - 노드들: {nodes_time:.2f}초")
        logger.info(f"  - 그래프 구성: {graph_total_time:.2f}초")
        
        logger.info(f"노드 구성: {len(node_names)}개 노드")
        logger.info("Assembled 기반 워크플로우 구조:")
        logger.info("  단계 1: 이미지 및 텍스트 처리")
        logger.info("    → 이미지 저장 → 텍스트 정제")
        logger.info("  단계 2: 언어 처리")
        logger.info("    → 번역 → 문맥화 → 요소 생성")
        logger.info("  단계 3: 엔티티 추출 (병렬)")
        logger.info("    → 페이지별 추출 ┬→ 이미지 엔티티")
        logger.info("                    └→ 테이블 엔티티")
        logger.info("  단계 4: 최종 처리 (병렬)")
        logger.info("    → 엔티티 병합 → 요소 재구성 ┬→ 마크다운 생성")
        logger.info("                              └→ 문서 생성")
        logger.info("  단계 5: 저장")
        logger.info("    → 최종 상태 저장")
        
        logger.info("병렬 처리 구간: 2개")
        logger.info("  - 이미지/테이블 엔티티 추출")
        logger.info("  - 마크다운/문서 생성")
        logger.info("🔧 PDF 파싱 단계 제거: assembled 데이터 활용")
        logger.info("=" * 50)

    return parent_graph


def load_assembled_state(assembled_path):

    import json
    
    logger.info(f"Assembled 파일 로드 중: {assembled_path}")
    
    if not os.path.exists(assembled_path):
        raise FileNotFoundError(f"Assembled 파일을 찾을 수 없습니다: {assembled_path}")
    
    with open(assembled_path, 'r', encoding='utf-8') as f:
        assembled_data = json.load(f)
    
    if "elements" in assembled_data:
        elements = assembled_data["elements"]
    elif "assembled_elements" in assembled_data:
        elements = assembled_data["assembled_elements"]
    else:
        raise ValueError("Assembled 파일에서 elements 데이터를 찾을 수 없습니다.")
    
    base_name = os.path.splitext(os.path.basename(assembled_path))[0]
    if "_assembled" in base_name:
        original_name = base_name.split("_assembled")[0]
    else:
        original_name = base_name
    
    possible_paths = [
        f"data/{original_name}.pdf",
        f"data/{original_name}_TEST1P.pdf",
        f"{original_name}.pdf"
    ]
    
    original_filepath = None
    for path in possible_paths:
        if os.path.exists(path):
            original_filepath = path
            break
    
    if not original_filepath:
        logger.warning(f"원본 PDF 파일을 찾을 수 없습니다. 추정 경로들: {possible_paths}")
        original_filepath = f"data/{original_name}.pdf"
    
    texts_by_page = {}
    for element in elements:
        page = element.get("page", 1)
        if page not in texts_by_page:
            texts_by_page[page] = []
        
        content = element.get("content", {})
        text = content.get("text", "") or content.get("markdown", "") or ""
        if text:
            texts_by_page[page].append(text)
    
    state = {
        "filepath": original_filepath,
        "elements_from_parser": elements,
        "texts_by_page": texts_by_page,
        "assembled_source": assembled_path
    }
    
    logger.info(f"Assembled 데이터 로드 완료:")
    logger.info(f"  - 요소 수: {len(elements)}")
    logger.info(f"  - 페이지 수: {len(texts_by_page)}")
    logger.info(f"  - 원본 파일: {original_filepath}")
    
    return state


def run_workflow_from_assembled(assembled_path, verbose=True):

    start_time = time.time()
    
    if verbose:
        setup_verbose_logging(verbose)
        logger.info(f"Assembled 기반 워크플로우 실행 시작: {assembled_path}")
        
        if PSUTIL_AVAILABLE:
            try:
                memory_info = psutil.virtual_memory()
                logger.info(f"시스템 메모리: {memory_info.total // (1024**3):.1f}GB (사용률: {memory_info.percent:.1f}%)")
                logger.info(f"CPU 코어: {psutil.cpu_count()} (사용률: {psutil.cpu_percent(interval=1):.1f}%)")
            except Exception as e:
                logger.debug(f"시스템 정보 수집 실패: {e}")
        
        if os.path.exists(assembled_path):
            file_size = os.path.getsize(assembled_path) / (1024**2)
            logger.info(f"입력 파일 크기: {file_size:.1f}MB")
        else:
            logger.warning(f"입력 파일을 찾을 수 없습니다: {assembled_path}")
        
        logger.info("-" * 50)
    
    try:
        data_load_start = time.time()
        initial_state = load_assembled_state(assembled_path)
        data_load_time = time.time() - data_load_start
        
        if verbose:
            logger.info(f"Assembled 데이터 로드 완료 ({data_load_time:.2f}초)")
    except Exception as e:
        logger.error(f"Assembled 파일 로드 실패: {e}")
        raise
    
    workflow_creation_start = time.time()
    workflow = create_workflow_from_assembled(verbose=verbose)
    workflow_creation_time = time.time() - workflow_creation_start
    
    if verbose:
        logger.info(f"워크플로우 생성 완료 ({workflow_creation_time:.2f}초)")
    
    try:
        execution_start = time.time()
        logger.info("워크플로우 실행 시작...")
        
        config = {"configurable": {"thread_id": "assembled_workflow_thread"}}
        final_state = workflow.invoke(initial_state, config=config)
        
        execution_time = time.time() - execution_start
        logger.info(f"워크플로우 실행 완료 ({execution_time:.2f}초)")
        
        total_time = time.time() - start_time
        
        if verbose:
            logger.info("=" * 60)
            logger.info("Assembled 기반 워크플로우 실행 완료 - 최종 결과 요약")
            logger.info("=" * 60)
            
            logger.info(f"전체 실행 시간: {total_time:.2f}초")
            logger.info(f"데이터 로드: {data_load_time:.2f}초")
            logger.info(f"워크플로우 생성: {workflow_creation_time:.2f}초")
            logger.info(f"실제 처리 시간: {execution_time:.2f}초")
            
            logger.info("-" * 40)
            logger.info("처리 결과 통계:")
            
            if "elements_from_parser" in final_state:
                elements = final_state["elements_from_parser"]
                logger.info(f"원본 요소 수: {len(elements)}")
                
                categories = {}
                for element in elements:
                    cat = element.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                
                for category, count in sorted(categories.items()):
                    logger.info(f"  {category}: {count}개")
            
            if "image_paths" in final_state:
                image_count = len(final_state["image_paths"])
                logger.info(f"저장된 이미지 수: {image_count}")
            
            if "documents" in final_state:
                doc_count = len(final_state['documents'])
                logger.info(f"생성된 LangChain 문서 수: {doc_count}")
                
                total_chars = sum(len(doc.page_content) for doc in final_state['documents'])
                avg_chars = total_chars / doc_count if doc_count > 0 else 0
                logger.info(f"총 문서 길이: {total_chars:,} 문자")
                logger.info(f"평균 문서 길이: {avg_chars:.0f} 문자")
            
            logger.info("-" * 40)
            logger.info("생성된 파일들:")
            
            if "comprehensive_markdown" in final_state:
                logger.info(f"마크다운 파일: {final_state['comprehensive_markdown']}")
            
            if "documents_pickle_path" in final_state:
                logger.info(f"Documents pickle: {final_state['documents_pickle_path']}")
                
            if "final_state_json_path" in final_state:
                logger.info(f"최종 상태 JSON: {final_state['final_state_json_path']}")
                logger.info(f"최종 상태 pickle: {final_state['final_state_pickle_path']}")
            
            if PSUTIL_AVAILABLE:
                try:
                    final_memory = psutil.virtual_memory()
                    logger.info(f"최종 메모리 사용률: {final_memory.percent:.1f}%")
                except Exception:
                    pass
                
            logger.info("=" * 60)
        
        return final_state
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"워크플로우 실행 중 오류 발생 (실행 시간: {total_time:.2f}초)")
        logger.error(f"오류 내용: {e}")
        logger.error(f"오류 발생 지점: {type(e).__name__}")
        
        logger.debug("상세 스택 트레이스:", exc_info=True)
        
        try:
            if 'final_state' in locals():
                logger.info("부분적 처리 결과가 존재합니다.")
                if "elements_from_parser" in final_state:
                    logger.info(f"처리된 요소 수: {len(final_state['elements_from_parser'])}")
        except Exception:
            pass
            
        raise


def run_complete_workflow(pdf_filepath, batch_size=1, test_page=None, verbose=True):

    start_time = time.time()
    
    if verbose:
        setup_verbose_logging(verbose)
        logger.info(f"워크플로우 실행 시작: {pdf_filepath}")
        logger.info(f"배치 크기: {batch_size}")
        if test_page is not None:
            logger.info(f"테스트 페이지: {test_page}")
        
        if PSUTIL_AVAILABLE:
            try:
                memory_info = psutil.virtual_memory()
                logger.info(f"시스템 메모리: {memory_info.total // (1024**3):.1f}GB (사용률: {memory_info.percent:.1f}%)")
                logger.info(f"CPU 코어: {psutil.cpu_count()} (사용률: {psutil.cpu_percent(interval=1):.1f}%)")
            except Exception as e:
                logger.debug(f"시스템 정보 수집 실패: {e}")
        
        file_size = 0
        if os.path.exists(pdf_filepath):
            file_size = os.path.getsize(pdf_filepath) / (1024**2)
            logger.info(f"입력 파일 크기: {file_size:.1f}MB")
        else:
            logger.warning(f"입력 파일을 찾을 수 없습니다: {pdf_filepath}")
        
        logger.info("-" * 50)
    
    workflow_creation_start = time.time()
    workflow = create_complete_workflow(
        batch_size=batch_size, 
        test_page=test_page, 
        verbose=verbose
    )
    workflow_creation_time = time.time() - workflow_creation_start
    
    if verbose:
        logger.info(f"워크플로우 생성 완료 ({workflow_creation_time:.2f}초)")
    
    initial_state = {"filepath": pdf_filepath}
    
    try:
        execution_start = time.time()
        logger.info("워크플로우 실행 시작...")
        
        config = {"configurable": {"thread_id": "test_workflow_thread"}}
        final_state = workflow.invoke(initial_state, config=config)
        
        execution_time = time.time() - execution_start
        logger.info(f"워크플로우 실행 완료 ({execution_time:.2f}초)")
        
        total_time = time.time() - start_time
        
        if verbose:
            logger.info("=" * 60)
            logger.info("워크플로우 실행 완료 - 최종 결과 요약")
            logger.info("=" * 60)
            
            logger.info(f"전체 실행 시간: {total_time:.2f}초")
            logger.info(f"워크플로우 생성: {workflow_creation_time:.2f}초")
            logger.info(f"실제 처리 시간: {execution_time:.2f}초")
            logger.info(f"처리 속도: {file_size/total_time if file_size > 0 and total_time > 0 else 'N/A'} MB/초")
            
            logger.info("-" * 40)
            logger.info("처리 결과 통계:")
            
            if "elements_from_parser" in final_state:
                elements = final_state["elements_from_parser"]
                logger.info(f"파싱된 요소 수: {len(elements)}")
                
                categories = {}
                for element in elements:
                    cat = element.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                
                for category, count in sorted(categories.items()):
                    logger.info(f"  {category}: {count}개")
            
            if "image_paths" in final_state:
                image_count = len(final_state["image_paths"])
                logger.info(f"저장된 이미지 수: {image_count}")
            
            if "documents" in final_state:
                doc_count = len(final_state['documents'])
                logger.info(f"생성된 LangChain 문서 수: {doc_count}")
                
                total_chars = sum(len(doc.page_content) for doc in final_state['documents'])
                avg_chars = total_chars / doc_count if doc_count > 0 else 0
                logger.info(f"총 문서 길이: {total_chars:,} 문자")
                logger.info(f"평균 문서 길이: {avg_chars:.0f} 문자")
            
            logger.info("-" * 40)
            logger.info("생성된 파일들:")
            
            if "comprehensive_markdown" in final_state:
                logger.info(f"마크다운 파일: {final_state['comprehensive_markdown']}")
            
            if "documents_pickle_path" in final_state:
                logger.info(f"Documents pickle: {final_state['documents_pickle_path']}")
                
            if "final_state_json_path" in final_state:
                logger.info(f"최종 상태 JSON: {final_state['final_state_json_path']}")
                logger.info(f"최종 상태 pickle: {final_state['final_state_pickle_path']}")
            
            if PSUTIL_AVAILABLE:
                try:
                    final_memory = psutil.virtual_memory()
                    logger.info(f"최종 메모리 사용률: {final_memory.percent:.1f}%")
                except Exception:
                    pass
                
            logger.info("=" * 60)
        
        return final_state
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"워크플로우 실행 중 오류 발생 (실행 시간: {total_time:.2f}초)")
        logger.error(f"오류 내용: {e}")
        logger.error(f"오류 발생 지점: {type(e).__name__}")
        
        logger.debug("상세 스택 트레이스:", exc_info=True)
        
        try:
            if 'final_state' in locals():
                logger.info("부분적 처리 결과가 존재합니다.")
                if "elements_from_parser" in final_state:
                    logger.info(f"처리된 요소 수: {len(final_state['elements_from_parser'])}")
        except Exception:
            pass
            
        raise 