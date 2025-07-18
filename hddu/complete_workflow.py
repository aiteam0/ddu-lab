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
        logger.info("Workflow components initialization started...")
    
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
        logger.info(f"Document Parser nodes created ({parser_time:.2f}s)")
        logger.info(f"Settings: batch_size={batch_size}, test_page={test_page}")
    
    nodes_start = time.time()
    
    if verbose:
        logger.debug("Workflow nodes creation in progress...")
    
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
        logger.info(f"Created ({nodes_time:.2f}s)")

    graph_start = time.time()
    parent_workflow = StateGraph(ParseState)

    if verbose:
        logger.debug("Adding nodes to workflow graph...")

    node_names = [
        ("split_pdf_node", split_pdf_node, "PDF splitting"),
        ("document_parse_node", document_parse_node, "Document parsing"),
        ("post_document_parse_node", post_document_parse_node, "Post-document parsing"),
        ("working_queue_node", working_queue_node, "Working queue management"),
        ("parser_save_state_node", parser_save_state_node, "Parsing state saving"),
        ("save_image_node", save_image_node, "Image saving"),
        ("refine_content_node", refine_content_node, "Text refinement"),
        ("add_translation", add_translation_module, "Translation"),
        ("contextualize_text", contextualize_text, "Contextualization"),
        ("create_elements_node", create_elements_node, "Element creation"),
        ("page_elements_extractor", page_elements_extractor_node, "Page-based element extraction"),
        ("image_entity_extractor", image_entity_extractor_node, "Image entity extraction"),
        ("table_entity_extractor", table_entity_extractor_node, "Table entity extraction"),
        ("merge_entity_node", merge_entity_node, "Entity merging"),
        ("reconstruct_elements_node", reconstruct_elements_node, "Element reconstruction"),
        ("generate_comprehensive_markdown", generate_comprehensive_markdown_node, "Markdown generation"),
        ("langchain_document_node", langchain_document_node, "LangChain document generation"),
        ("save_final_state", save_final_state_node, "Final state saving")
    ]
    
    for node_name, node_instance, description in node_names:
        parent_workflow.add_node(node_name, node_instance)
        if verbose:
            logger.debug(f"  Added node: {node_name} ({description})")
    
    if verbose: logger.info(f"{len(node_names)} nodes added to workflow")

    if verbose:
        logger.debug("Defining connections between nodes...")
    
    parent_workflow.add_edge("split_pdf_node", "working_queue_node")
    parent_workflow.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    parent_workflow.add_edge("document_parse_node", "working_queue_node")
    parent_workflow.add_edge("post_document_parse_node", "parser_save_state_node")
    
    if verbose:
        logger.debug("Parsing stage connections completed (conditional loop included)")
    
    parent_workflow.add_edge("parser_save_state_node", "save_image_node")
    
    sequential_edges = [
        ("save_image_node", "refine_content_node", "Image saving -> Text refinement"),
        ("refine_content_node", "add_translation", "Text refinement -> Translation"),
        ("add_translation", "contextualize_text", "Translation -> Contextualization"),
        ("contextualize_text", "create_elements_node", "Contextualization -> Element creation"),
        ("create_elements_node", "page_elements_extractor", "Element creation -> Page-based extraction")
    ]
    
    for from_node, to_node, description in sequential_edges:
        parent_workflow.add_edge(from_node, to_node)
        if verbose:
            logger.debug(f"  Sequential connection: {description}")
    
    if verbose:
        logger.debug("Parallel processing section setup:")
        logger.debug("  Page-based extraction -> Image/table entity extraction (parallel)")
    
    parent_workflow.add_edge("page_elements_extractor", "image_entity_extractor")
    parent_workflow.add_edge("page_elements_extractor", "table_entity_extractor")
    
    parent_workflow.add_edge("image_entity_extractor", "merge_entity_node")
    parent_workflow.add_edge("table_entity_extractor", "merge_entity_node")
    
    if verbose:
        logger.debug("  Image/table entity extraction -> Entity merging")
    
    parent_workflow.add_edge("merge_entity_node", "reconstruct_elements_node")
    
    if verbose:
        logger.debug("  Element reconstruction -> Markdown/document generation (parallel)")
    
    parent_workflow.add_edge("reconstruct_elements_node", "generate_comprehensive_markdown")
    parent_workflow.add_edge("reconstruct_elements_node", "langchain_document_node")
    
    parent_workflow.add_edge("generate_comprehensive_markdown", "save_final_state")
    
    if verbose:
        logger.debug("  Markdown generation -> Final state saving")
        logger.debug("  Document generation completed in parallel")

    parent_workflow.set_entry_point("split_pdf_node")
    
    if verbose:
        logger.debug("Workflow starting point set: split_pdf_node")

    compile_start = time.time()
    parent_graph = parent_workflow.compile(checkpointer=MemorySaver())
    compile_time = time.time() - compile_start
    
    graph_total_time = time.time() - graph_start
    
    if verbose:
        logger.info(f"Workflow graph compiled ({compile_time:.2f}s)")
        logger.info(f"Total graph construction time: {graph_total_time:.2f}s")
    
    if verbose:
        setup_verbose_logging(verbose)

        total_creation_time = parser_time + nodes_time + graph_total_time
        logger.info("=" * 50)
        logger.info("Complete integrated workflow creation completed")
        logger.info("=" * 50)
        logger.info(f"Total creation time: {total_creation_time:.2f}s")
        logger.info(f"  - Parser nodes: {parser_time:.2f}s")
        logger.info(f"  - Other nodes: {nodes_time:.2f}s")
        logger.info(f"  - Graph construction: {graph_total_time:.2f}s")
        logger.info("=" * 50)

    return parent_graph


def run_complete_workflow(pdf_filepath, batch_size=1, test_page=None, verbose=True):
    
    start_time = time.time()
    
    if verbose:
        setup_verbose_logging(verbose)
        logger.info(f"Workflow execution started: {pdf_filepath}")
        logger.info(f"Batch size: {batch_size}")
        if test_page is not None:
            logger.info(f"Test page: {test_page}")
        
        if PSUTIL_AVAILABLE:
            try:
                memory_info = psutil.virtual_memory()
                logger.info(f"System memory: {memory_info.total // (1024**3):.1f}GB (usage: {memory_info.percent:.1f}%)")
                logger.info(f"CPU cores: {psutil.cpu_count()} (usage: {psutil.cpu_percent(interval=1):.1f}%)")
            except Exception as e:
                logger.debug(f"System information collection failed: {e}")

        file_size = 0
        if os.path.exists(pdf_filepath):
            file_size = os.path.getsize(pdf_filepath) / (1024**2)
            logger.info(f"Input file size: {file_size:.1f}MB")
        else:
            logger.warning(f"Input file not found: {pdf_filepath}")
        
        logger.info("-" * 50)
    
    workflow_creation_start = time.time()
    workflow = create_complete_workflow(
        batch_size=batch_size, 
        test_page=test_page, 
        verbose=verbose
    )
    workflow_creation_time = time.time() - workflow_creation_start
    
    if verbose:
        logger.info(f"Workflow creation completed ({workflow_creation_time:.2f}s)")
    
    initial_state = {"filepath": pdf_filepath}
    
    try:
        execution_start = time.time()
        logger.info("Workflow execution started...")
        
        config = {"configurable": {"thread_id": "test_workflow_thread"}}
        final_state = workflow.invoke(initial_state, config=config)
        
        execution_time = time.time() - execution_start
        logger.info(f"Workflow execution completed ({execution_time:.2f}s)")
        
        total_time = time.time() - start_time
        
        if verbose:
            logger.info("=" * 60)
            logger.info("Workflow execution completed - Summary of final results")
            logger.info("=" * 60)
            
            logger.info(f"Total execution time: {total_time:.2f}s")
            logger.info(f"Workflow creation time: {workflow_creation_time:.2f}s")
            logger.info(f"Actual processing time: {execution_time:.2f}s")
            logger.info(f"Processing speed: {file_size/total_time if file_size > 0 and total_time > 0 else 'N/A'} MB/s")
            
            logger.info("-" * 40)
            logger.info("Summary of processing results:")
            
            if "elements_from_parser" in final_state:
                elements = final_state["elements_from_parser"]
                logger.info(f"Parsed elements: {len(elements)}")
                
                categories = {}
                for element in elements:
                    cat = element.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
                
                for category, count in sorted(categories.items()):
                    logger.info(f"  {category}: {count} elements")
            
            if "image_paths" in final_state:
                image_count = len(final_state["image_paths"])
                logger.info(f"Saved images: {image_count}")
            
            if "documents" in final_state:
                doc_count = len(final_state['documents'])
                logger.info(f"Generated LangChain documents: {doc_count}")

                total_chars = sum(len(doc.page_content) for doc in final_state['documents'])
                avg_chars = total_chars / doc_count if doc_count > 0 else 0
                logger.info(f"Total document length: {total_chars:,} characters")
                logger.info(f"Average document length: {avg_chars:.0f} characters")
            
            logger.info("-" * 40)
            logger.info("Generated files:")
            
            if "comprehensive_markdown" in final_state:
                logger.info(f"Markdown file: {final_state['comprehensive_markdown']}")
            
            if "documents_pickle_path" in final_state:
                logger.info(f"Documents pickle: {final_state['documents_pickle_path']}")
                
            if "final_state_json_path" in final_state:
                logger.info(f"Final state JSON: {final_state['final_state_json_path']}")
                logger.info(f"Final state pickle: {final_state['final_state_pickle_path']}")
            
            if PSUTIL_AVAILABLE:
                try:
                    final_memory = psutil.virtual_memory()
                    logger.info(f"Final memory usage: {final_memory.percent:.1f}%")
                except Exception:
                    pass
                
            logger.info("=" * 60)
        
        return final_state
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Workflow execution error occurred (execution time: {total_time:.2f}s)")
        logger.error(f"Error content: {e}")
        logger.error(f"Error location: {type(e).__name__}")
        
        logger.debug("Detailed stack trace:", exc_info=True)
        
        try:
            if 'final_state' in locals():
                logger.info("Partial processing results exist.")
                if "elements_from_parser" in final_state:
                    logger.info(f"Processed elements: {len(final_state['elements_from_parser'])}")
        except Exception:
            pass
            
        raise 