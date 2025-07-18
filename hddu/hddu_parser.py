import os
from hddu.utils import SplitPDFFilesNode
from hddu.state import ParseState
from hddu.parser import (
    DocumentParseNode,
    PostDocumentParseNode,
    WorkingQueueNode,
    SaveStateNode,
    continue_parse,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from hddu.preprocessing import (
    CreateElementsNode,
    MergeEntityNode,
    ReconstructElementsNode,
    LangChainDocumentNode,
)
from hddu.export import ExportHTML, ExportMarkdown, ExportTableCSV, ExportImage
from hddu.extractor import (
    PageElementsExtractorNode,
    ImageEntityExtractorNode,
    TableEntityExtractorNode,
)
from langchain_utils.graphs import visualize_graph
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_parser_graph(
    batch_size: int = 30,
    test_page: int = None,
    verbose: bool = True,
    visualize: bool = False,
):
    split_pdf_node = SplitPDFFilesNode(
        batch_size=batch_size, test_page=test_page, verbose=verbose
    )

    document_parse_node = DocumentParseNode(
        lang="auto", verbose=verbose
    )

    post_document_parse_node = PostDocumentParseNode(verbose=verbose)
    working_queue_node = WorkingQueueNode(verbose=verbose)
    save_state_node = SaveStateNode(verbose=verbose)

    workflow = StateGraph(ParseState)

    workflow.add_node("split_pdf_node", split_pdf_node)
    workflow.add_node("document_parse_node", document_parse_node)
    workflow.add_node("post_document_parse_node", post_document_parse_node)
    workflow.add_node("working_queue_node", working_queue_node)
    workflow.add_node("save_state_node", save_state_node)

    workflow.add_edge("split_pdf_node", "working_queue_node")
    workflow.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    workflow.add_edge("document_parse_node", "working_queue_node")
    workflow.add_edge("post_document_parse_node", "save_state_node")

    workflow.set_entry_point("split_pdf_node")

    document_parse_graph = workflow.compile(checkpointer=MemorySaver())
    if visualize:
        visualize_graph(document_parse_graph)
    return document_parse_graph