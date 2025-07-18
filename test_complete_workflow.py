import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hddu.complete_workflow import run_complete_workflow
from hddu.logging_config import get_logger, init_project_logging

logger = get_logger(__name__)


def main():
    
    init_project_logging()
    
    logger.info("=" * 60)
    logger.info("Complete workflow test started")
    logger.info("=" * 60)
    
    pdf_filepath = "data/TEST.pdf"
    
    if not os.path.exists(pdf_filepath):
        logger.error(f"Test file not found: {pdf_filepath}")
        logger.error(f"Current working directory: {os.getcwd()}")
        return False
    
    logger.info(f"Test file found: {pdf_filepath}")
    
    batch_size = 2
    test_page = None
    verbose = True
    
    logger.info("Test settings:")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Test page: {'all' if test_page is None else test_page}")
    logger.info(f"  - Verbose: {verbose}")
    logger.info("")
    
    try:
        logger.info("Workflow execution started...")
        final_state = run_complete_workflow(
            pdf_filepath=pdf_filepath,
            batch_size=batch_size,
            test_page=test_page,
            verbose=verbose
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Workflow test successful!")
        logger.info("=" * 60)
        
        logger.info("\nResult summary:")
        logger.info("-" * 40)
        
        if "elements" in final_state:
            logger.info(f"Processed elements: {len(final_state['elements'])}")
        
        if "documents" in final_state:
            logger.info(f"Generated LangChain documents: {len(final_state['documents'])}")
        
        if "comprehensive_markdown" in final_state:
            logger.info(f"Generated comprehensive markdown: {final_state['comprehensive_markdown']}")
            
        if "documents_pickle_path" in final_state:
            logger.info(f"Documents pickle: {final_state['documents_pickle_path']}")
            
        if "final_state_json_path" in final_state:
            logger.info(f"Final state JSON: {final_state['final_state_json_path']}")
            logger.info(f"Final state pickle: {final_state['final_state_pickle_path']}")
        
        logger.info("\nGenerated files:")
        logger.info("-" * 40)
        
        export_dir = "export"
        if os.path.exists(export_dir):
            for file in os.listdir(export_dir):
                filepath = os.path.join(export_dir, file)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    logger.info(f"  {file} ({size:,} bytes)")
        
        data_images_dir = "data/images"
        if os.path.exists(data_images_dir):
            logger.info(f"\nGenerated images folder:")
            for category in ["figure", "table", "chart"]:
                category_dir = os.path.join(data_images_dir, category)
                if os.path.exists(category_dir):
                    image_count = len([f for f in os.listdir(category_dir) if f.endswith('.png')])
                    logger.info(f"  {category}/: {image_count} images")
        
        logger.info("\nComplete workflow test successful!")
        return True
        
    except Exception as e:
        logger.error(f"\nWorkflow execution error:")
        logger.error(f"   {type(e).__name__}: {e}")
        logger.debug("Detailed stack trace:", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)