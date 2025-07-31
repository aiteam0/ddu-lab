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
    logger.info("완전한 워크플로우 테스트 시작")
    logger.info("=" * 60)
    
    pdf_filepath = "data/TEST.pdf"
    
    if not os.path.exists(pdf_filepath):
        logger.error(f"테스트 파일을 찾을 수 없습니다: {pdf_filepath}")
        logger.error(f"현재 작업 디렉토리: {os.getcwd()}")
        return False
    
    logger.info(f"테스트 파일 확인: {pdf_filepath}")
    
    batch_size = 2
    test_page = None
    verbose = True
    
    logger.info("테스트 설정:")
    logger.info(f"  - 배치 크기: {batch_size}")
    logger.info(f"  - 테스트 페이지: {'전체' if test_page is None else test_page}")
    logger.info(f"  - 상세 로그: {verbose}")
    logger.info("")
    
    try:
        logger.info("워크플로우 실행 시작...")
        final_state = run_complete_workflow(
            pdf_filepath=pdf_filepath,
            batch_size=batch_size,
            test_page=test_page,
            verbose=verbose
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("워크플로우 테스트 성공!")
        logger.info("=" * 60)
        
        logger.info("\n결과 요약:")
        logger.info("-" * 40)
        
        if "elements" in final_state:
            logger.info(f"처리된 요소 수: {len(final_state['elements'])}")
        
        if "documents" in final_state:
            logger.info(f"생성된 LangChain 문서 수: {len(final_state['documents'])}")
        
        if "comprehensive_markdown" in final_state:
            logger.info(f"생성된 마크다운: {final_state['comprehensive_markdown']}")
            
        if "documents_pickle_path" in final_state:
            logger.info(f"Documents pickle: {final_state['documents_pickle_path']}")
            
        if "final_state_json_path" in final_state:
            logger.info(f"최종 상태 JSON: {final_state['final_state_json_path']}")
            logger.info(f"최종 상태 pickle: {final_state['final_state_pickle_path']}")
        
        logger.info("\n생성된 파일들 확인:")
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
            logger.info(f"\n생성된 이미지 폴더:")
            for category in ["figure", "table", "chart"]:
                category_dir = os.path.join(data_images_dir, category)
                if os.path.exists(category_dir):
                    image_count = len([f for f in os.listdir(category_dir) if f.endswith('.png')])
                    logger.info(f"  {category}/: {image_count}개 이미지")
        
        logger.info("\n전체 워크플로우 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        logger.error(f"\n워크플로우 실행 중 오류 발생:")
        logger.error(f"   {type(e).__name__}: {e}")
        logger.debug("상세 스택 트레이스:", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 