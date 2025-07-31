import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hddu.logging_config import get_logger, init_project_logging
from hddu.complete_workflow import (
    load_assembled_state,
    create_workflow_from_assembled,
    run_workflow_from_assembled
)

logger = get_logger(__name__)

def main():
    """메인 함수"""
    
    init_project_logging()
    
    logger.info("=" * 60)
    logger.info("Assembled 기반 워크플로우 테스트 시작")
    logger.info("=" * 60)
    
    if len(sys.argv) < 2:
        logger.error("사용법: python test_workflow_from_assembled.py <assembled_json_path>")
        logger.error("예시: python test_workflow_from_assembled.py intermediate/file_assembled.json")
        
        default_files = [
            "intermediate/디지털정부혁신_추진계획_TEST1P_0000_0000_assembled.json",
            "intermediate/디지털정부혁신_추진계획_TEST1P_0000_0000_assembled_result.json"
        ]
        
        found_file = None
        for default_file in default_files:
            if os.path.exists(default_file):
                found_file = default_file
                break
        
        if found_file:
            logger.info(f"기본 파일을 사용합니다: {found_file}")
            assembled_path = found_file
        else:
            logger.error("기본 파일도 찾을 수 없습니다.")
            return False
    else:
        assembled_path = sys.argv[1]
    
    if not os.path.exists(assembled_path):
        logger.error(f"Assembled 파일을 찾을 수 없습니다: {assembled_path}")
        logger.error(f"현재 작업 디렉토리: {os.getcwd()}")
        return False
    
    logger.info(f"Assembled 파일 확인: {assembled_path}")
    
    verbose = True
    
    logger.info("테스트 설정:")
    logger.info(f"  - Assembled 파일: {assembled_path}")
    logger.info(f"  - 상세 로그: {verbose}")
    logger.info("")
    
    try:
        logger.info("워크플로우 실행 시작...")
        final_state = run_workflow_from_assembled(
            assembled_path=assembled_path,
            verbose=verbose
        )
        
        logger.info("\\n" + "=" * 60)
        logger.info("워크플로우 테스트 성공!")
        logger.info("=" * 60)
        
        logger.info("\\n결과 요약:")
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
        
        logger.info("\\n생성된 파일들 확인:")
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
            logger.info(f"\\n생성된 이미지 폴더:")
            for category in ["figure", "table", "chart", "heading1", "paragraph", "footer"]:
                category_dir = os.path.join(data_images_dir, category)
                if os.path.exists(category_dir):
                    image_count = len([f for f in os.listdir(category_dir) if f.endswith('.png')])
                    if image_count > 0:
                        logger.info(f"  {category}/: {image_count}개 이미지")
        
        logger.info("\\nAssembled 기반 워크플로우 테스트가 성공적으로 완료되었습니다!")
        return True
        
    except Exception as e:
        logger.error(f"\\n워크플로우 실행 중 오류 발생:")
        logger.error(f"   {type(e).__name__}: {e}")
        logger.debug("상세 스택 트레이스:", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)