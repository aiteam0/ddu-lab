# run_assembly.py

import argparse
import logging
import time
import sys
import os
from assembly.main_assembler import DocumentAssembler
from assembly import config
from assembly.config import logger

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Document Assembly - Assemble parsing results with optimized performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--docling_file", type=str, required=False, 
                       help="Path to the Docling JSON result file.")
    parser.add_argument("--docyolo_file", type=str, required=False, 
                       help="Path to the DocYOLO JSON result file.")
    parser.add_argument("--output_file", type=str, required=True, 
                       help="Path to save the final assembled JSON file.")
    
    parser.add_argument("--async_mode", action='store_true', default=True,
                       help="Use asynchronous mode for LLM calls (default, faster).")
    parser.add_argument("--sync_mode", action='store_true',
                       help="Use synchronous mode (slower, better for debugging).")
    parser.add_argument("--disable_advanced_processing", action='store_true',
                       help="Disable advanced processing (faster, lower quality).")
    parser.add_argument("--verbose", action='store_true', 
                       help="Enable verbose logging (include HTTP requests).")
    
    parser.add_argument("--assembly_mode", type=str, choices=["merge", "docling_only", "docyolo_only"],
                       help="Assembly mode: merge (default), docling_only, or docyolo_only")
    parser.add_argument("--disable_llm_id_assignment", action='store_true',
                       help="Disable LLM-based ID assignment (use coordinate-based sorting)")
    
    parser.add_argument("--docling_only", action='store_true',
                       help="Process only Docling file (same as --assembly_mode docling_only)")
    parser.add_argument("--docyolo_only", action='store_true',
                       help="Process only DocYOLO file (same as --assembly_mode docyolo_only)")
    
    args = parser.parse_args()
    
    validate_input_arguments(args)
    
    apply_config_overrides(args)
    
    if args.verbose:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logger.info("Verbose logging enabled")
    
    use_async = args.async_mode and not args.sync_mode
    
    print_configuration_summary(args, use_async)
    
    start_time = time.time()
    
    try:
        assembler = DocumentAssembler()
        assembler.run(
            docling_path=args.docling_file,
            docyolo_path=args.docyolo_file,
            output_path=args.output_file,
            use_async=use_async
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("✅ ASSEMBLY COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total execution time: {duration:.2f} seconds")
        logger.info(f"💾 Output saved to: {args.output_file}")
        
        provide_performance_feedback(duration, use_async, args)
        
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        logger.error("Please check that the input files exist and paths are correct.")
        return 1
    except Exception as e:
        logger.critical(f"❌ Assembly process failed: {e}", exc_info=True)
        logger.error("Please check the error details above and try again.")
        return 1
    
    return 0

def validate_input_arguments(args):
    
    if not args.docling_file and not args.docyolo_file:
        logger.error("At least one input file (--docling_file or --docyolo_file) must be provided")
        sys.exit(1)
    
    if args.docling_file and not os.path.exists(args.docling_file):
        logger.error(f"Docling file not found: {args.docling_file}")
        sys.exit(1)
    
    if args.docyolo_file and not os.path.exists(args.docyolo_file):
        logger.error(f"DocYOLO file not found: {args.docyolo_file}")
        sys.exit(1)
    
    mode_flags = [args.docling_only, args.docyolo_only]
    if sum(mode_flags) > 1:
        logger.error("Only one assembly mode flag can be specified (--docling_only or --docyolo_only)")
        sys.exit(1)
    
    if args.assembly_mode and any(mode_flags):
        logger.error("Cannot specify both --assembly_mode and convenience flags (--docling_only, --docyolo_only)")
        sys.exit(1)

def apply_config_overrides(args):
    
    if args.docling_only:
        config.ASSEMBLY_MODE = "docling_only"
        logger.info("Assembly mode set to: docling_only (from --docling_only flag)")
    elif args.docyolo_only:
        config.ASSEMBLY_MODE = "docyolo_only"
        logger.info("Assembly mode set to: docyolo_only (from --docyolo_only flag)")
    elif args.assembly_mode:
        config.ASSEMBLY_MODE = args.assembly_mode
        logger.info(f"Assembly mode set to: {args.assembly_mode} (from --assembly_mode)")
    
    if args.disable_llm_id_assignment:
        config.LLM_BASED_ID_ASSIGNMENT = False
        logger.info("LLM-based ID assignment disabled")
    
    if args.disable_advanced_processing:
        config.PROCESS_ALL_IMAGES_INDIVIDUALLY = False
        logger.info("Advanced processing disabled - using legacy methods for speed")
    else:
        logger.info("Using advanced processing (recommended for quality)")

def print_configuration_summary(args, use_async):
    logger.info("=" * 60)
    logger.info("ENHANCED DOCUMENT ASSEMBLY")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  🔧 Processing: Advanced (Unified Visual Processing)")
    logger.info(f"  ⚡ Mode: {'Asynchronous' if use_async else 'Synchronous'}")
    logger.info(f"  🏗️  Assembly Mode: {config.ASSEMBLY_MODE}")
    logger.info(f"  🆔 ID Assignment: {'LLM-based' if config.LLM_BASED_ID_ASSIGNMENT else 'Coordinate-based'}")
    logger.info(f"  📄 Docling: {args.docling_file or 'None'}")
    logger.info(f"  📄 DocYOLO: {args.docyolo_file or 'None'}")
    logger.info(f"  💾 Output: {args.output_file}")
    logger.info("=" * 60)

def provide_performance_feedback(duration, use_async, args):
    if duration > 180:
        if not use_async:
            logger.info("💡 TIP: Try removing --sync_mode for better performance")
        elif args.disable_advanced_processing:
            logger.info("💡 TIP: Remove --disable_advanced_processing for better quality")
        else:
            logger.info("💡 TIP: Consider checking your internet connection")
    elif duration < 60:
        logger.info("🎉 Excellent performance!")

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)