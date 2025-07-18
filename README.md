# H-DDU (Hierarchical Document Data Understanding)

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/aiteam0/ddu-lab/actions)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive document processing and analysis system that leverages advanced AI/ML techniques for hierarchical document understanding, parsing, and intelligent content extraction.

## 🚀 Features

### Core Capabilities
- **Multi-format Document Processing**: Support for PDF, images, and various document formats
- **Advanced OCR**: Integration with EasyOCR, Tesseract, and PaddleOCR for text extraction
- **Layout Detection**: YOLO-based document layout analysis and structure recognition
- **Table Recognition**: Intelligent table detection and extraction with RapidTable
- **Formula Recognition**: Mathematical formula detection and recognition using UniMERNet
- **Korean Language Support**: Specialized processing for Korean documents with KoNLPy and KiwiPie
- **Multi-modal AI Integration**: Support for OpenAI, Anthropic, and Google models

### Processing Pipeline
- **Document Parsing**: Multiple parsing engines (Docling, DocYOLO)
- **Element Extraction**: Text, images, tables, and formulas
- **Content Assembly**: Intelligent document reconstruction and merging
- **Markdown Generation**: Comprehensive markdown output with structured formatting
- **Translation Support**: Multi-language translation capabilities with DeepL
- **LangGraph Integration**: Complex document processing workflows

## 📋 Requirements

- **Python**: 3.12+ (Required)
- **Operating System**: Linux, Windows, macOS
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional but recommended for ML model inference

## 🛠️ Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aiteam0/ddu-lab.git
cd ddu-lab

# Install using pip
pip install -e .
```

### Using UV (Recommended)

```bash
# Install UV package manager
pip install uv

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/aiteam0/ddu-lab.git
cd ddu-lab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Download required models
python 00-download_models_hf.py
```

## 📚 Dependencies

### Core Libraries
- **Document Processing**: `PyMuPDF`, `pdfminer.six`, `pdf2image`, `pdfplumber`, `docling`
- **Machine Learning**: `torch`, `torchvision`, `transformers`, `ultralytics`, `scikit-learn`
- **OCR & Vision**: `easyocr`, `pytesseract`, `doclayout-yolo`, `rapid-table`
- **Language Processing**: `langchain`, `konlpy`, `kiwipiepy`, `fast-langdetect`, `nltk`
- **AI APIs**: `openai`, `anthropic`, `langchain-openai`, `langchain-anthropic`
- **Data Handling**: `pandas`, `numpy`, `pyarrow`, `fastparquet`

### Model Requirements
The system requires several pre-trained models:
- Layout detection models (YOLO-based)
- OCR models (PaddleOCR, multilingual)
- Table recognition models
- Formula recognition models (UniMERNet)
- Language detection models

## 🎯 Usage

### Basic Document Processing

```python
from hddu.complete_workflow import run_complete_workflow
from hddu.logging_config import init_project_logging

# Initialize logging
init_project_logging()

# Process a PDF document
result = run_complete_workflow(
    pdf_filepath="path/to/document.pdf",
    batch_size=1,
    test_page=None,
    verbose=True
)

print(f"Processing completed: {result}")
```

### Advanced Configuration

```python
from hddu.complete_workflow import create_complete_workflow
from hddu.state import ParseState

# Create custom workflow
workflow = create_complete_workflow(
    batch_size=2,
    test_page=1,  # Process only page 1
    verbose=True
)

# Initialize state
initial_state = ParseState(
    pdf_filepath="document.pdf",
    batch_size=2
)

# Run workflow
result = workflow.invoke(initial_state)
```

### Configuration Files

#### Magic PDF Configuration
```json
{
    "models-dir": "/path/to/models",
    "device-mode": "cpu",
    "layout-config": {
        "model": "doclayout_yolo"
    },
    "formula-config": {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": true
    },
    "table-config": {
        "model": "rapid_table",
        "enable": true
    }
}
```

## 🏗️ Architecture

### Module Structure

```
hddu/
├── __init__.py
├── complete_workflow.py     # Main workflow orchestration
├── parser.py               # Document parsing logic
├── preprocessing.py        # Data preprocessing utilities
├── extractor.py           # Element extraction modules
├── interpreter.py         # Content interpretation
├── translate.py           # Translation services
├── assembly/              # Document assembly pipeline
│   ├── main_assembler.py
│   ├── matcher.py
│   └── merger.py
└── prompts/               # AI prompt templates
    ├── IMAGE-SYSTEM-PROMPT.yaml
    └── TABLE-SYSTEM-PROMPT.yaml
```

### Key Components

#### 1. Document Parser (`hddu.parser`)
- PDF splitting and page processing
- Multi-engine parsing support
- State management and checkpointing

#### 2. Element Extractor (`hddu.extractor`)
- Page element detection
- Image entity extraction
- Table structure recognition

#### 3. Content Assembly (`hddu.assembly`)
- Document reconstruction
- Element matching and merging
- Post-processing optimization

#### 4. LangChain Integration (`langchain_utils`)
- Custom document loaders
- Korean language tokenizers
- AI model integrations

## 🔧 Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DEEPL_API_KEY="your-deepl-key"

# Model Configuration
export MODELS_DIR="/path/to/models"
export DEVICE_MODE="cpu"  # or "gpu"
```

### Magic PDF Configuration
Copy `magic-pdf.template.json` to `magic-pdf.json` and configure:

```json
{
    "models-dir": "/path/to/models",
    "device-mode": "cpu",
    "layout-config": {
        "model": "doclayout_yolo"
    }
}
```

## 📊 Output Structure

### Directory Layout
```
output/
├── docling_output/        # Docling processing results
├── docyolo_output/        # DocYOLO processing results
├── intermediate/          # Intermediate processing files
└── export/               # Final output files
    ├── document.md       # Markdown output
    ├── document.pkl      # Serialized state
    └── document.json     # JSON metadata
```

### Output Formats
- **Markdown**: Structured document with headers, tables, images
- **JSON**: Metadata and processing statistics
- **Parquet**: Structured data for analysis
- **PNG**: Extracted images and figures

## 🧪 Testing

```bash
# Run basic test
python test_complete_workflow.py

# Run with specific document
python test_complete_workflow.py --pdf "path/to/document.pdf"

# Debug mode
python test_complete_workflow.py --verbose --debug
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Download missing models
   python 00-download_models_hf.py
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   export BATCH_SIZE=1
   ```

3. **GPU Issues**
   ```bash
   # Force CPU mode
   export DEVICE_MODE="cpu"
   ```

### Logging

Logs are stored in the `logs/` directory:
- `app.log`: Application logs
- `error.log`: Error messages
- `debug.log`: Debug information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 coding standards
- Add unit tests for new features
- Update documentation for API changes
- Use type hints for better code clarity

## 📄 License

This project is licensed under the **AGPL-3.0 License** - see the [LICENSE](LICENSE) file for details.

### Important License Notes:
- This project uses **MinerU/Magic-PDF** which is licensed under AGPL-3.0
- Due to AGPL's copyleft nature, the entire project must comply with AGPL-3.0 terms
- **Commercial use requires compliance with AGPL-3.0** including source code disclosure
- For commercial licensing options, consider contacting the MinerU team or removing AGPL dependencies

### Key Dependencies and Their Licenses:
- **Docling**: MIT License ✅
- **MinerU/Magic-PDF**: AGPL-3.0 License ⚠️ (affects entire project)
- **LangChain**: MIT License ✅
- **Other ML libraries**: Various open-source licenses

## 🙏 Acknowledgments

- **Docling**: Document processing framework
- **LangChain**: AI application framework
- **PaddleOCR**: OCR capabilities
- **YOLO**: Object detection models
- **UniMERNet**: Formula recognition

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/aiteam0/ddu-lab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aiteam0/ddu-lab/discussions)
- **Documentation**: [Wiki](https://github.com/aiteam0/ddu-lab/wiki)

---

**H-DDU** - Transforming document processing with AI-powered hierarchical understanding.