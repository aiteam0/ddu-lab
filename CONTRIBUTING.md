# Contributing to H-DDU v2.1

🎉 **Thank you for your interest in contributing to H-DDU!** We're excited to have you as part of our AI research community!

## 🚀 Quick Start for Contributors

### **🔧 Development Setup**

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/ddu-lab.git
cd ddu-lab

# 2. Install UV package manager (recommended)
pip install uv

# 3. Setup development environment
uv sync --all-extras

# 4. Download required models
uv run python 00-download_models_hf.py

# 5. Verify installation
uv run pytest test_complete_workflow.py -v
```

## 📋 How to Contribute

### **🐛 Reporting Bugs**
1. Check existing [issues](https://github.com/aiteam0/ddu-lab/issues) first
2. Use our bug report template
3. Include environment details (OS, Python version, GPU/CPU)
4. Provide minimal reproduction steps
5. Attach relevant logs from `logs/` directory

### **💡 Feature Requests**
1. Open a [discussion](https://github.com/aiteam0/ddu-lab/discussions) first
2. Describe the use case and expected behavior
3. Consider AI/ML performance implications
4. Check alignment with our roadmap

### **🔧 Code Contributions**

#### **Workflow Overview**
```bash
# 1. Create feature branch
git checkout -b feature/your-amazing-feature

# 2. Make changes and test
uv run pytest
uv run black hddu/ test_*.py
uv run pylint hddu/

# 3. Commit with conventional format
git commit -m "feat: add amazing document processing feature"

# 4. Push and open PR
git push origin feature/your-amazing-feature
```

## 🎯 Development Guidelines

### **🐍 Code Standards**

#### **Python Style**
- **Python Version**: 3.12+ required
- **Formatter**: Black (line length: 88)
- **Import Sorting**: isort with Black profile
- **Linting**: Pylint for code quality
- **Type Hints**: Required for all public functions

#### **Example Code Structure**
```python
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from hddu.base import BaseProcessor
from hddu.config import Config

logger = logging.getLogger(__name__)


class DocumentProcessor(BaseProcessor):
    """Enhanced document processor for H-DDU v2.1.
    
    This processor implements advanced AI techniques for
    hybrid-deep document understanding.
    
    Args:
        config: Configuration object with processing parameters
        enable_gpu: Whether to use GPU acceleration
        
    Example:
        >>> processor = DocumentProcessor(config, enable_gpu=True)
        >>> result = processor.process("document.pdf")
    """
    
    def __init__(self, config: Config, enable_gpu: bool = False) -> None:
        super().__init__(config)
        self.enable_gpu = enable_gpu
        logger.info(f"Initialized processor with GPU: {enable_gpu}")
    
    def process(
        self, 
        filepath: Union[str, Path], 
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process document with enhanced AI capabilities."""
        # Implementation here
        pass
```

### **🧪 Testing Requirements**

#### **Test Categories**
- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing  
- **Performance Tests**: Speed and memory benchmarks
- **AI Model Tests**: Accuracy and consistency validation

#### **Writing Tests**
```python
import pytest
from unittest.mock import Mock, patch

from hddu.complete_workflow import run_complete_workflow
from hddu.config import Config


class TestCompleteWorkflow:
    """Test suite for complete workflow functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock(spec=Config)
        config.batch_size = 1
        config.enable_gpu = False
        return config
    
    def test_workflow_basic_processing(self, mock_config):
        """Test basic document processing workflow."""
        with patch('hddu.parser.PDFParser') as mock_parser:
            mock_parser.return_value.parse.return_value = {"pages": 1}
            
            result = run_complete_workflow(
                pdf_filepath="test.pdf",
                config=mock_config
            )
            
            assert result["status"] == "success"
            assert "pages" in result
    
    @pytest.mark.slow
    def test_workflow_performance(self, mock_config):
        """Test workflow performance benchmarks."""
        import time
        
        start_time = time.time()
        result = run_complete_workflow("large_document.pdf", mock_config)
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert processing_time < 300  # 5 minutes max
        assert result["memory_usage"] < 1000  # MB
```

#### **Running Tests**
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=hddu --cov-report=html

# Run specific test categories
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m "integration"  # Only integration tests

# Run performance benchmarks
uv run pytest test_complete_workflow.py --benchmark
```

## 🔄 CI/CD Integration

### **Automated Checks**
Our CI/CD pipeline automatically runs:

#### **🧪 Testing Pipeline**
- **Multi-Python Testing**: 3.10, 3.11, 3.12
- **Cross-Platform**: Ubuntu, Windows, macOS
- **GPU/CPU Testing**: Both acceleration modes
- **Performance Benchmarks**: Speed and memory validation

#### **🔍 Code Quality**
- **Black Formatting**: Automatic style checking
- **Import Sorting**: isort validation
- **Linting**: Pylint code quality analysis
- **Type Checking**: mypy static analysis
- **Security Scanning**: Bandit vulnerability detection
- **Dependency Check**: pip-audit security validation

#### **📊 Coverage Requirements**
- **Minimum Coverage**: 80% overall
- **Critical Components**: 90%+ coverage
- **New Code**: 85%+ coverage required

### **PR Auto-Labeling**
Our smart labeling system automatically categorizes PRs:
- **Size Labels**: XS/S/M/L/XL based on changes
- **Component Labels**: core/ai/assembly/docs/tests
- **Type Labels**: feature/bugfix/docs/maintenance
- **Priority Labels**: Based on title keywords

## 📝 Commit Message Format

### **Conventional Commits**
We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### **Types**
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks
- **ci**: CI/CD changes

#### **Scopes**
- **core**: Core workflow components
- **ai**: AI/ML processing components
- **assembly**: Document assembly pipeline
- **config**: Configuration management
- **tests**: Testing framework
- **docs**: Documentation

#### **Examples**
```bash
# Good commit messages
git commit -m "feat(ai): add enhanced interpreter with 30% speed boost"
git commit -m "fix(core): resolve memory leak in batch processing"
git commit -m "docs: update README with v2.1 features"
git commit -m "test(assembly): add comprehensive merger unit tests"

# Bad commit messages
git commit -m "fix stuff"  # Too vague
git commit -m "WIP"  # Work in progress
git commit -m "Update file.py"  # Not descriptive
```

## 🏗️ Architecture Guidelines

### **AI/ML Components**

#### **Model Integration**
- **Model Loading**: Use lazy loading for better performance
- **GPU Management**: Proper CUDA memory handling
- **Error Handling**: Graceful fallback to CPU
- **Caching**: Implement model result caching

#### **Rate Limiting**
- **API Calls**: Always use rate limiting for external APIs
- **Retry Logic**: Implement exponential backoff
- **Monitoring**: Add performance metrics

### **Code Organization**

#### **Module Structure**
```
hddu/
├── core/              # Core functionality
│   ├── workflow.py    # Main workflow logic
│   └── base.py        # Base classes
├── ai/                # AI/ML components
│   ├── interpreter.py # AI interpretation
│   ├── translator.py  # Translation services
│   └── models/        # Model management
├── processing/        # Document processing
│   ├── parser.py      # PDF parsing
│   ├── extractor.py   # Element extraction
│   └── assembly/      # Document assembly
├── utils/             # Utility functions
│   ├── config.py      # Configuration
│   ├── logging.py     # Logging setup
│   └── rate_limit.py  # Rate limiting
└── tests/             # Test suite
    ├── unit/          # Unit tests
    ├── integration/   # Integration tests
    └── performance/   # Performance tests
```

## 🚀 Performance Considerations

### **Optimization Guidelines**

#### **Memory Management**
- Use generators for large datasets
- Implement proper cleanup in `__del__` methods
- Monitor memory usage in long-running processes
- Use memory-efficient data structures

#### **Processing Speed**
- Batch operations where possible
- Use multiprocessing for CPU-intensive tasks
- Implement caching for repeated operations
- Profile code regularly with `cProfile`

#### **AI Model Performance**
- Use appropriate batch sizes
- Implement model quantization where applicable
- Cache model predictions
- Monitor GPU utilization

## 📚 Documentation Standards

### **Code Documentation**
- **Docstrings**: Google style for all public functions
- **Type Hints**: Complete type annotations
- **Examples**: Include usage examples in docstrings
- **API Documentation**: Auto-generated from docstrings

### **User Documentation**
- **README**: Keep updated with new features
- **Tutorials**: Step-by-step guides for common use cases
- **API Reference**: Complete function documentation
- **Migration Guides**: For breaking changes

## 🔒 Security Guidelines

### **Security Best Practices**
- **No Hardcoded Secrets**: Use environment variables
- **Input Validation**: Sanitize all user inputs
- **Dependency Management**: Regular security updates
- **Error Messages**: Don't expose sensitive information

### **AI Model Security**
- **Model Validation**: Verify model integrity
- **Input Sanitization**: Clean model inputs
- **Output Filtering**: Validate model outputs
- **Privacy**: Handle sensitive documents appropriately

## 🎯 Release Process

### **Version Numbering**
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### **Release Checklist**
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Performance benchmarks validated
- [ ] Security scan completed
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Git tags created

## 🤝 Community Guidelines

### **Code of Conduct**
- Be respectful and inclusive
- Constructive feedback only
- Help newcomers learn
- Focus on the technology, not personal preferences

### **Communication Channels**
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code review and collaboration

## 🏆 Recognition

### **Contributor Rewards**
- **Contributors File**: Automatic addition to CONTRIBUTORS.md
- **Monthly Highlights**: Featured contributors in releases
- **Conference Opportunities**: Speaking opportunities at AI events
- **Research Collaboration**: Co-authorship on research papers

### **Contribution Levels**
- **First-Time Contributors**: Welcome package and mentoring
- **Regular Contributors**: Advanced access and early feature testing
- **Core Contributors**: Maintainer privileges and decision-making input

## 📞 Getting Help

### **Support Channels**
- **Documentation**: Check the [Wiki](https://github.com/aiteam0/ddu-lab/wiki) first
- **Issues**: Create a [GitHub Issue](https://github.com/aiteam0/ddu-lab/issues)
- **Discussions**: Join [GitHub Discussions](https://github.com/aiteam0/ddu-lab/discussions)

### **Development Questions**
- **Setup Issues**: Check the development setup guide
- **Testing Problems**: Review the testing documentation
- **Performance Issues**: Use profiling tools and benchmarks

---

**🚀 Thank you for contributing to H-DDU!** Together, we're building the future of document intelligence with AI.

*Your contributions help advance the state of the art in document understanding technology.*