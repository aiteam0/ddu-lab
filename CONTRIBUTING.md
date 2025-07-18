# Contributing to H-DDU (Hybrid-Deep Document Understanding)

First off, thank you for considering contributing to H-DDU! 🎉 Your contributions help make document processing more accessible and powerful for everyone.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## 🤝 Code of Conduct

This project adheres to a code of conduct that we expect all participants to follow:

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and encourage diverse perspectives
- **Be collaborative**: Help others and ask for help when needed
- **Be patient**: Remember that everyone has different experience levels

## 🚀 How Can I Contribute?

### 🐛 Reporting Bugs
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include system information (OS, Python version, dependencies)
- Provide steps to reproduce the issue
- Include relevant log files and error messages

### 💡 Suggesting Features
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the problem your feature would solve
- Provide use cases and examples
- Consider the impact on existing functionality

### 🔧 Code Contributions
- Fix bugs or implement new features
- Improve documentation
- Add tests for existing functionality
- Optimize performance

### 📚 Documentation
- Improve README and API documentation
- Add examples and tutorials
- Fix typos and improve clarity
- Translate documentation

## 🛠️ Development Setup

### Prerequisites
- **Python 3.12+** (Required)
- **Git** for version control
- **UV** or **pip** for dependency management

### Setup Instructions

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/ddu-lab.git
   cd ddu-lab
   ```

2. **Create Virtual Environment**
   ```bash
   # Using UV (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Or using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Using UV
   uv sync
   
   # Or using pip
   pip install -e .
   ```

4. **Download Models**
   ```bash
   python 00-download_models_hf.py
   ```

5. **Setup Configuration**
   ```bash
   cp magic-pdf.template.json magic-pdf.json
   # Edit magic-pdf.json with your settings
   ```

6. **Run Tests**
   ```bash
   python test_complete_workflow.py
   ```

## 📝 Contribution Guidelines

### 🎯 Focus Areas

#### Core Components
- **Document Parsing**: Improve PDF/document processing
- **OCR Integration**: Enhance text extraction accuracy
- **Layout Detection**: Better structure recognition
- **Korean Language**: Specialized Korean text processing
- **AI Integration**: LLM model improvements

#### Technical Improvements
- **Performance**: Optimize processing speed
- **Memory Usage**: Reduce memory footprint
- **Error Handling**: Better error messages and recovery
- **Logging**: Improve debugging capabilities

### 🗂️ Project Structure
```
hddu/
├── assembly/          # Document assembly pipeline
├── prompts/           # AI prompt templates
├── complete_workflow.py  # Main workflow
├── parser.py          # Document parsing
├── extractor.py       # Element extraction
├── preprocessing.py   # Data preprocessing
└── utils.py          # Utility functions
```

## 🔄 Pull Request Process

### Before Submitting
1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test Locally**
   ```bash
   python test_complete_workflow.py
   # Test with different document types
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Pull Request Template
- Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md)
- Provide clear description of changes
- Include test results and screenshots
- Reference related issues

### Review Process
1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code quality
3. **Testing**: Functionality and performance testing
4. **Documentation**: Check for updated docs
5. **Merge**: Approved PRs are merged

## 🐛 Issue Guidelines

### Bug Reports
- **Clear Title**: Describe the issue concisely
- **Environment**: Include OS, Python version, dependencies
- **Steps to Reproduce**: Detailed reproduction steps
- **Expected vs Actual**: What should happen vs what happens
- **Logs**: Include relevant error messages and logs

### Feature Requests
- **Problem Statement**: What problem does this solve?
- **Proposed Solution**: How would you implement it?
- **Alternatives**: What other approaches did you consider?
- **Impact**: How would this benefit users?

## 📐 Coding Standards

### Python Style
- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use type annotations
- **Docstrings**: Document functions and classes
- **Error Handling**: Proper exception handling

### Code Quality
```python
# Good example
def process_document(
    pdf_path: str, 
    config: Dict[str, Any]
) -> ProcessResult:
    """Process a PDF document with given configuration.
    
    Args:
        pdf_path: Path to the PDF file
        config: Processing configuration
        
    Returns:
        ProcessResult: Processing results
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ProcessingError: If processing fails
    """
    try:
        # Implementation
        pass
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise ProcessingError(f"Failed to process {pdf_path}") from e
```

### Commit Messages
Follow conventional commits format:
```
feat: add new OCR engine integration
fix: resolve memory leak in document processing
docs: update API documentation
test: add unit tests for parser module
refactor: improve code structure in extractor
```

## 🧪 Testing

### Test Types
1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test processing speed and memory

### Running Tests
```bash
# Run basic workflow test
python test_complete_workflow.py

# Test with specific document
python test_complete_workflow.py --pdf "path/to/document.pdf"

# Run with verbose output
python test_complete_workflow.py --verbose
```

### Test Documents
- Include various document types (PDF, images)
- Test with different languages (Korean, English)
- Include edge cases (corrupted files, empty documents)

## 📖 Documentation

### Documentation Standards
- **Clear Examples**: Provide working code examples
- **API Documentation**: Document all public functions
- **Tutorials**: Step-by-step guides for common tasks
- **Troubleshooting**: Common issues and solutions

### Building Documentation
```bash
# Update README with new features
# Add docstrings to new functions
# Create examples in examples/ directory
```

## 🏷️ Labels and Milestones

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `korean-language`: Korean text processing
- `performance`: Performance related
- `ai-integration`: AI/ML model integration

### Priority Labels
- `priority-high`: Critical issues
- `priority-medium`: Important issues
- `priority-low`: Nice to have

## 🎖️ Recognition

Contributors are recognized in:
- README.md acknowledgments
- Release notes
- GitHub contributors page

## 📞 Getting Help

### Communication Channels
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussions
- **Email**: For private matters

### Resources
- **Documentation**: Project README and wiki
- **Examples**: Sample code and tutorials
- **FAQ**: Common questions and answers

## 🔄 Release Process

### Versioning
- Follow semantic versioning (SemVer)
- Major.Minor.Patch (e.g., 1.0.0)

### Release Cycle
1. **Development**: Feature development and bug fixes
2. **Testing**: Comprehensive testing phase
3. **Documentation**: Update documentation
4. **Release**: Tag and release new version

---

## 🙏 Thank You

Thank you for contributing to H-DDU! Your efforts help advance document processing technology and make it more accessible to everyone. Every contribution, no matter how small, makes a difference.

For questions about contributing, please open an issue or start a discussion. We're here to help! 🚀