---
name: Bug Report
about: Create a report to help us improve H-DDU
title: '[BUG] '
labels: ['bug']
assignees: ''

---

## 🐛 Bug Description

**Clear and concise description of the bug**
A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

**Steps to reproduce the behavior:**
1. Go to '...'
2. Run command '....'
3. Use document '....'
4. See error

## ✅ Expected Behavior

**What you expected to happen**
A clear and concise description of what you expected to happen.

## ❌ Actual Behavior

**What actually happened**
A clear and concise description of what actually happened instead.

## 🖼️ Screenshots/Logs

**If applicable, add screenshots or logs to help explain your problem**
```
Paste error logs here
```

## 🔧 Environment

**Please complete the following information:**
- **OS**: [e.g., Windows 11, Ubuntu 22.04, macOS 14.0]
- **Python Version**: [e.g., 3.12.1]
- **H-DDU Version**: [e.g., 0.1.0]
- **Installation Method**: [e.g., pip, uv, from source]

**Dependencies (if relevant):**
- PyMuPDF Version: [e.g., 1.24.9]
- Docling Version: [e.g., 2.15.1]
- PyTorch Version: [e.g., 2.2.2]

## 📄 Document Information

**Document details (if applicable):**
- **Document Type**: [e.g., PDF, image]
- **Document Size**: [e.g., 2.5MB, 10 pages]
- **Language**: [e.g., Korean, English, Mixed]
- **Document Source**: [e.g., scanned, digital, web]

**Can you share the document?**
- [ ] Yes, I can share the document
- [ ] No, the document is confidential
- [ ] I can share a similar document
- [ ] I can create a minimal example

## 🔍 Additional Context

**Processing Configuration:**
```json
{
  "batch_size": 1,
  "test_page": null,
  "verbose": true
}
```

**Error occurs in which component?**
- [ ] Document parsing (parser.py)
- [ ] OCR processing (extractor.py)
- [ ] Layout detection
- [ ] Text extraction
- [ ] Image processing
- [ ] Table recognition
- [ ] Assembly pipeline
- [ ] Export functionality
- [ ] Other: _________

**How often does this occur?**
- [ ] Always
- [ ] Sometimes
- [ ] Rarely
- [ ] Only with specific documents

## 🛠️ Debugging Information

**Command used:**
```bash
# Paste the exact command you ran
python test_complete_workflow.py --pdf "path/to/document.pdf"
```

**Configuration files:**
- [ ] Using custom configuration
- [ ] No configuration file

**System Resources:**
- **Available Memory**: [e.g., 16GB]
- **CPU**: [e.g., Intel i7, Apple M2]
- **GPU**: [e.g., NVIDIA GTX 1080, None]

## 📋 Logs

**Application logs (if available):**
```
Paste logs from logs/app.log here
```

**Error logs (if available):**
```
Paste logs from logs/error.log here
```

**Python traceback (if available):**
```
Paste full Python traceback here
```

## 🧪 Attempted Solutions

**What have you tried to fix this?**
- [ ] Reinstalled dependencies
- [ ] Updated to latest version
- [ ] Tried different document
- [ ] Changed configuration
- [ ] Checked documentation
- [ ] Searched existing issues

**Other solutions tried:**
<!-- Describe any other solutions you've attempted -->

## 🎯 Impact

**How does this bug affect you?**
- [ ] Blocks core functionality
- [ ] Affects document processing accuracy
- [ ] Causes performance issues
- [ ] Minor inconvenience
- [ ] Other: _________

**Priority Level:**
- [ ] High - Critical functionality broken
- [ ] Medium - Important feature affected
- [ ] Low - Minor issue

## 📚 Related Issues

**Are there any related issues?**
- Fixes #(issue number)
- Related to #(issue number)
- Duplicate of #(issue number)

## ✅ Checklist

**Before submitting this bug report:**
- [ ] I have searched existing issues
- [ ] I have read the documentation
- [ ] I have provided all requested information
- [ ] I have tested with the latest version
- [ ] I have included relevant logs/screenshots

---

**Additional Information:**
<!-- Add any other context about the problem here -->

**Would you like to contribute a fix?**
- [ ] Yes, I'd like to work on this
- [ ] No, I need help from maintainers
- [ ] Maybe, if I get guidance

Thank you for helping improve H-DDU! 🚀