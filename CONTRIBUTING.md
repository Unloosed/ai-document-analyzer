# Contributing to ai-document-analyzer

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to this project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to see if the problem has already been reported. When reporting a bug, please include:

- A clear and concise description of what the bug is.
- Steps to reproduce the behavior.
- Expected behavior and what actually happened.
- Environment details (OS, Python version, etc.).

### Suggesting Enhancements

If you have an idea for a feature or an enhancement, feel free to open an issue. Provide as much detail as possible about the proposed change and why it would be beneficial.

### Your First Code Contribution

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ai-document-analyzer.git`.
3. Create a branch for your work: `git checkout -b my-new-feature`.
4. Install the development environment: `make dev-install`.
5. Make your changes.
6. Run tests and linting: `make test`, `make lint`, `make type-check`.
7. Commit your changes: `git commit -m 'Add some feature'`. (Ensure `pre-commit` hooks pass!)
8. Push to the branch: `git push origin my-new-feature`.
9. Submit a pull request.

## Styleguides

### Python Styleguide

- We use `ruff` for linting and formatting.
- We use `mypy` for static type checking.
- Please ensure your code follows these standards before submitting a pull request.

### Documentation Styleguide

- We use `Sphinx` for documentation.
- Docstrings should follow the Google style.

## Questions?

Feel free to open an issue if you have any questions!
