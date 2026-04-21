# Documentation Setup Guide

This README provides instructions for setting up and building the documentation for the FSI-Data-Shelter project using Sphinx.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuring Sphinx](#configuring-sphinx)
- [Documenting Your Code](#documenting-your-code)
- [Adding Documentation Files (.rst)](#adding-documentation-files-rst)
- [Building the Documentation](#building-the-documentation)
- [Iterating and Updating](#iterating-and-updating)

## Prerequisites

- **Python:** Ensure you have Python 3 installed on your system.
- **pip:** Python package installer (should come with Python).
- **pandoc:** A universal document converter used by `nbsphinx` ([download the `.msi`](https://pandoc.org/installing.html))
Note: Using `conda-forge` may work as an avenue to install `pandoc`:

    ```bash
    conda install -c conda-forge pandoc
    ```

## Installation

1. **Navigate to the project's root directory** in your terminal or command prompt.

2. **Install Sphinx and the necessary themes/extensions:**

    ```bash
    pip install sphinx pydata-sphinx-theme myst-parser sphinx-copybutton nbsphinx
    ```

## Configuring Sphinx

The main configuration file for Sphinx is `source/conf.py`. Here are some key settings to be aware of:

- **`html_theme`:** This should be set to `'pydata_sphinx_theme'` to use the recommended theme.
- **`extensions`:** Ensure the following extensions are included:

    ```python
    extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.doctest',
    'sphinx_copybutton',
    'myst_parser',
    'nbsphinx',
    'pydata_sphinx_theme',

]
    ```
**NOTE: `sphinx.ext.autodoc` will import modules in order to document them, meaning code not guarded by**
    ```python
    if **name** == '**main**':
    ```
    **will be executed!**

## Documenting Your Code

Sphinx primarily uses **docstrings** within your Python code to generate documentation. Follow these guidelines:

- **Use a consistent docstring style:** NumPy or Google style are widely used and well-supported by the `sphinx.ext.napoleon` extension.
- **Document all modules, classes, functions, and methods:** Explain their purpose, arguments, return values, and any exceptions they might raise.
- **Include examples** in your docstrings to illustrate how to use the code. These examples can also be run as tests if you have the `sphinx.ext.doctest` extension enabled.

**Example Python Function with Docstring:**

```python
def example_function(input_value: int) -> str:
    """A brief summary of the function.

    A more detailed explanation of what the function does.

    .. todo:: Refactor this function for better performance.

    Args:
        input_value (int): Description of the input value.

    Returns:
        str: Description of the return value.

    Example:
        >>> example_function(5)
        'Result based on 5'
    """
    return f"Result based on {input_value}"
```

**Note:** The `.. todo::` directive is Sphinx-specific, as `napoleon` does not recognize any `TODO:` headers within Google-style docstrings

For more details, please see the [Google](https://google.github.io/styleguide/pyguide.html#docstrings>) documentation for more information.

## Adding Documentation Files (.rst)

Sphinx documentation is written in **reStructuredText (.rst)** format. Here's how to create and add these files:

### For API documentation (documenting Python code):

We use an automated script to generate `.rst` files for Python modules. This script is **automatically executed** during the Sphinx build process.

1. **Ensure the project root is in your PYTHONPATH** (if running Sphinx manually):

    ```bash
    export PYTHONPATH=$PYTHONPATH:.
    ```

    The build process scans the project and automatically places `.rst` files in the appropriate subdirectories within `source/` (e.g., `source/utils/`, `source/reference/`).

2. **Verify the generated files** in the `source/` directory after the build.

### For manual documentation:

1. **Create new `.rst` files** using a plain text editor (e.g., `concepts/my_concept.rst`, `tutorials/my_tutorial.rst`).
2. **Use reStructuredText syntax** to structure your documentation. Refer to the [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for details.

### Linking Files:

**Link your `.rst` files in the `toctree` directive** within your main `source/index.rst` file. This creates the navigation structure of your documentation.

```rst
.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   :glob:

   api/*
   utils/*
   reference/*
```

`:glob:` allows us to use wildcard characters

*NOTE:* Sphinx is compatible with `markdown` files as well. If you have static documentation that does not depend on code, you may put it into a `.md` file and include it in the appropriate documentation directories.

## Building the Documentation

1. **Navigate to the project's root directory** in your terminal.

2. **Run the Sphinx build command:**

    ```bash
    sphinx-build -M html source build
    ```

3. **The generated HTML documentation will be located in the `build/html` subdirectory.** Open the `build/html/index.html` file in your web browser to view it.

## Iterating and Updating

After making changes to your docstrings, you can simply rebuild the documentation. The `utils/generate_rst.py` script will be triggered automatically:

```bash
export PYTHONPATH=$PYTHONPATH:.
rm -rf ./build
sphinx-build -M html source build
```
