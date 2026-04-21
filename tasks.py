import glob
import os
import shutil

from invoke import task


@task
def clean(c):
    """Remove build artifacts, docs, and cache files (Cross-platform)"""
    # 1. Directories to remove entirely
    directories = [
        "build",
        "dist",
        "source/build",  # If sphinx puts things here
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "htmlcov",
        "*.egg-info",
    ]

    # 2. Specific files to remove
    files = [".coverage", "coverage.xml"]

    print("Cleaning project...")

    # Handle Directories (including glob patterns like *.egg-info)
    for pattern in directories:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  - Removed directory: {path}")

    # Handle Files
    for pattern in files:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                os.remove(path)
                print(f"  - Removed file: {path}")

    # 3. Optional: Remove all __pycache__ folders recursively
    for root, dirs, _ in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                path = os.path.join(root, d)
                shutil.rmtree(path)
                print(f"  - Removed cache: {path}")

    print("Done!")


@task
def dev_install(c):
    """Install dev dependencies and pre-commit"""
    c.run('pip install -e ".[dev,sphinx]"')
    # c.run("pre-commit install")  # TODO: Figure out how to get pre-commit to work


@task
def docs(c):
    """Generate Sphinx documentation (Cross-platform)"""
    # This mimics 'export PYTHONPATH=$PYTHONPATH:.' but works on Windows/Linux
    current_env = os.environ.copy()
    sep = ";" if os.name == "nt" else ":"
    current_env["PYTHONPATH"] = f"{current_env.get('PYTHONPATH', '')}{sep}."

    # Run sphinx-build using the modified environment
    c.run("sphinx-build -M html source build", env=current_env)


@task
def format(c):
    """Run ruff for formatting"""
    c.run("ruff format .")


@task
def install(c):
    """Install production dependencies"""
    c.run("pip install .")


@task
def lint(c):
    """Run ruff for linting"""
    c.run("ruff check .")


@task(help={"all_files": "Run on all files (default: True)"})
def pre_commit(c, all_files=True):
    """Run pre-commit hooks"""
    cmd = "pre-commit run"
    if all_files:
        cmd += " --all-files"
    c.run(cmd)


@task
def test(c):
    """Run tests"""
    c.run("pytest")


@task
def type_check(c):
    """Run mypy"""
    c.run("mypy .")
