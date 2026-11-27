# Developer Guide – `paidiverpy`

This document contains guidelines and instructions for contributors and maintainers of **`paidiverpy`**.
For user-facing documentation, see [README.rst](README.rst).

---

## Supported Python Versions

* 3.10
* 3.11
* 3.12

---

## Installation for Development

### 1. Clone the repository

```bash
# SSH
git clone git@github.com:paidiver/paidiverpy.git

# HTTPS
# git clone https://github.com/paidiver/paidiverpy.git

cd paidiverpy
```

### 2. Create environment and install package

#### Option A: Conda

```bash
conda env create -f environment.yml
conda activate Paidiverpy

# Install the package and the tools needed for development and tests.
python -m pip install -e ".[dev,test,gui]"
```

Conda provides Python and pip. Pip installs Paidiverpy and its dependencies
from `pyproject.toml`. Include `docs` or `notebooks` in the same install
command when you need Sphinx or a Jupyter kernel.

#### Option B: venv

```bash
python -m venv env
source env/bin/activate

python -m pip install -e ".[dev,test,gui]"
```
---

## Testing

Run tests with:

```bash
pytest -v
```

### Coverage

```bash
coverage run
coverage report
```

HTML and other output formats are available. See `coverage help`.

---

## Linting & Code Style

This project uses [ruff](https://beta.ruff.rs/docs/) for linting and [yapf](https://github.com/google/yapf) for formatting.

```bash
# lint check
ruff check .

# lint with auto-fix
ruff check . --fix
```

Enable git pre-commit hook for automatic linting:

```bash
git config --local core.hooksPath .githooks
```

---

## Documentation

* Documentation lives in [`docs/`](docs/).
* Generated with **Sphinx** and the **ReadTheDocs theme**.
* API docs are generated with [AutoAPI](https://sphinx-autoapi.readthedocs.io/).

Build locally:

```bash
cd docs
make html
```

Or, without `make`:

```bash
sphinx-build -b html docs docs/_build/html
```

Check for undocumented objects:

```bash
cd docs
make coverage
cat _build/coverage/python.txt
```

---

## Versioning

We use [semantic versioning](https://guide.esciencecenter.nl/#/best_practices/releases?id=semantic-versioning).
Version is managed in `pyproject.toml` with [bump-my-version](https://github.com/callowayproject/bump-my-version).

Examples:

```bash
bump-my-version bump major  # 0.3.2 → 1.0.0
bump-my-version bump minor  # 0.3.2 → 0.4.0
bump-my-version bump patch  # 0.3.2 → 0.3.3
```

---

## Release Process

Releases consist of three parts:

### 1. Preparation

* Update [CHANGELOG.md](CHANGELOG.md).
* Verify [`CITATION.cff`](CITATION.cff).
* Bump version.
* Run tests:

  ```bash
  pytest -v
  ```

### 2. Publish to PyPI


The publish to pypi is handled via GitHub Actions. See [.github/workflows/release.yml](.github/workflows/release.yml) for details.


### 3. GitHub Release

Create a [new release](https://github.com/paidiver/paidiverpy/releases/new) on GitHub.
This also triggers Zenodo to mint a DOI snapshot.

---

## Additional Development Notes

* **Logging**: use the `logging` module (not `print`).
* **CI**: tests run via GitHub Actions across all supported Python versions.
* **Policies**: see [CODE\_OF\_CONDUCT.md](CODE_OF_CONDUCT.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
