# Project Setup

Here we provide some details about the project setup. Most of the choices are explained in the
[guide](https://guide.esciencecenter.nl). Links to the relevant sections are included below. Feel free to remove this
text when the development of the software package takes off.

For a quick reference on software development, we refer to [the software guide
checklist](https://guide.esciencecenter.nl/#/best_practices/checklist).

## Python versions

This repository is set up with Python versions:

- 3.10
- 3.11
- 3.12

Add or remove Python versions based on project requirements. See [the
guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python) for more information about Python
versions.

## Package management and dependencies

You can use either pip or conda for installing dependencies and package management. This repository does not force you
to use one or the other, as project requirements differ. For advice on what to use, please check [the relevant section
of the
guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=dependencies-and-package-management).

- Runtime dependencies should be added to `pyproject.toml` in the `dependencies` list under `[project]`.
- Development dependencies, such as for testing or documentation, should be added to `pyproject.toml` in one of the lists under `[project.optional-dependencies]`.

## Packaging/One command install

You can distribute your code using PyPI.
[The guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=building-and-packaging-code) can
help you decide which tool to use for packaging.

## Testing and code coverage

- Tests should be put in the `tests` folder.
- The `tests` folder contains:
  - Example tests that you should replace with your own meaningful tests (file: `test_my_module.py`)
- The testing framework used is [PyTest](https://pytest.org)
  - [PyTest introduction](https://pythontest.com/pytest-book/)
  - PyTest is listed as a development dependency
  - This is configured in `pyproject.toml`
- The project uses [GitHub action workflows](https://docs.github.com/en/actions) to automatically run tests on GitHub infrastructure against multiple Python versions
  - Workflows can be found in [`.github/workflows`](.github/workflows/)
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=testing)

## Documentation

- Documentation should be put in the [`docs/`](docs/) directory. The contents have been generated using `sphinx-quickstart` (Sphinx version 1.6.5).
- We recommend writing the documentation using Restructured Text (reST) and Google style docstrings.
  - [Restructured Text (reST) and Sphinx CheatSheet](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html)
  - [Google style docstring examples](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- The documentation is set up with the ReadTheDocs Sphinx theme.
  - Check out its [configuration options](https://sphinx-rtd-theme.readthedocs.io/en/latest/).
- [AutoAPI](https://sphinx-autoapi.readthedocs.io/) is used to generate documentation for the package Python objects.
- `.readthedocs.yaml` is the ReadTheDocs configuration file. When ReadTheDocs is building the documentation this package and its development dependencies are installed so the API reference can be rendered.- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=writingdocumentation)

## Coding style conventions and code quality

- [Relevant section in the NLeSC guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=coding-style-conventions) and [README.dev.md](README.dev.md).

## Continuous code quality

[Sonarcloud](https://sonarcloud.io/) is used to perform quality analysis and code coverage report

- `sonar-project.properties` is the SonarCloud [configuration](https://docs.sonarqube.org/latest/analysis/analysis-parameters/) file
- `.github/workflows/sonarcloud.yml` is the GitHub action workflow which performs the SonarCloud analysis

## Package version number

- We recommend using [semantic versioning](https://guide.esciencecenter.nl/#/best_practices/releases?id=semantic-versioning).
- For convenience, the package version is stored in a single place: `pyproject.toml` under the `tool.bumpversion` header.
- Don't forget to update the version number before [making a release](https://guide.esciencecenter.nl/#/best_practices/releases)!

## Logging

- We recommend using the logging module for getting useful information from your module (instead of using print).
- The project is set up with a logging example.
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=logging)

## CHANGELOG.md

- Document changes to your software package
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/releases?id=changelogmd)

## CITATION.cff

- To allow others to cite your software, add a `CITATION.cff` file
- It only makes sense to do this once there is something to cite (e.g., a software release with a DOI).
- Follow the [making software citable](https://guide.esciencecenter.nl/#/citable_software/making_software_citable) section in the guide.

## CODE_OF_CONDUCT.md

- Information about how to behave professionally
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/documentation?id=code-of-conduct)

## CONTRIBUTING.md

- Information about how to contribute to this software package
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/documentation?id=contribution-guidelines)

## MANIFEST.in

- List non-Python files that should be included in a source distribution
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/language_guides/python?id=building-and-packaging-code)

## NOTICE

- List of attributions of this project and Apache-license dependencies
- [Relevant section in the guide](https://guide.esciencecenter.nl/#/best_practices/licensing?id=notice)

# `paidiverpy` developer documentation

If you're looking for user documentation, go [here](README.md).

## Development install

1. Clone the repository:

   ```bash
   # ssh
   git clone git@github.com:paidiver/paidiverpy.git

   # https
   # git clone https://github.com/paidiver/paidiverpy.git

   cd paidiverpy
   ```

2. Create env and install package

  - Using `conda` (recommended):
    ```bash
    conda init

    # Command to restart the terminal. This command may not be necessary if conda init has already been successfully run before
    exec bash

    conda env create -f environment.yml
    conda activate Paidiverpy

    # (from the project root directory)
    # install paidiverpy as an editable package
    pip install --no-cache-dir --editable .
    # install development dependencies
    pip install --no-cache-dir --editable .[dev]
    # install documentation dependencies only
    pip install --no-cache-dir --editable .[docs]
    ```

  - Using `venv`:
    ```bash
    # Create a virtual environment, e.g. with
    python -m venv env

    # activate virtual environment
    source env/bin/activate

    # make sure to have a recent version of pip and setuptools
    python -m pip install --upgrade pip setuptools

    # (from the project root directory)
    # install paidiverpy as an editable package
    python -m pip install --no-cache-dir --editable .
    # install development dependencies
    python -m pip install --no-cache-dir --editable .[dev]
    # install documentation dependencies only
    python -m pip install --no-cache-dir --editable .[docs]
    ```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

```bash
pytest -v
```

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
In an activated virtual environment with the development tools installed, inside the package directory, run:

```shell
coverage run
```

This runs tests and stores the result in a `.coverage` file.
To see the results on the command line, run

```shell
coverage report
```

`coverage` can also generate output in HTML and other formats; see `coverage help` for more information.## Running linters locally

## Lint

For linting and sorting imports we will use [ruff](https://beta.ruff.rs/docs/). Running the linters requires an
activated virtual environment with the development tools installed.

```shell
# linter
ruff check .

# linter with automatic fixing
ruff check . --fix
```

To fix readability of your code style you can use [yapf](https://github.com/google/yapf).You can enable automatic linting with `ruff` on commit by enabling the git hook from `.githooks/pre-commit`, like so:

```shell
git config --local core.hooksPath .githooks
```

## Generating the API docs

```shell
cd docs
make html
```

The documentation will be in `docs/_build/html`

If you do not have `make` use

```shell
sphinx-build -b html docs docs/_build/html
```

To find undocumented Python objects run

```shell
cd docs
make coverage
cat _build/coverage/python.txt
```

<!-- To [test snippets](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html) in documentation run

```shell
cd docs
make doctest
``` -->

## Versioning

Bumping the version across all files is done with [bump-my-version](https://github.com/callowayproject/bump-my-version), e.g.

```shell
bump-my-version bump major  # bumps from e.g. 0.3.2 to 1.0.0
bump-my-version bump minor  # bumps from e.g. 0.3.2 to 0.4.0
bump-my-version bump patch  # bumps from e.g. 0.3.2 to 0.3.3
```

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page).
1. Verify that the information in [`CITATION.cff`](CITATION.cff) is correct.
1. Make sure the [version has been updated](#versioning).
1. Run the unit tests with `pytest -v`

### (2/3) PyPI

In a new terminal:

```shell
# OPTIONAL: prepare a new directory with fresh git clone to ensure the release
# has the state of origin/main branch
cd $(mktemp -d paidiverpy.XXXXXX)
git clone git@github.com:paidiver/paidiverpy .

# make sure to have a recent version of pip and the publishing dependencies
python -m pip install --upgrade pip
python -m pip install .[publishing]

# create the source distribution and the wheel
python -m build

# upload to test pypi instance (requires credentials)
python -m twine upload --repository testpypi dist/*
```

Visit
[https://test.pypi.org/project/paidiverpy](https://test.pypi.org/project/paidiverpy)
and verify that your package was uploaded successfully. Keep the terminal open, we'll need it later.

In a new terminal, without an activated virtual environment or an env directory:

```shell
cd $(mktemp -d paidiverpy-test.XXXXXX)

# prepare a clean virtual environment and activate it
python -m venv env
source env/bin/activate

# make sure to have a recent version of pip and setuptools
python -m pip install --upgrade pip

# install from test pypi instance:
python -m pip -v install --no-cache-dir \
--index-url https://test.pypi.org/simple/ \
--extra-index-url https://pypi.org/simple paidiverpy
```

Check that the package works as it should when installed from pypitest.

Then upload to pypi.org with:

```shell
# Back to the first terminal,
# FINAL STEP: upload to PyPI (requires credentials)
python -m twine upload dist/*
```

### (3/3) GitHub

Don't forget to also make a [release on GitHub](https://github.com/paidiver/paidiverpy/releases/new).GitHub-Zenodo integration will also trigger Zenodo into making a snapshot of your repository and sticking a DOI on it.
