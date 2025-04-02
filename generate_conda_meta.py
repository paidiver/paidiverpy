"""Generate the meta.yaml file for the conda recipe."""

import sys
from pathlib import Path
import toml
from jinja2 import Template

TEMPLATE_STR = """
package:
  name: {% raw %}{{ name|lower }}{% endraw %}

  version: {% raw %}{{ version }}{% endraw %}

source:
  url: https://pypi.org/packages/source/{% raw %}{{ name[0] }}{% endraw %}/{% raw %}{{ name }}{% endraw %}/{% raw %}{{ name }}{% endraw %}-{% raw %}{{ version }}{% endraw %}.tar.gz
  sha256: {% raw %}{{ sha256 }}{% endraw %}


build:
  entry_points:
    - {{ name|lower }} = cli.main:main
  noarch: python
  script: {% raw %}{{ PYTHON }}{% endraw %} -m pip install . -vv --no-deps --no-build-isolation
  number: 0
  {% if bioconda %}
  run_exports:
    - {% raw %}{{ pin_subpackage('paidiverpy', max_pin="x.x") }}{% endraw %}

  {% endif %}

requirements:
  host:
    - python {% raw %}{{ python_min }}{% endraw %}

    - setuptools >=64.0.0
    - setuptools-scm
    - wheel
    - pip
  run:
    - python >={% raw %}{{ python_min }}{% endraw %}

{% for item in dependencies %}
    - {{ item }}
{% endfor %}

test:
  imports:
    - cli
    - {{ name|lower }}
  commands:
    - pip check
    - {{ name|lower }} --help
  requires:
    - python {% raw %}{{ python_min }}{% endraw %}

    - pip

about:
  summary: {{ description }}
  home: https://github.com/paidiver/paidiverpy
  license: Apache-2.0
  license_file: {{ license_file }}

extra:
  recipe-maintainers:
    - soutobias

"""  # noqa: E501


def load_toml() -> dict:
    """Load the pyproject.toml file.

    Returns:
        dict: The pyproject.toml data.
    """
    repo_root = Path.resolve(Path(__file__).parent)
    toml_path = repo_root / "pyproject.toml"

    with toml_path.open() as file:
        return toml.load(file)


def create_meta_yaml(pyproject_data: dict, bioconda: bool = False) -> str:
    """Create the meta.yaml file content.

    Args:
        pyproject_data (dict): The pyproject.toml data.
        bioconda (bool): Flag to use bioconda dependencies

    Returns:
        str: The content of the meta.yaml file.
    """
    project = pyproject_data["project"]
    python_min = project["requires-python"].replace(">", "").replace("<", "").replace("=", "")
    version = project["version"]
    name = project["name"]
    description = project["description"]
    license_file = project["license"]["file"]

    dependencies = project["dependencies"]
    dependencies = [
        " <".join(item.split("<"))
        if "<" in item
        else " >".join(item.split(">"))
        if ">" in item
        else " ==".join(item.split("=="))
        if "==" in item
        else item
        for item in dependencies
    ]

    template = Template(TEMPLATE_STR, trim_blocks=True, lstrip_blocks=True)

    # template_without_header = template.render(
    #     description=description, license_file=license_file, name=name, bioconda=bioconda
    # )

    template_without_header = template.render(
        description=description, license_file=license_file, dependencies=dependencies, name=name, bioconda=bioconda
    )

    header_str = "{% set python_min = {{ python_min }} %}\n{% set version = {{ version }} %}\n{% set name = {{ name }} %}\n"
    header_str = header_str.replace("{{ python_min }}", f'"{python_min}"').strip()
    header_str = header_str.replace("{{ version }}", f'"{version}"').strip()
    header_str = header_str.replace("{{ name }}", f'"{name.lower()}"').strip()

    template_text = header_str + template_without_header
    template_text = template_text.replace("opencv-python", "opencv")
    return template_text.replace("matplotlib", "matplotlib-base")


def save_meta_yaml(meta_yaml_content: str, bioconda: bool = False) -> str:
    """Save the meta.yaml content to the repository.

    Args:
        meta_yaml_content (str): The content of the meta.yaml file.
        bioconda (bool): Flag to use bioconda dependencies

    Returns:
        str: The name of the output file.
    """
    output_file = "meta_bioconda.yaml" if bioconda else "meta.yaml"
    repo_root = Path.resolve(Path(__file__).parent)
    meta_yaml_path = repo_root / "conda_recipes" / output_file

    with meta_yaml_path.open("w") as file:
        file.write(meta_yaml_content)
    return output_file


if __name__ == "__main__":
    bioconda_flag = "--bioconda" in sys.argv
    pyproject_data = load_toml()
    meta_yaml_content = create_meta_yaml(pyproject_data, bioconda=bioconda_flag)
    output_file = save_meta_yaml(meta_yaml_content, bioconda=bioconda_flag)
