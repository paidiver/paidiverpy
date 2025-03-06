"""Generate the meta.yaml file for the conda recipe."""

from pathlib import Path
import toml
from jinja2 import Template

TEMPLATE_STR = """
package:
    name: {{ name|lower }}
    version: {{ version }}

source:
    url: https://pypi.org/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
    sha256: 9b4145561e05ffb854dca446da8490cbf2705a214b7d1ebeed16983c052512a0

build:
    entry_points:
    - {{ name|lower }} = cli.main:main
    noarch: python
    script: {% raw %}{{ PYTHON }}{% endraw %} -m pip install . -vv --no-deps --no-build-isolation
    number: 0

requirements:
    host:
    - python {% raw %}{{ python_min }}{% endraw %}

    - setuptools >=64.0.0
    - setuptools-scm
    - wheel
    - pip
    run:
    - python >={% raw %}{{ python_min }}{% endraw %}

    {% for dep in dependencies %}
    - {{ dep }}
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

"""


def load_toml() -> dict:
    """Load the pyproject.toml file.

    Returns:
        dict: The pyproject.toml data.
    """
    repo_root = Path.resolve(Path(__file__).parent)
    toml_path = repo_root / "pyproject.toml"

    with toml_path.open() as file:
        return toml.load(file)


def create_meta_yaml(pyproject_data: dict) -> str:
    """Create the meta.yaml file content.

    Args:
        pyproject_data (dict): The pyproject.toml data.

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

    template_without_header = template.render(
        name=name, version=version, description=description, license_file=license_file, dependencies=dependencies
    )

    header_str = """
    {% set python_min = {{ python_min }} %}
    """
    header_str = header_str.replace("{{ python_min }}", f'"{python_min}"').strip()
    template_text = header_str + template_without_header
    template_text = template_text.replace("opencv-python", "opencv")
    return template_text.replace("matplotlib", "matplotlib-base")


def save_meta_yaml(meta_yaml_content: str) -> None:
    """Save the meta.yaml content to the repository.

    Args:
        meta_yaml_content (str): The content of the meta.yaml file.
    """
    repo_root = Path.resolve(Path(__file__).parent)
    meta_yaml_path = repo_root / "conda_recipes" / "meta.yaml"

    with meta_yaml_path.open("w") as file:
        file.write(meta_yaml_content)


pyproject_data = load_toml()
meta_yaml_content = create_meta_yaml(pyproject_data)
save_meta_yaml(meta_yaml_content)
