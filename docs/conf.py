# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = "Paidiverpy"
copyright = "2024-2025, Paidiver"
author = "Paidiver Developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.1.4"
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    # "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
    'numpydoc',
    "autoapi.extension",
    "myst_parser",
    'sphinx_last_updated_by_git',
    'sphinx_codeautolink',
    'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Use autoapi.extension to run sphinx-apidoc -------

autoapi_dirs = ["../src/paidiverpy"]
autoapi_root = 'api'

autoapi_keep_files = True
autoapi_options = [
    # "members",
    # "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
    "no-private-members",
    "no-inherited-members",
    "no-class-attributes",
]

# autoapi_ignore = [
#     "paidiverpy.metadata_parser.metadata_parser",  # Exclude it as a submodule
# ]

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "sphinx_rtd_theme"
html_theme = 'sphinx_book_theme'

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#


html_logo = "_static/logo_paidiver_docs.png"
html_favicon = '_static/favicon_paidiver.ico'
# For sphinx_book_theme:
html_theme_options = {
    "repository_url": "https://www.github.com/paidiver/paidiverpy",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "repository_branch": "master",
    # "html_logo": "_static/argopy_logo_long.png",
    "logo": {"image": html_logo},
    # "display_version": True,
    # "logo_only": True,
    # "show_navbar_depth": 2,  # https://sphinx-book-theme.readthedocs.io/en/stable/customize/sidebar-primary.html?highlight=logo#control-the-depth-of-the-left-sidebar-lists-to-expand

    "show_nav_level": 1,  # https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/navigation.html#control-how-many-navigation-levels-are-shown-by-default
    'collapse_navigation': False,  # https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/navigation.html#remove-reveal-buttons-for-sidebar-items
    # 'show_toc_level': 3,  # https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/page-toc.html#show-more-levels-of-the-in-page-toc-by-default
    # 'launch_buttons': { "thebe": True}
}
# nbsphinx options
nbsphinx_execute = 'always'
nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]

nbsphinx_thumbnails = {
    'gallery/thumbnail-from-conf-py': 'gallery/a-local-file.png',
    'gallery/*-rst': 'images/notebook_icon.png',
    'orphan': '_static/favicon.svg',
}

# -- Options for Intersphinx

intersphinx_mapping = {
    'IPython': ('https://ipython.readthedocs.io/en/stable/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'python': ('https://docs.python.org/3/', None),
}

autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "fullqualname": True
}

add_module_names = True
