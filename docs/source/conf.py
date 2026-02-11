import os
import sys
from importlib.metadata import version, PackageNotFoundError
import warnings

sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Locat'
copyright = '2024, Wes Lewis, Francesco Strino, Fabio Parisi, Yariv Aizenbud, Yuval Kluger'
author = 'Lewis, Francesco Strino, Fabio Parisi, Yariv Aizenbud, Yuval Kluger'
github_url = "https://github.com/KlugerLab/Locat"

# Get the version using scm (https://pypi.org/project/setuptools-scm/7.0.5/)
try:
    release_scm = version('locat')
except PackageNotFoundError:
    warnings.warn('The package "locat" is not installed. '
                  'Run e.g. "pip install -e ." to make the SCM version available')
    release_scm = '0.0.0-dev'
release = '.'.join(release_scm.split('.')[:3])
version = release
branch = 'main' if 'dev' in release_scm else f'v{version}'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'myst_parser',
    'numpydoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.linkcode',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_context = {
    "default_mode": "light",
    "github_user": "KlugerLab",
    "github_repo": "Locat",
    "github_version": "main",
    "doc_path": "docs",
}

html_sidebars = {
    "examples": []
}

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": github_url,
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

plot_include_source = True


def linkcode_resolve(domain, info):
    """
    Called by sphinx.ext.linkcode to add a link

    See https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
    """
    if domain == 'py' and info['module']:
        filename = info['module'].replace('.', '/')
        return f"{github_url}/tree/{branch}/{filename}.py"
