# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import tomllib
from pathlib import Path


# -- Paths ---------------------------------------------------------------------

_lib_path = Path(__file__).resolve().parents[2]/'src'
sys.path.insert(0, _lib_path.as_posix())

# Version from the library configuration file
with open(_lib_path/'pystarburst99/pystarburst99.toml', 'rb') as fp:
    _pySB99_cfg = tomllib.load(fp)


# -- Project information -------------------------------------------------------

project = 'pyStarburst99'
copyright = '2025, Calum Hawcroft'
author = 'Calum Hawcroft'
release = _pySB99_cfg['metadata']['version']


# -- General configuration -----------------------------------------------------

extensions = ['myst_nb',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax']

# Markdown configuration
myst_enable_extensions = ['dollarmath', 'colon_fence', 'deflist']
myst_heading_anchors = 2

source_suffix = {'.rst': 'restructuredtext',
                 '.md': 'myst-nb'}

# The documentation has no notebooks to execute
nb_execution_mode = "off"

# Autodoc options
autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []


# -- HTML output configuration -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']