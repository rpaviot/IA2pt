"""Sphinx configuration for the IA2pt documentation (Read the Docs)."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "IA2pt"
author = "Romain Paviot"
copyright = "2026, Romain Paviot"
release = "0.2.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
# the heavy scientific dependencies are not installed on the docs builder
autodoc_mock_imports = ["pyccl", "fastpt", "iminuit", "nautilus", "emcee"]
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

html_theme = "furo"
html_title = "IA2pt"
