# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "beetroots"
copyright = "2024, Pierre Palud"
author = "Pierre Palud"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"


#! duration does not seem to work
extensions = [
    # Official Sphinx extensions
    # https://www.sphinx-doc.org/en/master/usage/extensions/index.html
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    # "sphinx.ext.autosectionlabel",  # Allow reference sections using its title
    # "sphinx.ext.autosummary",  # Generate autodoc summaries
    "sphinx.ext.coverage",  # Collect doc coverage stats
    "sphinx.ext.doctest",  # Test snippets in the documentation
    # "sphinx.ext.duration",  # Measure durations of Sphinx processing
    "sphinx.ext.extlinks",  # Markup to shorten external links
    # "sphinx.ext.githubpages",  # Publish HTML docs in GitHub Pages
    # "sphinx.ext.graphviz",  # Add Graphviz graphs
    # "sphinx.ext.ifconfig",  # Include content based on configuration
    # "sphinx.ext.imgconverter",  # A reference image converter using Imagemagick
    "sphinx.ext.inheritance_diagram",  # Include inheritance diagrams
    # "sphinx.ext.intersphinx",  # Link to other projects’ documentation
    # "sphinx.ext.linkcode",  # Add external links to source code
    # "sphinx.ext.imgmath",  # Render math as images
    "sphinx.ext.mathjax",  # Render math via JavaScript
    # "sphinx.ext.jsmath",  # Render math via JavaScript
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.todo",  # Support for todo items # .. todo:: directive
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    # Non-official Sphinx extensions need to be installed
    # https://github.com/sphinx-contrib/
    # "sphinxcontrib.bibtex",  # Sphinx extension for BibTeX style citations
    # "sphinxcontrib.proof",  # Sphinx extension to typeset theorems, proofs
    # Non-official Sphinx extension for matplotlib plots
    # https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html?highlight=plot_directive#module-matplotlib.sphinxext.plot_directive
    "matplotlib.sphinxext.plot_directive",  # .. plot:: directive for plt.plot
    "myst_parser",
    "sphinx_design",
    # "sphinx_gallery",
    "nbsphinx",
    "nbsphinx_link",
]
napoleon_google_docstring = False

autodoc_default_options = {"ignore-module-all": True}
autoclass_content = "both"

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*_old*", "tests*"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
