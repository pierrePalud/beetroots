.. beetroots documentation master file, created by
   sphinx-quickstart on Wed Jan  3 16:34:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to beetroots's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Beetroots (BayEsian infErence with spaTial Regularization of nOisy multi-line ObservaTion mapS) is a Python package that performs Bayesian inference of physical parameters from multispectral-structured cubes with a dedicated sampling algorithm.
Thanks to this sampling algorithm, `beetroots` provides maps of credibility intervals along with estimated maps.

The sampling algorithm is introduced in

   P. Palud, P.-A. Thouvenin, P. Chainais, E. Bron, and F. Le Petit - **Efficient sampling of non log-concave posterior distributions with mixture of noises**, *IEEE Transactions on Signal Processing*, vol. 71, pp. 2491 -- 2501, 2023. DOI: `10.1109/TSP.2023.3289728 <https://doi.org/10.1109/TSP.2023.3289728>`_


Such inversions rely on a forward model that is assumed to emulate accurately the physics of the observed environment.
In parallel of the inversion, `beetroots` tests this hypothesis to evaluate the validity of the inference results.
The testing method is described in (in French)

   P. Palud, P. Chainais, F. Le Petit, P.-A. Thouvenin and E. Bron - **Problèmes inverses et test bayésien d'adéquation du modèle**, *GRETSI - Groupe de Recherche en Traitement du Signal et des Images* in *29e Colloque sur le traitement du signal et des images*, Grenoble, pp. 705 -- 708, 2023.

This package was applied e.g., to infer physical conditions in different regions of the interstellar medium in

   P. Palud, P.-A. Thouvenin, P. Chainais, E. Bron, F. Le Petit and ORION-B consortium - **Bayesian inversion of large interstellar medium observation maps**, in prep

It was also exploited to assert and compare the relevance of tracers and combination of tracers to constrain physical conditions in

   L. Einig, P. Palud, A. Roueff, P.-A. Thouvenin, P. Chainais, E. Bron, F. Le Petit, J. Pety, J. Chanussot and ORION-B consortium -  **Entropy-based selection of most informative observables for inference from interstellar medium observations**, in prep

.. note::

   Astrophysics applications rely on a neural network-based approximation of the forward model for

   - faster evaluation
   - ability to evaluate derivatives

   The package used to derive this approximation is `nnbma` (Neural Network-Based Model Approximation).
   Here are the links towards the corresponding
   `GitHub repository <https://github.com/einigl/ism-model-nn-approximation>`_,
   `PyPi package <https://pypi.org/project/nnbma/>`_
   and `documentation <https://ism-model-nn-approximation.readthedocs.io/en/latest/?badge=latest>`_.
   The paper presenting this package is

      P. Palud, L. Einig, F. Le Petit, E. Bron, P. Chainais, J. Chanussot, J. Pety, P.-A. Thouvenin and ORION-B consortium - **Neural network-based emulation of interstellar medium models**, *Astronomy & Astrophysics*, 2023, 678, pp.A198. DOI: `10.1051/0004-6361/202347074 <https://doi.org/10.1051/0004-6361/202347074>`_


============
Installation
============

To prepare and perform an inversion, we recommend installing the package.
The package can be installed with ``pip``:

.. code-block:: bash

   pip install beetroots

or by cloning the repo.
To clone, install and test the package, run:

.. code-block:: bash

   git clone git@github.com:pierrePalud/beetroots.git
   cd beetroots
   poetry install # or poetry install -E notebook -E docs for extra dependencies
   poetry shell
   pytest


=======================================================
Package structure and how to adapt it to your use cases
=======================================================

This package is large and contains a lot of python modules.
To facilitate code exploration and use, here is an un-rigorous UML class diagram of the code:

.. image:: ../examples/img/uml_classes/uml_classes_diagram.svg
   :width: 100%
   :alt: UML class diagram
   :align: center

|

The examples in the :ref:`Gallery of examples` clarify the package structure and in particular what the user needs to interact with.
This diagram is maintained here for completeness.


.. toctree::
   :maxdepth: 4
   :caption: Contents

   modules

.. toctree::
   :maxdepth: 2

   gallery-examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

Here are the references used throughout this documentation:

.. bibliography::
