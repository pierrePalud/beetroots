.. beetroots documentation master file, created by
   sphinx-quickstart on Wed Jan  3 16:34:24 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to beetroots's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Beetroots (BayEsian invErsion with spaTial Regularization of nOisy multi-line ObservaTion mapS) is a Python package that performs Bayesian inference of physical parameters from multispectral-structured cubes with a dedicated sampling algorithm.
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
   The GitHub repository can be found
   `here <https://github.com/einigl/ism-model-nn-approximation>`_,
   the package `here <https://pypi.org/project/nnbma/>`_
   and the corresponding documentation `here <https://ism-model-nn-approximation.readthedocs.io/en/latest/?badge=latest>`_.
   The paper presenting this package is

      P. Palud, L. Einig, F. Le Petit, E. Bron, P. Chainais, J. Chanussot, J. Pety, P.-A. Thouvenin and ORION-B consortium - **Neural network-based emulation of interstellar medium models**, *Astronomy & Astrophysics*, 2023, 678, pp.A198. DOI: `10.1051/0004-6361/202347074 <https://doi.org/10.1051/0004-6361/202347074>`_


============
Installation
============

To perform an inversion, we recommend installing the package.
The package can be installed with ``pip``:

.. code-block:: bash

   pip install beetroots

or by cloning the repo.
To clone, install and test the package, run:

.. code-block:: bash

   git clone git@github.com:pierrePalud/beetroots.git
   cd beetroots
   poetry install
   poetry shell
   poetry run pytest


Alternatively, you can download a zip file.


=======================================================
Package structure and how to adapt it to your use cases
=======================================================

TODO: correct.

The ``run_simulations.sh`` file is used to run a sampling process.
The sampling process to run and the corresponding main params are defined in ``beetroots/__main__.py``.

1. it instantiates a *Simulation* object
2. it run the *Simulation* ``setup`` method. This method sets the whole inference problem: importation of observation, definitions of forward map, of likelihood, posterior, etc. and initializes the corresponding output folder.
3. it runs the *Simulation* ``main`` method, which runs optimization / sampling (depending on the input).

The ``simulations`` folder contains the most general classes of the repo.
The most important file is ``abstract_simulation.py``, which got a bit too big.
A priori: only need to go there to choose which likelihood to use in posterior distribution definitions.

The ``sampler`` folder contains the definition of the sampler (``mysampler.py``) and of the frequent saves of the MC evolution (``saver.py``).
A priori: should change the ``sample`` method of the ``MySampler`` class to incorporate the sampling of the auxiliary variable.

The ``modelling`` folder contains all the ingredient of the inverse problem.
Its main file is the ``posterior.py``.
A priori, the 2 main parts to change in this folder:

- the likelihood (add a lognormal one)
- the posterior file

The ``space_transform`` folder encodes bijections for switching between parameters natural space (linear) and the parameter space used for sampling (normalized log).
Mainly used at the saving step, to save parameters in their natural space.
No need to change anything there.

.. note::

   All classes in ``modelling`` are implemented in a way tries to avoid duplicated computations.
   Since I could not cache functions with complicated inputs (dicts, np.ndarray, etc.), I compute everything and store it in dedicated dict.
   There are 2 such dictionaries:

   - ``forward_map_evals``: computed with the ``evaluate_all_forward_map`` ForwardMap method. It computes the forward map image, gradient, hessian, gradient of the log and/or hessian of the log, depending on the necessity.
   - ``nll_utils``: computed with the ``evaluate_all_nll_utils`` likelihood method. The contents depend on the considered likelihood class. This dict is empty for all likelhoods except for the ``approx_censored_add_mult.py``. For this class, the nll_utils dict contains all biases :math:`m_a`, :math:`m_m`, variances :math:`s_a`, :math:`s_m`, the mixing weight :math:`\lambda` and all their gradients and hessian.

   Both of them are made / updated with the ``compute_all`` method of the ``Posterior`` class.


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
