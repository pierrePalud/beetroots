Two-dimensional Gaussian mixture
================================

This example shows how the ``beetroots`` package can be used to perform inversion on a simple case: a 2D Gaussian mixture.

TODO: explain that what is needed to prepare the run is only

* a dedicated ``Likelihood`` class to encode the data-fidelity term.
* a dedicated ``Simulation`` class to setup the observation and posterior

already implemented:

* the identity forward map
* the prior encoding validity intervals (a smooth variant for the sampler)
* the ``Posterior`` class
* the ``Sampler``class
* the ``Saver`` class
* the inversion itself, with the ``RunMCMC`` class
* the inversion result extraction, with the ``ResultMCMC`` class

Ajouter un schéma des différentes classes, et de quand elles sont appelées.


.. code:: bash

   poetry shell
   python examples/gaussian_mixture/gaussian_mixture_simu.py input_params_pmtm0p9.yaml

or in one line:

.. code:: bash

   poetry run python examples/gaussian_mixture/gaussian_mixture_simu.py input_params_pmtm0p9.yaml

To check other inpout file: run with ``input_params_pmtm0p9.yaml``

TODO: Description of the setup and of the run.


Output:

>>> poetry run python examples/gaussian_mixture/gaussian_mixture_simu.py input_params_pmtm0p9.yaml
starting sampling
starting from a random point
100%|█████████████████████████████████████████████████████████| 10000/10000 [00:49<00:00, 200.76it/s]
sampling done
N = 1, L (fit) = 2, D_sampling = 2, D = 2
starting clppd plots
starting plot of accepted frequencies
plots of accepted frequencies done
starting plot of log proba accept
plots of log proba accept done
starting plot of objective function
plot of objective function done
100%|█████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.41s/it]
starting Bayesian p-value plots
Bayesian p-value plots plots done
starting plot proportion of well reconstructed pixels
plot proportion of well reconstructed pixels done
Simulation and analysis finished. Total duration : 00:01:04 s


The images will be in ``outputs/gaussian_mixture_[yyyy]-[mm]-[dd]_[hh]/img``.
The ESS and MSE values will be in ``outputs/gaussian_mixture_[yyyy]-[mm]-[dd]_[hh]/data/output``.

True distribution:

.. image:: img/gaussian_mixture/true_gaussians.png
   :width: 50%
   :alt: True Gaussians
   :align: center

Result of the sampling algorithm: 2D histogram

.. image:: img/gaussian_mixture/results.png
   :width: 50%
   :alt: Sampling results
   :align: center
