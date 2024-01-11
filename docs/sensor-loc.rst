Sensor localization problem
===========================

This example shows how the ``beetroots`` package can be used to perform inversion on a slightly more complicated case: the sensor localization problem.
This second example appears in Section IV.B of :cite:t:`paludEfficientSamplingNon2023`.



TODO: explain that what is needed to prepare the run is only

* a dedicated ``Likelihood`` class to encode the data-fidelity term.
* a dedicated ``Simulation`` class to setup the observation and posterior
* a dedicated ``ForwardMap`` class to compute distances between sensors

already implemented:

* the prior encoding validity intervals (a smooth variant for the sampler)
* the ``Posterior`` class
* the ``Sampler`` class
* the ``Saver`` class
* the inversion itself, with the ``RunMCMC`` class
* the inversion result extraction, with the ``ResultMCMC`` class

Ajouter un schéma des différentes classes, et de quand elles sont appelées.

.. code:: bash

    poetry run python examples/sensor_loc/sensor_loc_simu.py input_params_pmtm0p9.yaml


.. code:: bash

    poetry run python examples/sensor_loc/sensor_loc_simu.py input_params_pmtm0p1.yaml


Incompatible model and observation :

.. code:: bash

    poetry run python examples/sensor_loc/sensor_loc_simu.py input_params_pmtm0p1_false.yaml


The images will be in ``outputs/sensor_loc_[yyyy]-[mm]-[dd]_[hh]/img``.
The ESS and MSE values will be in ``outputs/sensor_loc_[yyyy]-[mm]-[dd]_[hh]/data/output``.


>>> python examples/sensor_loc/sensor_loc_simu.py input_params_pmtm0p9.yaml
starting sampling
starting from a random point
100%|██████████████████████████████████████████████████| 3000/3000 [02:39<00:00, 18.76it/s]
sampling done
N = 8, L (fit) = 11, D_sampling = 2, D = 2
starting clppd plots
starting plot of accepted frequencies
plots of accepted frequencies done
starting plot of log proba accept
plots of log proba accept done
starting plot of objective function
plot of objective function done
100%|███████████████████████████████████████████████████████| 8/8 [00:16<00:00,  2.08s/it]
starting Bayesian p-value plots
Bayesian p-value plots plots done
starting plot proportion of well reconstructed pixels
plot proportion of well reconstructed pixels done
Simulation and analysis finished. Total duration : 00:03:13 s


.. image:: img/sensor_loc/graph_sensors.PNG
   :width: 50%
   :alt: Observation graph
   :align: center


Sampling results: 2D histograms of the marginal distributions

.. image:: img/sensor_loc/marginals.PNG
   :width: 50%
   :alt: Sampling results
   :align: center


See sensor_loc_pb_definition for more details on the construction of the problem.
