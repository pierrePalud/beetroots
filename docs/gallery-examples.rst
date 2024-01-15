Gallery of examples
===================

This gallery contains several application examples for the ``beetroots`` package to illustrate diverse features.


.. Before diving into the code and examples, one may point that other inversion codes already exist (in astrophysics, but not only).
.. We first explain :ref:`Why we needed a new inversion code in the first place`.


We explain **how to use this package** for setup and visualization:

* :ref:`How to visualize maps` of observations or of estimators
* :ref:`How to set up the input file` associated to your inversion

We then propose **detailed walkthroughs for three examples**:

* Sampling from a :ref:`Two-dimensional Gaussian mixture`
* Solving the :ref:`Sensor localization problem` introduced in :cite:t:`ihlerNonparametricBeliefPropagation2005`
* An :ref:`Application to a synthetic ISM map`, introduced in Section IV.C of :cite:t:`paludEfficientSamplingNon2023`
.. * An :ref:`Application to NGC 7023`, a one-pixel observation already analyzed in :cite:t:`joblinStructurePhotodissociationFronts2018`


Finally, we present **secondary yet important features** of the ``beetroots`` package:

.. * How to :ref:`Run multiple inversions in parallel`, using ``slurm``
* :ref:`How to adjust the likelihood approximation parameter` in ``beetroots.modelling.likelhoods.approx_censored_add_mult``
.. * :ref:`How to reproduce a sampling`, thanks to random seeds and the ``numpy`` random generator



.. toctree::
   :maxdepth: 1
   :caption: Gallery

   .. other_inversion_codes

   visualize_maps
   create_input_files

   gaussian-mixture
   sensor-loc
   sensor_loc_pb_definition
   application-toycase-astro
   .. application-ngc7023-astro

   .. run_inversion_in_parallel

   likelihood-params-adjustment

   .. sampling-reproducibility
