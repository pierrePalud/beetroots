Gallery of examples
===================

This gallery contains several application examples for the ``beetroots`` package to illustrate diverse features.



We first explain **how to use this package** and :ref:`How to set up the input file` associated to your inversion.

We then propose **detailed walkthroughs for three examples**:

* Sampling from a :ref:`Two-dimensional Gaussian mixture`
* Solving the :ref:`Sensor localization problem` introduced in :cite:t:`ihlerNonparametricBeliefPropagation2005`
* An :ref:`Application to a synthetic ISM map`, introduced in Section IV.C of :cite:t:`paludEfficientSamplingNon2023`
* An :ref:`Application to NGC 7023`, a 1-pixel observation already analyzed in :cite:t:`joblinStructurePhotodissociationFronts2018`


Finally, we present **secondary yet important features** of the ``beetroots`` package:

* :ref:`How to visualize maps` of observations or of estimators
* :ref:`How to adjust the likelihood approximation parameter` in ``beetroots.modelling.likelhoods.approx_censored_add_mult``



.. toctree::
   :maxdepth: 1
   :caption: Gallery

   create_input_files

   gaussian-mixture
   sensor-loc
   sensor_loc_pb_definition
   application-toycase-astro
   application-ngc7023-astro

   visualize_maps
   likelihood-params-adjustment
