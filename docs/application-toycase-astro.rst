Application to a synthetic ISM map
================================================

With nn-based approximation of forward model:

.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn.py nn_N10_fixed_angle.yaml


.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn.py nn_N10_fixed_AV.yaml


.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn.py nn_N64_fixed_angle.yaml


With hierarchical model

.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn.py nn_N10_fixed_angle_nomtm.yaml


.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn_hierarchical_posterior.py


With polynomial approximation of forward model

.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_polyreg.py


.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_polyreg_hierarchical_posterior.py
