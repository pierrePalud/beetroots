Application to NGC 7023
=======================

NGC 7023:

.. code:: bash

    poetry run python beetroots/simulations/astro/real_data/ngc7023_nn.py input_params.yaml



Orion Bar:

.. code:: bash

    poetry run python beetroots/simulations/astro/real_data/orionbar_nn.py input_params.yaml


.. code:: bash

    poetry run python beetroots/simulations/astro/real_data/orionbar_nn.py input_params_nochp10.yaml



Carina nebula:

.. code:: bash

    poetry run python beetroots/simulations/astro/real_data/carina_nn.py input_params_1p7_with_spatial_regu.yaml


.. code:: bash

    poetry run python beetroots/simulations/astro/real_data/carina_nn.py input_params_1p7_with_spatial_regu_optim_mle.yaml


OMC1:

.. code:: bash

    poetry run python beetroots/simulations/astro/real_data/omc1_nn.py input_params_45per_with_spatial_regu_lines1234_fixed_angle.yaml


.. code:: bash

    poetry run python beetroots/simulations/astro/real_data/omc1_nn.py input_params_45per_with_spatial_regu_lines1234_fixed_angle_optim_mle.yaml
