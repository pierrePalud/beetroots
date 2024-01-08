Adjusting likelihood parameter
==============================

Mixture of additive Gaussian noise and multiplicative lognormal noise sources.
How to adjust the parameter presented in \[1\] ?


Likelihood approximation parameters optimization

.. code:: bash

    poetry run python beetroots/approx_optim/nn_bo_toycase.py
    poetry run python beetroots/approx_optim/nn_bo_orionbar.py
    poetry run python beetroots/approx_optim/nn_bo_ngc7023.py

Updated version (TODO: finish)

.. code:: bash

    poetry run python beetroots/approx_optim/nn_bo_real_data.py ./data/ngc7023/input_nn_bo.yaml
