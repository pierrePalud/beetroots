How to adjust the likelihood approximation parameter
====================================================

Introduction
------------

The likelihood associated to the following noise model

.. math::

    y_{n\ell} = \epsilon^{(m)}_{n\ell} f_\ell(\theta_n) + \epsilon^{(a)}_{n\ell}, \quad \epsilon^{(a)}_{n\ell} \sim \mathcal{N}(0, \sigma_{a,n\ell}^2), \quad \epsilon^{(m)}_{n\ell} \sim \text{Lognormal}(- \sigma_m^2/2, \sigma_m^2)

cannot be written simply without resortint to a hierarchical model.
The :class:`.MixingModelsLikelihood` class implements the approximation proposed in Section II of :cite:t:`paludEfficientSamplingNon2023`.
The approximation combines a Gaussian approximation:

.. math::

    y_{n\ell} \simeq f_\ell(\theta_n) + e^{(a)}_{n\ell}, \quad e^{(a)}_{n\ell} \sim \mathcal{N}(m_{a,n\ell}, s_{a,n\ell}^2)

and a lognormal approximation

.. math::

    y_{n\ell} \simeq e^{(m)}_{n\ell} f_\ell(\theta_n), \quad e^{(m)}_{n\ell} \sim \text{Lognormal}(m_{m,n\ell}, s_{m,n\ell}^2)


where :math:`m_{m,n\ell}`, :math:`s_{m,n\ell}^2`, :math:`m_{a,n\ell}` and :math:`s_{a,n\ell}^2` are obtained with moment matching.
The combination :math:`\widetilde{\pi}` is defined as a weighted geometric mean of the two likelihood functions :math:`\pi^{(a)}` and :math:`\pi^{(m)}` associated to these two approximations:


.. math::

    \widetilde{\pi}(y_{n\ell} \vert \theta_n) \propto \pi^{(a)}(y_{n\ell} \vert \theta_n)^{1 - \lambda(\theta_n, a_\ell)} \; \pi^{(m)}(y_{n\ell} \vert \theta_n)^{\lambda(\theta_n, a_\ell)}


The weight :math:`\lambda(\theta_n, a_\ell) \in [0, 1]` of the multiplicative approximation depends on a parameter :math:`a_\ell` that pinpoints special positions:


.. image:: img/likelihood_params_optim/lambda_function.PNG
    :width: 65%
    :align: center

In this document, we explain how to adjust the parameter to obtain a good approximation of the true likelihood.
The used approach is presented in the Appendix A of :cite:t:`paludEfficientSamplingNon2023`.


Perform optimization
--------------------

For real observations, it is very simple to tune the parameters :math:`a_\ell`, as one simply needs to run:

.. code:: bash

    python beetroots/approx_optim/nn_bo_real_data.py ./data/ngc7023/input_nn_bo_fast.yaml

with

.. code-block:: yaml
    :caption: input_nn_bo_fast.yaml
    :name: nn-bo-fast

    list_lines: # List[str]: names of the lines for which the parameters are to be adjusted
        - "co_v0_j11__v0_j10"
        - "co_v0_j12__v0_j11"
        - "co_v0_j13__v0_j12"
        - "co_v0_j15__v0_j14"
        - "co_v0_j16__v0_j15"
        - "co_v0_j17__v0_j16"
        - "co_v0_j18__v0_j17"
        - "co_v0_j19__v0_j18"
        #
        - "h2_v0_j2__v0_j0"
        - "h2_v0_j3__v0_j1"
        - "h2_v0_j4__v0_j2"
        - "h2_v0_j5__v0_j3"
        - "h2_v0_j6__v0_j4"
        - "h2_v0_j7__v0_j5"
        #
        - "chp_j1__j0"
        - "chp_j2__j1"
        - "chp_j3__j2"

    # data
    filename_int: "Nebula_NGC_7023_Int.pkl" # map of observations
    filename_err: "Nebula_NGC_7023_Err.pkl" # map of additive standard deviation
    sigma_m_float_linscale: 1.3 # multiplicative noise standard deviation in linear scale (sigma_m = log(sigma_m_float_linscale)). Constant over the full map.

    # parameters to run the optimization
    simu_init:
        name: "bo_nn_ngc7023"
        D: 5
        D_no_kappa: 4
        K: 20
        log10_f_grid_size: 100
        N_samples_y: 10_000 # 200_000 # 250_000
        max_workers: 3

    # parameters to set up the optimization
    main_params:
        dict_forward_model:
            forward_model_name: "meudon_pdr_model_dense"
            force_use_cpu: true
            fixed_params: # must contain all the params in list_names of the SImulation object. Values are in linear scale.
            kappa: null
            P: null
            radm: null
            Avmax: null
            angle: 60.0
            is_log_scale_params: # defines the scale to work with for each param (either log or lin)
            kappa: True
            P: True
            radm: True
            Avmax: True
            angle: False
        #
        lower_bounds_lin:
            - 1.0e-1 # kappa
            - 1.0e+5 # Pth
            - 1.0e+0 # G0
            - 1.0e+0 # AVtot
            - 0.0 # angle
        upper_bounds_lin:
            - 1.0e+1 # kappa
            - 1.0e+9 # Pth
            - 1.0e+5 # G0
            - 4.0e+1 # AVtot
            - 60.0 # angle
        n_iter: 40


Results
-------

Evolution of the Gaussian process mean during the optimization process:

.. image:: img/likelihood_params_optim/gp_n0_co_v0_j17__v0_j16_mean_iter5.PNG
    :width: 32%
.. image:: img/likelihood_params_optim/gp_n0_co_v0_j17__v0_j16_mean_iter20.PNG
    :width: 32%
.. image:: img/likelihood_params_optim/gp_n0_co_v0_j17__v0_j16_mean_iter40.PNG
    :width: 32%


Evolution of the Gaussian process standard deviation (i.e., uncertianty on the cost function) during the optimization process:

.. image:: img/likelihood_params_optim/gp_n0_co_v0_j17__v0_j16_std_iter5.PNG
    :width: 32%
.. image:: img/likelihood_params_optim/gp_n0_co_v0_j17__v0_j16_std_iter20.PNG
    :width: 32%
.. image:: img/likelihood_params_optim/gp_n0_co_v0_j17__v0_j16_std_iter40.PNG
    :width: 32%


Final weight function :math:`\lambda` :

.. image:: img/likelihood_params_optim/hist_final_log_f_Theta_n0_co_v0_j17__v0_j16.PNG
    :width: 80%
    :align: center


Among other intermediate outputs, the command above will create a ``best_params.csv`` file that is needed for any inference based on the :class:`.MixingModelsLikelihood` likelihood.
