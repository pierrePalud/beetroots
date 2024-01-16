Application to a synthetic ISM map
==================================

This example appears in Section IV.C of :cite:t:`paludEfficientSamplingNon2023`.
It studies a synthetic molecular cloud seen edge on.
The figure below illustrates the structure of this molecular cloud.


.. image:: img/toy_case_astro/toy_structure.png
   :width: 75%
   :align: center

|

The considered forward map is the neural network approximation of the Meudon PDR code derived in :cite:t:`paludNeuralNetworkbasedEmulation2023`. It takes as inputs :math:`D=5` physical parameters:

* an observational nuisance parameter :math:`\kappa \in [0.1, 10]`
* a thermal pressure :math:`P_{th} \in [10^5, 10^9]` K cm :math:`^{-3}`
* a radiation field intensity :math:`G_0 \in [10^0, 10^5]` (Habing units)
* a visual extinction :math:`A_V \in [1, 40]` mag
* an inclination angle :math:`\alpha \in [0, 60]` deg

In this toy case, the inclination angle is set to :math:`\alpha=0` deg, while the others are inferred.
Here are the true maps of the thermal pressure, radiation field intensity and visual extinction. The full map of :math:`\kappa` is set to 1.


.. image:: img/toy_case_astro/toy_true.png
   :width: 100%
   :alt: classes to prepare
   :align: center

|

The forward model is restricted to :math:`L=10` CO lines.
These lines are generated from the true maps and altered with additive noise, multipliative noise and censorship:


.. image:: img/toy_case_astro/observation_line_0.png
   :width: 32%
.. image:: img/toy_case_astro/observation_line_9.png
   :width: 32%
.. image:: img/toy_case_astro/proportion_censored_lines_original.png
   :width: 32%

|


Python simulation preparation
-----------------------------

Here are the classes that are necessary to sample from this distribution.
The green classes indicate the already implemented classes, and the red classes indicate the classes to implement.


.. image:: img/simulation-structures/astro-appli.svg
   :width: 85%
   :alt: classes to prepare
   :align: center

|

For astrophysics applications, all required classes are already implemented.


YAML file
---------

.. code-block:: yaml
    :caption: nn_N64_fixed_angle.yaml
    :name: nn_N64_fixed_angle

    simu_init:
        simu_name: "astro_toy_N64_fixed_angle"
        cloud_name: "astro_toy_N64"
        max_workers: 10
        #
        params_names:
            kappa: $\kappa$
            P: $P_{th}$
            radm: $G_0$
            Avmax: $A_V^{tot}$
            angle: $\alpha$
        #
        list_lines_fit:
            - "co_v0_j4__v0_j3"
            - "co_v0_j5__v0_j4"
            - "co_v0_j6__v0_j5"
            - "co_v0_j7__v0_j6"
            - "co_v0_j8__v0_j7"
            - "co_v0_j9__v0_j8"
            - "co_v0_j10__v0_j9"
            - "co_v0_j11__v0_j10"
            - "co_v0_j12__v0_j11"
            - "co_v0_j13__v0_j12"
    #
    to_run_optim_map: true
    to_run_mcmc: true
    #
    forward_model:
        forward_model_name: "meudon_pdr_model_dense"
        force_use_cpu: false
        fixed_params: # must contain all the params in list_names of the Simulation object. Values are in linear scale.
            kappa: null
            P: null
            radm: null
            Avmax: null
            angle: 0.0
        is_log_scale_params: # defines the scale to work with for each param (either log or lin)
            kappa: True
            P: True
            radm: True
            Avmax: True
            angle: False
    #
    #
    sigma_a_float: 1.38715e-10
    sigma_m_float_linscale: 1.1
    #
    # prior indicator
    prior_indicator:
        indicator_margin_scale: 1.0e-1
        lower_bounds_lin:
            - 1.0e-1 # kappa
            - 1.0e+5 # thermal pressure
            - 1.0e+0 # G0
            - 1.0e+0 # AVtot
            - 0.0 # angle
        upper_bounds_lin:
            - 1.0e+1 # kappa
            - 1.0e+9 # thermal pressure
            - 1.0e+5 # G0
            - 4.0e+1 # AVtot
            - 60.0 # angle
    #
    list_gaussian_approx_params: []
    mixing_model_params_filename: ["best_params.csv"]
    #
    # spatial prior
    with_spatial_prior: true
    spatial_prior:
        name: "L2-laplacian"
        use_next_nearest_neighbors: false
        initial_regu_weights: [10.0, 2.0, 3.0, 4.0]
    #
    # sampling params
    sampling_params:
        map:
            initial_step_size:  5.0e-4
            extreme_grad: 1.0e-5
            history_weight: 0.99
            selection_probas: [0.2, 0.8] # (p_mtm, p_pmala)
            k_mtm: 20
            is_stochastic: false
            compute_correction_term: false
        mcmc:
            initial_step_size:  3.0e-5
            extreme_grad: 1.0e-5
            history_weight: 0.99
            selection_probas: [0.5, 0.5] # (p_mtm, p_pmala)
            k_mtm: 10
            is_stochastic: true
            compute_correction_term: false
    #
    # run params
    run_params:
        map:
            N_MCMC: 1
            T_MC: 500 #2000
            T_BI: 20 # 100
            batch_size: 20
            freq_save: 1
            start_from: null
        mcmc:
            N_MCMC: 1
            T_MC: 500 #10_000
            T_BI: 15 #2_500
            plot_1D_chains: true
            plot_2D_chains: true
            plot_ESS: true
            plot_comparisons_yspace: true
            batch_size: 10
            freq_save: 1
            start_from: "MAP"
            regu_spatial_N0: !!float inf # sets to infinite
            regu_spatial_scale: 1.0
            regu_spatial_vmin: 1.0e-8
            regu_spatial_vmax: 1.0e+8
            list_CI: [68, 90, 95, 99]



Sampling
--------

With nn-based approximation of forward model:

.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn.py nn_N64_fixed_angle.yaml


Minimum mean square error (MMSE) estimator, i.e., mean of the iterates of the Markov chain (the average is evaluated in the scaled space):


.. image:: img/toy_case_astro/MMSE_mixing_0.png
   :width: 35%
.. image:: img/toy_case_astro/MMSE_mixing_1.png
   :width: 35%
.. image:: img/toy_case_astro/MMSE_mixing_2.png
   :width: 35%
.. image:: img/toy_case_astro/MMSE_mixing_3.png
   :width: 35%

|

Size of the 95% credibility intervals (written as a multiplicative factor):


.. image:: img/toy_case_astro/95_CI_size_mixing_0.png
   :width: 40%
.. image:: img/toy_case_astro/95_CI_size_mixing_1.png
   :width: 40%
.. image:: img/toy_case_astro/95_CI_size_mixing_2.png
   :width: 40%
.. image:: img/toy_case_astro/95_CI_size_mixing_3.png
   :width: 40%

|

As detailed in :cite:t:`paludEfficientSamplingNon2023`, both the MMSE and the credibility intervals are physically meaningful.


The ``data/toycase`` folder contains other resolutions with other ``.yaml`` input files.
For instance:

.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn.py nn_N10_fixed_angle.yaml


.. code:: bash

    poetry run python beetroots/simulations/astro/toy_case/toy_case_nn.py nn_N10_fixed_AV.yaml
