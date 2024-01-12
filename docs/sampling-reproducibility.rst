How to reproduce a sampling
===========================

During the sampling, the sampler uses a Random Generator object (from numpy) and regularly saves its state together with some indicators of the sampling (values of :math:`\Theta`, of objective function, etc.) in a ``.hdf5`` file. Therefore, to reproduce a sampling from a given point (value of :math:`\Theta` and state of the random generator), load these values from ``.hdf5`` file and then :

- run ``sampler.set_rng_state(state_bytes, inc_bytes)`` to use correct random generator state

- run the sampler with corresponding values for ``Theta_0``
