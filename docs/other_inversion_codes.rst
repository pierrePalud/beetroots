Why we needed a new inversion code in the first place
=====================================================

TODO: finish

**Note**: the following list does not try to be exhaustive.
However, it is complete enough to provide a good overview of the existing codes in the interstellar medium community.

New observation instruments produce large maps, such as the James Webb spatial telescope (JWST).
Important to design new tools to analyze in detail the costly and complex maps they produce, and to compare them with state-of-the-art physical models of the interstellar medium.


Interstellar medium studies involve both sampling-based and optimization-based inference methods.

For optimization approaches:

* Grid search estimations are dominant, especially in physical parameters estimations from maps of integrated intensities, e.g., in
.. :cite:t:`shefferPDRMODELMAPPING2011`,
.. :cite:t:`shefferPDRMODELMAPPING2013`,
.. :cite:t:`joblinStructurePhotodissociationFronts2018`
.. and :cite:t:`leeRadiativeMechanicalFeedback2019`.

.. * Gradient descent-based optimization procedures are a minority, see e.g., :cite:t:`wuConstrainingPhysicalConditions2018`



Riemannian integration with a grid of pdf evaluations

**Bayesian:**

* random walk Metropolis-Hastings (RWMH)

* ``emcee``

.. * MultiNest :cite:t:`ferozBayesianModellingClusters2009``

* UltraNest

* Sequential Monte Carlo

* Hierarchical models

**Bayesian codes:**

RWMH:

.. * CosmoMC :cite:t:`lewisCosmologicalParametersCMB2002`

.. * UCLCHEMCMCMCMC :cite:t:`keilUCLCHEMCMCMCMCInference2022`

.. * GalMC :cite:t:`acquavivaSpectralEnergyDistribution2011`

.. * BEAGLE :cite:t:`chevallardModellingInterpretingSpectral2016`

.. * BAMBI :cite:t:`graffBAMBIBlindAccelerated2012`

.. * MULTIGRIS :cite:t:`lebouteillerTopologicalModelsInfer2022`

.. * CIGALE :cite:t:`nollAnalysisGalaxySpectral2009`

.. * HerBIE :cite:t:`gallianoDustSpectralEnergy2018`

.. * Hii-Chi-Mistry :cite:t:`perez-monteroDerivingModelbasedTeconsistent2014`

.. * IZI :cite:t:`blancIZIInferringGas2015``

.. * BOND :cite:t:`valeasariBONDBayesianOxygen2016`

.. * NebulaBayes :cite:t:`thomasInterrogatingSeyfertsNebulaBayes2018`


**Codes handling high dimensional maps (for line fitting):**

.. * ROHSA :cite:t:`marchalROHSARegularizedOptimization2019`

.. * CubeFit :cite:t:`paumardRegularized3DSpectroscopy2022`
