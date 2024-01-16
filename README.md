# beetroots

[![PyPI version](https://badge.fury.io/py/beetroots.svg)](https://badge.fury.io/py/beetroots)
[![Documentation Status](https://readthedocs.org/projects/beetroots/badge/?version=latest)](https://beetroots.readthedocs.io/en/latest/?badge=latest)

Beetroots (BayEsian infErence with spaTial Regularization of nOisy multi-line ObservaTion mapS) is a Python package that performs Bayesian inference of physical parameters from multispectral-structured cubes with a dedicated sampling algorithm.
Thanks to this sampling algorithm, `beetroots` provides maps of credibility intervals along with estimated maps.

The sampling algorithm is introduced in

> \[1\] P. Palud, P.-A. Thouvenin, P. Chainais, E. Bron, and F. Le Petit - **Efficient sampling of non log-concave posterior distributions with mixture of noises**, *IEEE Transactions on Signal Processing*, vol. 71, pp. 2491 -- 2501, 2023. [doi:10.1109/TSP.2023.3289728](https://doi.org/10.1109/TSP.2023.3289728)

Such inversions rely on a forward model that is assumed to emulate accurately the physics of the observed environment.
In parallel of the inversion, `beetroots` tests this hypothesis to evaluate the validity of the inference results.
The testing method is described in (in French)

> \[2\] P. Palud, P. Chainais, F. Le Petit, P.-A. Thouvenin and E. Bron - **Problèmes inverses et test bayésien d'adéquation du modèle**, *GRETSI - Groupe de Recherche en Traitement du Signal et des Images* in *29e Colloque sur le traitement du signal et des images*, Grenoble, pp. 705 -- 708, 2023.

This package was applied e.g., to infer physical conditions in different regions of the interstellar medium in

> \[3\] P. Palud, P.-A. Thouvenin, P. Chainais, E. Bron, F. Le Petit and ORION-B consortium - **Bayesian inversion of large interstellar medium observation maps**, in prep

It was also exploited to assert and compare the relevance of tracers and combination of tracers to constrain physical conditions in

> \[4\] L. Einig, P. Palud, A. Roueff, P.-A. Thouvenin, P. Chainais, E. Bron, F. Le Petit, J. Pety, J. Chanussot and ORION-B consortium -  **Entropy-based selection of most informative observables for inference from interstellar medium observations**, in prep

## Complex forward models

In numerous real-life applications, the forward model is a complex numerical simulation.
In the astrophysics applications presented in the documentation, the numerical simulation is replaced with a neural network-based approximation of the forward model for

- faster evaluation
- ability to evaluate derivatives

The package used to derive this approximation is `nnbma` (Neural Network-Based Model Approximation).
The GitHub repository can be found [here](https://github.com/einigl/ism-model-nn-approximation), the package [here](https://pypi.org/project/nnbma/) and the corresponding documentation [here](https://ism-model-nn-approximation.readthedocs.io/en/latest/?badge=latest).
The paper presenting this package is

> \[5\] P. Palud, L. Einig, F. Le Petit, E. Bron, P. Chainais, J. Chanussot, J. Pety, P.-A. Thouvenin and ORION-B consortium - **Neural network-based emulation of interstellar medium models**, *Astronomy & Astrophysics*, 2023, 678, pp.A198. [doi:10.1051/0004-6361/202347074](https://doi.org/10.1051/0004-6361/202347074)

## Installation and testing

To prepare and perform an inversion, we recommend installing the package.
The package can be installed with `pip`:

```shell
pip install beetroots
```

or by cloning the repo.
To clone, install and test the package, run:

```shell
git clone git@github.com:pierrePalud/beetroots.git
cd beetroots
poetry install # or poetry install -E notebook -E docs for extra dependencies
poetry shell
pytest
```
