# Carina nebula

## run simulation

To run :

```bash
poetry shell
```

Then:

- With version 1.7 and spatial regularization:

```bash
poetry run python beetroots/simulations/astro/real_data/carina_nn.py input_params_1p7_with_spatial_regu.yaml
```

- With version 1.7 and no spatial regularization:

```bash
poetry run python beetroots/simulations/astro/real_data/carina_nn.py input_params_1p7_no_spatial_regu.yaml
```

- With version 1.5.4 (with PAH) and spatial regularization:

```bash
poetry run python beetroots/simulations/astro/real_data/carina_nn.py input_params_1p5p4_PAH_with_spatial_regu.yaml
```

- With version 1.5.4 (with PAH) and no spatial regularization:

```bash
poetry run python beetroots/simulations/astro/real_data/carina_nn.py input_params_1p5p4_PAH_no_spatial_regu.yaml
```

## Note on .dat files

This folder contains files of intensity (in I\[W m-2 sr-1\]) and error associated of the Carina cloud.

Note : Last modification : 30/07/2017
