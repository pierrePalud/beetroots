# OMC1

## run simulation

To run :

```bash
poetry shell
```

```bash
poetry run python beetroots/simulations/astro/real_data/omc1_nn_direct_posterior.py input_params_45per_with_spatial_regu_lines1234.yaml
```

**Carefull: this multiplicative noise really requires the optimization of approx params.**

## Multiplicative noise of 45%

Calibration error :
$$
\\epsilon^{(m)} \\sim \\log \\mathcal{N}(-\\sigma_m^2 / 2 , , \\sigma^2_m)
$$
$\\sigma_m = \\log(1.1)$ pour une erreur moyenne de 10%

Error on model :
$$
\\epsilon^{(s)} \\sim \\log \\mathcal{N}(-\\sigma_s^2 / 2 , , \\sigma^2_s)
$$
with $\\sigma_s = \\log(1.44)$ pour une erreur inférieure à un facteur 3 avec une proba de 99% ($1.44 \\simeq 3^{\\frac{1}{3}}$ ).

Total multiplicative error:

$$
\\epsilon^{(t)} = \\epsilon^{(m)} \\epsilon^{(s)} \\sim \\log \\mathcal{N} \\left(

- \\frac{\\sigma_m^2 + \\sigma_s^2}{2} , ,
  \\sigma^2_m + \\sigma^2_s
  \\right)
  $$

Let $\\sigma_t = \\sqrt{\\sigma_m^2 + \\sigma_s^2} \\simeq 0.373 \\simeq \\log(1.452)$ .
