import numpy as np

from beetroots.approx_optim.nn_bo import ApproxParamsOptimNNBO

if __name__ == "__main__":
    # for toycase, the noise variances are constant all over the map
    # and for all lines
    sigma_a = 1.38715e-10
    sigma_m = np.log(1.1)

    list_lines = [
        "co_v0_j4__v0_j3",
        "co_v0_j5__v0_j4",
        "co_v0_j6__v0_j5",
        "co_v0_j7__v0_j6",
        "co_v0_j8__v0_j7",
        "co_v0_j9__v0_j8",
        "co_v0_j10__v0_j9",
        "co_v0_j11__v0_j10",
        "co_v0_j12__v0_j11",
        "co_v0_j13__v0_j12",
    ]

    approx_optim = ApproxParamsOptimNNBO(
        list_lines,
        name="bo_nn_toycase",
        D=4,
        D_no_kappa=3,
        K=30,
        log10_f_grid_size=100,
        N_samples_y=200_000,  # 250_000
        max_workers=20,
        sigma_a=sigma_a,
        sigma_m=sigma_m,
    )
    approx_optim.main(
        forward_model_name="meudon_pdr_model_dense",
        angle=0.0,
        lower_bounds_lin=np.array([1e-1, 1e5, 1e0, 1e0]),
        upper_bounds_lin=np.array([1e1, 1e9, 1e5, 4e1]),
        n_iter=40,
    )
