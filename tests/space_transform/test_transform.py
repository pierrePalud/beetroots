import numpy as np

from beetroots.space_transform.transform import MyScaler

N_grid = 500_000
D = 3
D_no_kappa = 2
Theta_grid_lin = np.zeros((N_grid, D))
Theta_grid_lin[:, D - D_no_kappa :] = 10 ** np.random.normal(
    0, 1, size=(N_grid, D_no_kappa)
)
Theta_grid_lin[:, : D - D_no_kappa] = 1

# with normal distributed in scaled, the standard scaler should have no effect
N_test = 10
normal_samples = np.random.normal(0, 1, size=(N_test, D_no_kappa))

Theta_test_lin = np.zeros((N_test, D))
Theta_test_lin[:, D - D_no_kappa :] = 10**normal_samples
Theta_test_lin[:, : D - D_no_kappa] = 1

Theta_test_scaled = np.zeros((N_test, D))
Theta_test_scaled[:, D - D_no_kappa :] = normal_samples
Theta_test_scaled[:, : D - D_no_kappa] = 0

mean_ = np.mean(Theta_test_scaled[:, D - D_no_kappa :], axis=0)
std_ = np.std(Theta_test_scaled[:, D - D_no_kappa :], axis=0)
list_is_log = [True for _ in range(D)]

Theta_test_scaled[:, D - D_no_kappa :] = (
    Theta_test_scaled[:, D - D_no_kappa :] - mean_[None, :]
) / std_[None, :]


def test_from_scaled_to_lin():
    scaler = MyScaler(
        Theta_grid_lin, D_no_kappa, mean_=mean_, std_=std_, list_is_log=list_is_log
    )
    Theta_lin = scaler.from_scaled_to_lin(Theta_test_scaled)
    assert Theta_lin.shape == Theta_test_scaled.shape
    assert np.allclose(
        Theta_lin[:, :-D_no_kappa], Theta_test_lin[:, :-D_no_kappa]
    )  # kappa
    assert np.allclose(
        Theta_lin[:, -D_no_kappa:],
        Theta_test_lin[:, -D_no_kappa:],
        rtol=2e-2,
        atol=1e-1,
    )  # theta


def test_from_lin_to_scaled():
    scaler = MyScaler(
        Theta_grid_lin, D_no_kappa, mean_=mean_, std_=std_, list_is_log=list_is_log
    )
    Theta_scaled = scaler.from_lin_to_scaled(Theta_test_lin)
    assert Theta_scaled.shape == Theta_test_scaled.shape
    assert np.allclose(
        Theta_scaled[:, :-D_no_kappa], Theta_test_scaled[:, :-D_no_kappa]
    )  # kappa
    assert np.allclose(
        Theta_scaled[:, -D_no_kappa:],
        Theta_test_scaled[:, -D_no_kappa:],
        rtol=2e-2,
        atol=1e-1,
    )  # theta
