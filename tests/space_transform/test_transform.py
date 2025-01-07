import numpy as np

from beetroots.space_transform.transform import MyScaler

N_grid = 500_000
D = 3

# grid without kappa
Theta_grid_lin = np.zeros((N_grid, D - 1))
Theta_grid_lin = np.exp(np.log(10) * np.random.normal(0, 1, size=(N_grid, D - 1)))

list_is_log = [True for _ in range(D)]

mean_ = np.mean(np.log10(Theta_grid_lin), axis=0)  # (D-1,)
std_ = np.std(np.log10(Theta_grid_lin), axis=0)  # (D-1,)
assert mean_.size == D - 1

N_test = 10
normal_samples = np.random.normal(0, 3.0, size=(N_test, D))

Theta_test_lin = np.exp(np.log(10) * normal_samples)

Theta_test_scaled = normal_samples

Theta_test_scaled[:, 0] *= np.log(10)  # divide by std = 1/log(10)

Theta_test_scaled[:, 1:] = (Theta_test_scaled[:, 1:] - mean_[None, :]) / std_[None, :]

scaler = MyScaler(mean_, std_, list_is_log)


def test_init():
    assert np.isclose(scaler.mean_[0], 0.0)
    assert np.isclose(scaler.std_[0], 1 / np.log(10))
    assert scaler.mean_.size == D
    assert scaler.std_.size == D


def test_from_scaled_to_lin():
    Theta_lin = scaler.from_scaled_to_lin(Theta_test_scaled)
    assert Theta_lin.shape == Theta_test_scaled.shape
    assert np.allclose(Theta_lin[:, 0], Theta_test_lin[:, 0])  # kappa
    assert np.allclose(Theta_lin[:, 1:], Theta_test_lin[:, 1:])  # theta


def test_from_lin_to_scaled():
    Theta_scaled = scaler.from_lin_to_scaled(Theta_test_lin)
    assert Theta_scaled.shape == Theta_test_scaled.shape
    assert np.allclose(Theta_scaled[:, 0], Theta_test_scaled[:, 0])  # kappa
    assert np.allclose(Theta_scaled[:, 1:], Theta_test_scaled[:, 1:])  # theta
