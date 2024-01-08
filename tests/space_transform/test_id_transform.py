import numpy as np

from beetroots.space_transform.id_transform import IdScaler


def test_from_scaled_to_lin():
    scaler = IdScaler()
    Theta_scaled = np.random.rand(10, 4)
    Theta_lin = scaler.from_scaled_to_lin(Theta_scaled)
    assert Theta_scaled.shape == Theta_lin.shape
    assert np.allclose(Theta_scaled, Theta_lin)


def test_from_lin_to_scaled():
    scaler = IdScaler()
    Theta_lin = np.random.rand(10, 4)
    Theta_scaled = scaler.from_scaled_to_lin(Theta_lin)
    assert Theta_scaled.shape == Theta_lin.shape
    assert np.allclose(Theta_scaled, Theta_lin)
