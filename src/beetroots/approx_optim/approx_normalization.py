import abc

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import simpson
from scipy.optimize import least_squares

# Assuming P_lambda is defined elsewhere in your file as before
P_lambda = np.poly1d(np.array([-6.0, 15.0, -10.0, 0.0, 0.0, 1.0]))


class ApproxNormalization(abc.ABC):
    r"""Computes normalization constant of the mixing model approximation of the likelihood distribution."""

    FIT_PARAM_COLUMNS = [
        "fit_Z_poly_c0",
        "fit_Z_poly_c1",
        "fit_Z_poly_c2",
        "fit_Z_poly_c3",
        "fit_Z_poly_c4",
        "fit_Z_B",
        "fit_Z_d",
        "fit_Z_omega",
        "fit_Z_k",
        "fit_Z_phi",
    ]
    FIT_POLY_DEGREE = 4

    def compute_fit_normalization(
        self,
        df_best,
        sigma_a,
        sigma_m,
        omega,
        n_ftheta,
        n_y,
        plot_Z_fit,
        path_image_folder,
        path_output_sim,
    ):
        def wrapper_fit_exp_poly(df_best_row):
            return self.fit_exp_poly_chirped(
                df_best_row,
                sigma_a,
                sigma_m,
                omega,
                n_ftheta,
                n_y,
                plot_Z_fit,
                path_image_folder,
            )

        df_best[self.FIT_PARAM_COLUMNS] = df_best.apply(
            wrapper_fit_exp_poly, axis=1, result_type="expand"
        )

        df_best.to_csv(f"{path_output_sim}/best_params.csv", index=False)
        return df_best

    def compute_normalization_elementwise(
        self, df_best_row, sigma_a, sigma_m, omega, n_ftheta, n_y
    ):
        transition_center = df_best_row["a0_best"] * np.log(10)
        transition_radius = df_best_row["a1_best"] * np.log(10)

        sigma_a = sigma_a[df_best_row["n"], df_best_row["ell"]]
        sigma_m = sigma_m[df_best_row["n"], df_best_row["ell"]]
        omega = omega[df_best_row["n"], df_best_row["ell"]]

        log_fm1 = transition_center - transition_radius
        log_fp1 = transition_center + transition_radius

        # Generate grid of f_theta values (for interpolation afterwards)
        f_theta = np.linspace(start=np.exp(log_fm1), stop=np.exp(log_fp1), num=n_ftheta)
        log_f_theta = np.log(f_theta)

        # Compute the parameters of the two pure likelihood approximations
        s_a = np.sqrt(sigma_a**2 + f_theta**2 * (np.exp(sigma_m**2) - 1))
        m_m = -0.5 * (
            sigma_m**2 + np.log(1 + sigma_a**2 / (f_theta**2 * np.exp(sigma_m**2)))
        )
        s_m = np.sqrt(-2 * m_m)

        # Generate grid of y values for integration for each f_theta value
        start = np.maximum(omega, f_theta - 10 * s_a)
        stop_peak = f_theta + 5 * s_a
        stop_tail = f_theta + 100 * s_a
        y_peak = np.linspace(
            start, stop_peak, n_y // 2, axis=-1
        )  # Lots of points around the peak
        y_tail = np.geomspace(
            stop_peak, stop_tail, n_y // 2, axis=-1
        )  # Fewer points in the tails but still far for lognormal heavy tails
        y = np.concatenate([y_peak, y_tail], axis=-1)

        # Instantiate the additive (a) and multiplicative (m) likelihood approximations pdf arrays
        pi_a = np.zeros_like(y)
        pi_m = np.zeros_like(y)

        # Compute the interpolation weights for each f_theta value
        lambda_ = np.where(
            log_f_theta <= log_fm1,
            1,
            np.where(
                log_f_theta >= log_fp1,
                0,
                P_lambda((log_f_theta - log_fm1) / (log_fp1 - log_fm1)),
            ),
        )

        # Non-censored part
        pi_a[y >= omega] = stats.norm.pdf(
            y[y >= omega],
            loc=f_theta[np.where(y >= omega)[0]],
            scale=s_a[np.where(y >= omega)[0]],
        )
        pi_m[y >= omega] = stats.lognorm.pdf(
            y[y >= omega],
            s=s_m[np.where(y >= omega)[0]],
            scale=f_theta[np.where(y >= omega)[0]]
            * np.exp(m_m[np.where(y >= omega)[0]]),
        )
        pi_interp_continuous = (
            pi_a ** lambda_[:, np.newaxis] * pi_m ** (1 - lambda_)[:, np.newaxis]
        )  # Interpolated PDF

        # Integration over the non-censored part
        Z_interp = simpson(pi_interp_continuous, x=y, axis=1)

        # adding the censored mass at omega
        Z_interp += stats.norm.cdf(
            omega, loc=f_theta, scale=s_a
        ) ** lambda_ * stats.lognorm.cdf(omega, s=s_m, scale=f_theta * np.exp(m_m)) ** (
            1 - lambda_
        )

        return (
            f_theta,
            Z_interp,
        )

    def exp_poly_chirped_peak(self, f_theta, params, degree):
        """Exponential polynomial envelope (degree 4) plus damped chirped oscillations."""
        lb = f_theta.min()
        ub = f_theta.max()
        z = (f_theta - lb) / (ub - lb)
        z = np.clip(z, 1e-12, 1.0 - 1.0e-12)

        poly_params = params[: degree + 1]
        B, d, omega_chirp, k, phi = params[degree + 1 :]

        poly_eval = np.polyval(poly_params, z)
        base_peak = np.exp(poly_eval)
        oscillations = B * np.exp(-d * z) * np.sin(omega_chirp * z + k * (z**2) + phi)
        window = z * (1.0 - z)

        return (base_peak + oscillations) * window

    def fit_exp_poly_chirped(
        self,
        df_best_row,
        sigma_a,
        sigma_m,
        omega,
        n_ftheta,
        n_y,
        plot_Z_fit,
        path_image_folder,
    ):
        f_theta, Z = self.compute_normalization_elementwise(
            df_best_row, sigma_a, sigma_m, omega, n_ftheta, n_y
        )

        y_peak = 1.0 - Z
        degree = self.FIT_POLY_DEGREE

        def residuals(params):
            return self.exp_poly_chirped_peak(f_theta, params, degree) - y_peak

        poly_guess = np.zeros(degree + 1)
        osc_guess = np.array([0.01, 5.0, 25.0, 0.0, 0.0])
        initial_guess = np.concatenate([poly_guess, osc_guess])

        lower_bounds = np.concatenate(
            [np.full(degree + 1, -np.inf), np.array([-1.0, 0.0, 0.0, -np.inf, -np.pi])]
        )
        upper_bounds = np.concatenate(
            [
                np.full(degree + 1, np.inf),
                np.array([1.0, np.inf, np.inf, np.inf, np.pi]),
            ]
        )

        result = least_squares(
            fun=residuals,
            x0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            max_nfev=15000,
        )
        popt = result.x

        if plot_Z_fit:
            fitted_peak = np.clip(
                self.exp_poly_chirped_peak(f_theta, popt, degree), 0.0, 1.0
            )
            fitted_dip = 1.0 - fitted_peak
            residuals = Z - fitted_dip

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )

            # Main plot
            ax1.plot(
                f_theta,
                Z,
                ".",
                color="lightgray",
                label=r"Sample $Z(f_{\ell}(\theta_k))$",
            )
            ax1.plot(
                f_theta,
                fitted_dip,
                "-",
                color="C9",
                linewidth=2,
                label="Exp-Poly (deg 4) + Chirped Oscillation Fit",
            )

            ax1.axvline(
                f_theta.min(),
                color="black",
                linestyle=":",
                linewidth=2,
                label="Transition\nboundaries",
            )
            ax1.axvline(f_theta.max(), color="black", linestyle=":", linewidth=2)

            ax1.set_ylabel(r"$Z(f_{\ell}(\theta))$")
            ax1.set_title(
                r"Exp-Poly (deg 4) + Chirped fit for normalization constant $Z$"
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Residuals plot
            ax2.plot(f_theta, residuals, color="C3", linewidth=1)
            ax2.axhline(0, color="black", linestyle="--", linewidth=1)
            ax2.set_xlabel(r"$f_{\ell}(\theta)$")
            ax2.set_ylabel("Residuals")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                f"{path_image_folder}/normalization_fit_exp_poly_chirp_n{df_best_row['n']}_ell{df_best_row['ell']}.png"
            )

        return tuple(popt.tolist())
