"""
Blasius laminar boundary layer over a flat plate.
==================================================

Solves the Blasius similarity equation

Author: Seyed Mohammad Amin Hosseini
"""

from __future__ import annotations

import numpy as np


def _blasius_rhs(_eta: float, y: np.ndarray) -> np.ndarray:
    f, fp, fpp = y
    return np.array([fp, fpp, -0.5 * f * fpp])


def _rk4(rhs, eta: np.ndarray, y0: np.ndarray) -> np.ndarray:
    y = np.empty((eta.size, y0.size))
    y[0] = y0
    for i in range(eta.size - 1):
        h = eta[i + 1] - eta[i]
        k1 = rhs(eta[i], y[i])
        k2 = rhs(eta[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = rhs(eta[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = rhs(eta[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def solve_blasius(eta_max: float = 10.0, n: int = 2001,
                  tol: float = 1e-12, max_iter: int = 50):

    eta = np.linspace(0.0, eta_max, n)

    def residual(s: float) -> float:
        y = _rk4(_blasius_rhs, eta, np.array([0.0, 0.0, s]))
        return y[-1, 1] - 1.0          # f'(eta_max) - 1

    s = 0.3
    for _ in range(max_iter):
        r = residual(s)
        if abs(r) < tol:
            break
        ds = 1e-8 * max(1.0, abs(s))
        drds = (residual(s + ds) - r) / ds     # numerical Jacobian
        s -= r / drds
    else:
        raise RuntimeError("Shooting method failed to converge")

    y = _rk4(_blasius_rhs, eta, np.array([0.0, 0.0, s]))
    return eta, y, s


def solve_thermal(eta: np.ndarray, f: np.ndarray, Pr: float):

    from scipy.integrate import cumulative_trapezoid

    F = cumulative_trapezoid(f, eta, initial=0.0)      # int_0^eta f deta
    integrand = np.exp(-0.5 * Pr * F)
    I = cumulative_trapezoid(integrand, eta, initial=0.0)
    theta = I / I[-1]
    dtheta0 = integrand[0] / I[-1]                     # theta'(0)
    return theta, dtheta0


def boundary_layer_metrics(eta: np.ndarray, y: np.ndarray) -> dict:
    f, fp, fpp = y[:, 0], y[:, 1], y[:, 2]

    idx = np.argmax(fp >= 0.99)
    eta99 = np.interp(0.99, [fp[idx - 1], fp[idx]], [eta[idx - 1], eta[idx]])

    delta_star = eta[-1] - f[-1]

    theta_mom = np.trapezoid(fp * (1.0 - fp), eta)

    return {
        "f''(0)": fpp[0],
        "delta_99 * sqrt(Re_x)/x": eta99,
        "delta_star * sqrt(Re_x)/x": delta_star,
        "theta_mom * sqrt(Re_x)/x": theta_mom,
        "cf * sqrt(Re_x)": 2.0 * fpp[0],
        "shape_factor_H": delta_star / theta_mom,
    }


def velocity_profile(x: float, U_inf: float, nu: float,
                     eta: np.ndarray, y: np.ndarray):
    Re_x = U_inf * x / nu
    y_phys = eta * x / np.sqrt(Re_x)
    u = U_inf * y[:, 1]
    v = 0.5 * U_inf / np.sqrt(Re_x) * (eta * y[:, 1] - y[:, 0])
    return y_phys, u, v


def main() -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eta, y, s = solve_blasius()
    m = boundary_layer_metrics(eta, y)

    reference = {
        "f''(0)": 0.332057336,
        "delta_99 * sqrt(Re_x)/x": 4.910,
        "delta_star * sqrt(Re_x)/x": 1.7208,
        "theta_mom * sqrt(Re_x)/x": 0.6641,
        "cf * sqrt(Re_x)": 0.664,
        "shape_factor_H": 2.591,
    }

    print("=" * 68)
    print("BLASIUS FLAT-PLATE BOUNDARY LAYER  (RK4 + Newton shooting)")
    print("=" * 68)
    print(f"{'quantity':<28}{'computed':>13}{'reference':>13}{'rel.err':>13}")
    print("-" * 68)
    for k, v in m.items():
        ref = reference[k]
        print(f"{k:<28}{v:>13.6f}{ref:>13.6f}{abs(v-ref)/ref:>13.2e}")

    print("\nThermal boundary layer (isothermal wall):")
    print(f"{'Pr':>8}{'Nu_x/sqrt(Re_x)':>20}{'0.332 Pr^(1/3)':>18}{'rel.err':>12}")
    print("-" * 58)
    pr_err = {}
    for Pr in (0.7, 1.0, 7.0, 10.0):
        _, dtheta0 = solve_thermal(eta, y[:, 0], Pr)
        corr = 0.332 * Pr ** (1 / 3)
        pr_err[Pr] = (dtheta0, corr)
        print(f"{Pr:>8.1f}{dtheta0:>20.5f}{corr:>18.5f}"
              f"{abs(dtheta0-corr)/corr:>12.2e}")

    figdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(y[:, 1], eta, lw=2, label=r"$u/U_\infty = f'$")
    ax[0].plot(y[:, 2], eta, "--", lw=1.5, label=r"$f''$ (shear)")
    ax[0].axhline(m["delta_99 * sqrt(Re_x)/x"], color="k", ls=":",
                  lw=1, label=r"$\eta_{99}=4.91$")
    ax[0].set_xlabel("similarity velocity / shear")
    ax[0].set_ylabel(r"$\eta = y\sqrt{U_\infty/\nu x}$")
    ax[0].set_ylim(0, 8)
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_title("Blasius velocity profile")

    for Pr in (0.7, 1.0, 7.0, 10.0):
        theta, _ = solve_thermal(eta, y[:, 0], Pr)
        ax[1].plot(theta, eta, lw=1.8, label=f"Pr = {Pr}")
    ax[1].set_xlabel(r"$\theta = (T-T_w)/(T_\infty-T_w)$")
    ax[1].set_ylabel(r"$\eta$")
    ax[1].set_ylim(0, 8)
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[1].set_title("Thermal boundary layer vs. Prandtl number")

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "blasius.png"), dpi=140)
    print(f"\nFigure written to figures/blasius.png")


if __name__ == "__main__":
    main()
