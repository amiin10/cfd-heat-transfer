"""
Transient one-dimensional conduction in a plane wall with convection.
=====================================================================

Physical problem (Incropera, Ch. 5): a plane wall of half-thickness L,
initially at uniform T_i, is suddenly immersed in a fluid at T_inf with
convection coefficient h.  By symmetry only 0 <= x <= L is solved:

    dT/dt = alpha d2T/dx2
    dT/dx = 0                      at x = 0   (symmetry plane)
    -k dT/dx = h (T - T_inf)       at x = L   (convective surface)

Three finite-difference schemes are implemented and compared against the
exact separation-of-variables solution:

    1. FTCS  (explicit)        - O(dt, dx^2),  stable only for Fo <= 1/2
    2. BTCS  (implicit)        - O(dt, dx^2),  unconditionally stable
    3. Crank-Nicolson          - O(dt^2, dx^2), unconditionally stable

Exact solution:
    theta = (T - T_inf)/(T_i - T_inf) = sum_n C_n exp(-zeta_n^2 Fo)
                                                cos(zeta_n x/L)
    with  zeta_n tan(zeta_n) = Bi   and
          C_n = 4 sin(zeta_n) / (2 zeta_n + sin(2 zeta_n))

The eigenvalues are found with Brent's method bracketed inside each
branch of the tangent function.

The script also demonstrates the FTCS stability limit by deliberately
running at Fo = 0.6 and showing the solution blow up.

Author: <your name>
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


# --------------------------------------------------------------------------
# Exact solution
# --------------------------------------------------------------------------

def eigenvalues(Bi: float, n_roots: int = 60) -> np.ndarray:
    """Roots of  zeta tan(zeta) = Bi , one per branch of tan."""
    roots = []
    for n in range(n_roots):
        lo = n * np.pi + 1e-10
        hi = (n + 0.5) * np.pi - 1e-10
        f = lambda z: z * np.sin(z) - Bi * np.cos(z)   # rearranged, no poles
        roots.append(brentq(f, lo, hi, xtol=1e-14, rtol=1e-15))
    return np.array(roots)


def exact_theta(xi: np.ndarray, Fo: float, Bi: float,
                zeta: np.ndarray | None = None) -> np.ndarray:
    """Dimensionless temperature; xi = x/L, Fo = alpha t / L^2."""
    if zeta is None:
        zeta = eigenvalues(Bi)
    C = 4.0 * np.sin(zeta) / (2.0 * zeta + np.sin(2.0 * zeta))
    xi = np.atleast_1d(xi)
    terms = C[None, :] * np.exp(-zeta[None, :] ** 2 * Fo) * \
        np.cos(zeta[None, :] * xi[:, None])
    return terms.sum(axis=1)


# --------------------------------------------------------------------------
# Numerical schemes
# --------------------------------------------------------------------------

def _operator(nx: int, dx: float, Bi_dx: float):
    """
    Build the discrete Laplacian A (units 1/dx^2) for dtheta/dt = alpha A theta
    including the symmetry BC at node 0 and the convective BC at node nx-1
    via the ghost-node (energy balance) formulation.

    Bi_dx = h dx / k  is the cell Biot number at the surface node.
    """
    main = np.full(nx, -2.0)
    lower = np.ones(nx - 1)
    upper = np.ones(nx - 1)

    # Node 0: symmetry -> ghost node T_{-1} = T_1  =>  2(T_1 - T_0)
    upper[0] = 2.0

    # Node nx-1: convective surface, ghost T_{n} = T_{n-2} - 2 Bi_dx T_{n-1}
    main[-1] = -2.0 * (1.0 + Bi_dx)
    lower[-1] = 2.0

    A = sp.diags([lower, main, upper], [-1, 0, 1], format="csr") / dx ** 2
    return A


def solve_transient(scheme: str, nx: int, Fo_target: float, Bi: float,
                    dFo: float):
    """
    March theta from 1.0 to Fourier number Fo_target.

    Working in dimensionless form: xi in [0,1], time = Fo, alpha = 1.
    dFo is the time step in Fourier-number units; the mesh Fourier number
    is  Fo_mesh = dFo / dxi^2  (the FTCS stability parameter).
    """
    xi = np.linspace(0.0, 1.0, nx)
    dxi = xi[1] - xi[0]
    A = _operator(nx, dxi, Bi * dxi)          # Bi_dx = Bi * dxi  (since Bi=hL/k)
    I = sp.identity(nx, format="csr")

    theta = np.ones(nx)
    nsteps = max(1, int(round(Fo_target / dFo)))
    dt = Fo_target / nsteps

    if scheme == "explicit":
        M = (I + dt * A).tocsr()
        for _ in range(nsteps):
            theta = M @ theta
    elif scheme == "implicit":
        lu = spla.splu((I - dt * A).tocsc())
        for _ in range(nsteps):
            theta = lu.solve(theta)
    elif scheme == "crank-nicolson":
        lhs = spla.splu((I - 0.5 * dt * A).tocsc())
        rhs = (I + 0.5 * dt * A).tocsr()
        for _ in range(nsteps):
            theta = lhs.solve(rhs @ theta)
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return xi, theta


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Bi = 1.0            # h L / k
    Fo_end = 0.2
    nx = 41
    dxi = 1.0 / (nx - 1)

    zeta = eigenvalues(Bi)
    print("=" * 72)
    print("1D TRANSIENT CONDUCTION, PLANE WALL WITH CONVECTION")
    print(f"Bi = {Bi},  Fo = {Fo_end},  {nx} nodes")
    print("=" * 72)
    print(f"first four eigenvalues: {np.round(zeta[:4], 6)}")
    print("(reference for Bi=1: 0.860334, 3.425618, 6.437298, 9.529334)\n")

    xi_e = np.linspace(0, 1, nx)
    theta_e = exact_theta(xi_e, Fo_end, Bi, zeta)

    # ---- accuracy comparison at a mesh Fourier number that is stable ----
    Fo_mesh = 0.4
    dFo = Fo_mesh * dxi ** 2
    print(f"{'scheme':<18}{'Fo_mesh':>10}{'max error':>14}{'RMS error':>14}")
    print("-" * 60)
    results = {}
    for scheme in ("explicit", "implicit", "crank-nicolson"):
        xi, th = solve_transient(scheme, nx, Fo_end, Bi, dFo)
        err = np.abs(th - theta_e)
        results[scheme] = th
        print(f"{scheme:<18}{Fo_mesh:>10.2f}{err.max():>14.3e}"
              f"{np.sqrt(np.mean(err**2)):>14.3e}")

    # ---- temporal order of accuracy of Crank-Nicolson vs implicit ----
    # The error is measured against a reference computed on the SAME mesh with
    # a very small time step.  This cancels the spatial discretisation error
    # and isolates the temporal error, which is what we want to rate.
    print("\nTemporal order of accuracy (same mesh, error vs. dt->0 reference):")
    print(f"{'dFo':>12}{'BTCS err':>14}{'order':>8}{'CN err':>16}{'order':>8}")
    print("-" * 60)
    nx_f = 81
    _, th_ref = solve_transient("crank-nicolson", nx_f, Fo_end, Bi, 1e-7)
    prev = {}
    for dFo_t in (4e-3, 2e-3, 1e-3, 5e-4):
        line = f"{dFo_t:>12.1e}"
        for s in ("implicit", "crank-nicolson"):
            _, th = solve_transient(s, nx_f, Fo_end, Bi, dFo_t)
            e = np.abs(th - th_ref).max()
            o = "" if s not in prev else \
                f"{np.log(prev[s][0]/e)/np.log(prev[s][1]/dFo_t):8.2f}"
            line += f"{e:>14.3e}{o:>8}" if s == "implicit" else f"{e:>16.3e}{o:>8}"
            prev[s] = (e, dFo_t)
        print(line)
    print("(expected: BTCS -> 1st order in dt,  Crank-Nicolson -> 2nd order)")

    # ---- explicit-scheme stability demonstration ----
    print("\nFTCS stability check (theory: unstable for Fo_mesh > 0.5):")
    for Fo_mesh_test in (0.25, 0.50, 0.60):
        _, th = solve_transient("explicit", 21, 0.05, Bi,
                                Fo_mesh_test * (1 / 20) ** 2)
        peak = np.abs(th).max()
        status = "STABLE" if peak < 5 else "DIVERGED"
        print(f"  Fo_mesh = {Fo_mesh_test:.2f} -> max|theta| = {peak:.3e}  {status}")

    # ---------------- figures ----------------
    figdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))

    xi_fine = np.linspace(0, 1, 200)
    for Fo in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        ax[0].plot(xi_fine, exact_theta(xi_fine, Fo, Bi, zeta), lw=1.6,
                   label=f"Fo = {Fo}")
    ax[0].set_xlabel(r"$x/L$"); ax[0].set_ylabel(r"$\theta$")
    ax[0].set_title(f"Exact transient profiles (Bi = {Bi})")
    ax[0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax[0].grid(alpha=0.3)

    ax[1].plot(xi_e, theta_e, "k-", lw=2.5, label="exact series")
    styles = {"explicit": "o", "implicit": "s", "crank-nicolson": "^"}
    for s, th in results.items():
        ax[1].plot(xi_e[::4], th[::4], styles[s], ms=6, mfc="none", label=s)
    ax[1].set_xlabel(r"$x/L$"); ax[1].set_ylabel(r"$\theta$")
    ax[1].set_title(f"Scheme comparison at Fo = {Fo_end}")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    for Fo_mesh_test, c in ((0.25, "tab:green"), (0.60, "tab:red")):
        _, th = solve_transient("explicit", 21, 0.05, Bi,
                                Fo_mesh_test * (1 / 20) ** 2)
        ax[2].plot(np.linspace(0, 1, 21), th, "-o", ms=4, color=c,
                   label=f"$Fo_{{mesh}}$ = {Fo_mesh_test}")
    ax[2].set_yscale("symlog", linthresh=1.0)
    ax[2].set_xlabel(r"$x/L$"); ax[2].set_ylabel(r"$\theta$ (symlog)")
    ax[2].set_title("FTCS stability limit (symlog scale)")
    ax[2].legend(); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "conduction_1d_transient.png"), dpi=140)
    print("\nFigure written to figures/conduction_1d_transient.png")


if __name__ == "__main__":
    main()
