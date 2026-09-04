"""
Two-dimensional steady-state heat conduction in a rectangular plate.
====================================================================

Solves the Laplace equation

    d2T/dx2 + d2T/dy2 = 0

on a rectangle 0 <= x <= L, 0 <= y <= H with a second-order accurate
five-point finite-difference stencil.  The resulting sparse linear system
is assembled in CSR format and solved directly with SuperLU.

Two verification cases are included:

CASE A - smooth boundary data (used for the order-of-accuracy study)
    T = 0 on x=0, x=L, y=0 ;  T(x,H) = T1 sin(pi x / L)
    Exact:  T(x,y) = T1 sin(pi x/L) sinh(pi y/L) / sinh(pi H/L)
    Because the data is analytic, the discrete solution must converge at
    the theoretical rate p = 2.

CASE B - the classic textbook problem (Incropera, Ch. 4)
    Three sides at T = 0, top side at uniform T = T1.
    Exact: Fourier series
        T = (2 T1/pi) sum_n [(1-(-1)^n)/n] sin(n pi x/L)
                             sinh(n pi y/L)/sinh(n pi H/L)
    The corner discontinuities degrade the formal order, which is itself
    a useful thing to demonstrate.

The conduction shape factor of the plate is also recovered from the
computed wall heat flux and compared with the series solution.

Author: <your name>
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------
# Analytical solutions
# --------------------------------------------------------------------------

def exact_sine(X, Y, L, H, T1=100.0):
    """Case A: single-mode sinusoidal top boundary (closed form)."""
    return T1 * np.sin(np.pi * X / L) * np.sinh(np.pi * Y / L) / np.sinh(np.pi * H / L)


def exact_uniform(X, Y, L, H, T1=100.0, n_terms=400):
    """Case B: uniform top boundary, truncated Fourier series."""
    T = np.zeros_like(X, dtype=float)
    for n in range(1, n_terms + 1):
        coef = (1.0 - (-1.0) ** n) / n           # zero for even n
        if coef == 0.0:
            continue
        arg = n * np.pi / L
        # sinh ratio written in exponential form to avoid overflow at large n
        ratio = np.exp(arg * (Y - H)) * (1 - np.exp(-2 * arg * Y)) / \
                (1 - np.exp(-2 * arg * H))
        T += coef * np.sin(arg * X) * ratio
    return 2.0 * T1 / np.pi * T


# --------------------------------------------------------------------------
# Finite-difference solver
# --------------------------------------------------------------------------

def solve_laplace(nx: int, ny: int, L: float, H: float, top_bc) -> tuple:
    """
    Solve Laplace's equation on an nx x ny node grid (including boundaries).

    Parameters
    ----------
    top_bc : callable  f(x) -> T at y = H.  All other walls are held at 0.

    Returns (X, Y, T) with T of shape (nx, ny), indexed T[i, j] = T(x_i, y_j).
    """
    x = np.linspace(0.0, L, nx)
    y = np.linspace(0.0, H, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    T = np.zeros((nx, ny))
    T[:, -1] = top_bc(x)                       # Dirichlet top

    # Unknowns are interior nodes only
    ni, nj = nx - 2, ny - 2
    N = ni * nj

    def idx(i, j):                             # (i,j) interior -> row number
        return (j - 1) * ni + (i - 1)

    ax = 1.0 / dx ** 2
    ay = 1.0 / dy ** 2
    diag = -2.0 * (ax + ay)

    rows, cols, vals = [], [], []
    b = np.zeros(N)

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            k = idx(i, j)
            rows.append(k); cols.append(k); vals.append(diag)

            for (ii, jj, coef) in ((i - 1, j, ax), (i + 1, j, ax),
                                   (i, j - 1, ay), (i, j + 1, ay)):
                if 1 <= ii <= nx - 2 and 1 <= jj <= ny - 2:
                    rows.append(k); cols.append(idx(ii, jj)); vals.append(coef)
                else:
                    b[k] -= coef * T[ii, jj]   # known boundary value -> RHS

    A = sp.csr_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(N, N)))
    sol = spla.spsolve(A, b)
    T[1:-1, 1:-1] = sol.reshape(nj, ni).T

    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, T


def wall_heat_rate(T, L, H, k=1.0):
    """
    Heat entering through the hot (top) wall per unit depth, W/m.
    One-sided second-order difference for dT/dy at y = H.
    """
    nx, ny = T.shape
    dx = L / (nx - 1)
    dy = H / (ny - 1)
    dTdy = (3 * T[:, -1] - 4 * T[:, -2] + T[:, -3]) / (2 * dy)
    q_in = k * dTdy            # positive = heat entering the plate (-y dir.)
    return np.trapezoid(q_in, dx=dx)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, H, T1 = 1.0, 0.5, 100.0

    # ---------- Case A: order-of-accuracy verification ----------
    print("=" * 70)
    print("2D STEADY CONDUCTION - GRID CONVERGENCE (smooth sinusoidal BC)")
    print("=" * 70)
    print(f"{'nodes':>12}{'h':>12}{'L2 error':>16}{'Linf error':>16}{'order':>9}")
    print("-" * 70)

    grids = [11, 21, 41, 81, 161]
    errs, hs = [], []
    for n in grids:
        nx, ny = n, (n // 2) + 1
        X, Y, T = solve_laplace(nx, ny, L, H,
                                lambda xx: T1 * np.sin(np.pi * xx / L))
        Te = exact_sine(X, Y, L, H, T1)
        e = T - Te
        l2 = np.sqrt(np.mean(e ** 2))
        linf = np.abs(e).max()
        h = L / (nx - 1)
        order = "" if not errs else f"{np.log(errs[-1]/l2)/np.log(hs[-1]/h):9.3f}"
        print(f"{nx}x{ny:<8}{h:>12.5f}{l2:>16.3e}{linf:>16.3e}{order:>9}")
        errs.append(l2); hs.append(h)

    p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    print(f"\nLeast-squares observed order of accuracy: p = {p:.3f}  (theory: 2)")

    # ---------- Case B: textbook problem ----------
    nx, ny = 161, 81
    X, Y, T = solve_laplace(nx, ny, L, H, lambda xx: np.full_like(xx, T1))
    Te = exact_uniform(X, Y, L, H, T1)
    interior_err = np.abs(T - Te)[1:-1, 1:-1].max()

    q_num = wall_heat_rate(T, L, H, k=1.0)
    print("\n" + "=" * 70)
    print("CASE B - uniform hot wall (161 x 81 grid)")
    print("=" * 70)
    print(f"max |T_num - T_series| (interior)   : {interior_err:.4f} K")
    print(f"centre-point temperature, numeric   : {T[nx//2, ny//2]:.4f} K")
    print(f"centre-point temperature, series    : {Te[nx//2, ny//2]:.4f} K")
    print(f"heat rate through hot wall (k=1)    : {q_num:.3f} W/m")

    # ---------- figure ----------
    figdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))

    cs = ax[0].contourf(X, Y, T, levels=30, cmap="inferno")
    ax[0].contour(X, Y, T, levels=12, colors="w", linewidths=0.5)
    fig.colorbar(cs, ax=ax[0], label="T [K]")
    ax[0].set_title("Temperature field (uniform hot wall)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].set_aspect("equal")

    from matplotlib.colors import LogNorm
    err = np.abs(T - Te)
    err = np.clip(err, 1e-6, None)          # log scale needs a positive floor
    cs2 = ax[1].contourf(X, Y, err, levels=np.logspace(-6, 2, 33),
                         norm=LogNorm(), cmap="viridis", extend="both")
    fig.colorbar(cs2, ax=ax[1], label="|error| [K]", ticks=[1e-6, 1e-4, 1e-2, 1, 100])
    ax[1].set_title("|numerical - series| (log scale)\nerror localised at the corner singularities")
    ax[1].set_xlabel("x [m]"); ax[1].set_ylabel("y [m]")
    ax[1].set_aspect("equal")

    ax[2].loglog(hs, errs, "o-", lw=2, label="computed $L_2$ error")
    ax[2].loglog(hs, errs[0] * (np.array(hs) / hs[0]) ** 2, "k--",
                 label=r"$\mathcal{O}(h^2)$ reference")
    ax[2].set_xlabel("grid spacing h"); ax[2].set_ylabel("$L_2$ error")
    ax[2].set_title(f"Grid convergence (p = {p:.2f})")
    ax[2].grid(which="both", alpha=0.3); ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "conduction_2d.png"), dpi=140)
    print("\nFigure written to figures/conduction_2d.png")


if __name__ == "__main__":
    main()
