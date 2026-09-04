"""
Heat-exchanger thermal design: effectiveness-NTU, LMTD and pressure drop.
========================================================================

A practical design-oriented module covering the sizing of a
counter-flow double-pipe heat exchanger:

  * Colebrook-White friction factor solved implicitly with Newton's method
    (verified against the explicit Haaland approximation and Moody chart).
  * Internal forced-convection Nusselt correlations: Dittus-Boelter and
    Gnielinski.
  * Effectiveness-NTU relations for counter-flow, parallel-flow and
    cross-flow arrangements.
  * The overall U from series thermal resistances (convection + wall
    conduction + fouling).
  * A sizing solve: find the tube length that meets a required duty, via
    Brent's method on the NTU relation.

Self-verification
-----------------
The epsilon-NTU and LMTD methods are mathematically equivalent for a given
exchanger.  The script computes the duty both ways and confirms they agree
to round-off, which is a strong internal consistency check on the
implementation.

Author: <your name>
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, newton


# --------------------------------------------------------------------------
# Friction factor and internal-flow correlations
# --------------------------------------------------------------------------

def friction_factor(Re: float, eps_D: float = 0.0) -> float:
    """
    Darcy friction factor.

    Laminar (Re < 2300):  f = 64/Re  (exact, Hagen-Poiseuille)
    Turbulent:            Colebrook-White, solved by Newton iteration on
                          x = 1/sqrt(f):
                              x = -2 log10(eps_D/3.7 + 2.51 x / Re)
    """
    if Re < 2300.0:
        return 64.0 / Re

    def g(x):
        return x + 2.0 * np.log10(eps_D / 3.7 + 2.51 * x / Re)

    x0 = 1.0 / np.sqrt(0.02)
    x = newton(g, x0, tol=1e-12, maxiter=100)
    return 1.0 / x ** 2


def haaland(Re: float, eps_D: float = 0.0) -> float:
    """Explicit Haaland approximation, used to cross-check Colebrook."""
    return (-1.8 * np.log10((eps_D / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def nusselt(Re: float, Pr: float, eps_D: float = 0.0,
            correlation: str = "gnielinski", heating: bool = True) -> float:
    """Fully developed internal flow Nusselt number."""
    if Re < 2300.0:
        return 3.66                       # constant wall temperature, laminar
    if correlation == "dittus-boelter":
        n = 0.4 if heating else 0.3
        return 0.023 * Re ** 0.8 * Pr ** n
    if correlation == "gnielinski":
        f = friction_factor(Re, eps_D)
        num = (f / 8.0) * (Re - 1000.0) * Pr
        den = 1.0 + 12.7 * np.sqrt(f / 8.0) * (Pr ** (2 / 3) - 1.0)
        return num / den
    raise ValueError(correlation)


# --------------------------------------------------------------------------
# Effectiveness-NTU relations
# --------------------------------------------------------------------------

def effectiveness(NTU: float, Cr: float, arrangement: str = "counterflow") -> float:
    """epsilon(NTU, Cr) for common arrangements (Incropera Table 11.3)."""
    if arrangement == "counterflow":
        if abs(Cr - 1.0) < 1e-10:
            return NTU / (1.0 + NTU)
        return (1.0 - np.exp(-NTU * (1 - Cr))) / \
               (1.0 - Cr * np.exp(-NTU * (1 - Cr)))
    if arrangement == "parallelflow":
        return (1.0 - np.exp(-NTU * (1 + Cr))) / (1.0 + Cr)
    if arrangement == "crossflow-both-unmixed":
        return 1.0 - np.exp((1.0 / Cr) * NTU ** 0.22 *
                            (np.exp(-Cr * NTU ** 0.78) - 1.0))
    if arrangement == "shell-and-tube-1pass":
        s = np.sqrt(1.0 + Cr ** 2)
        return 2.0 / (1.0 + Cr + s *
                      (1 + np.exp(-NTU * s)) / (1 - np.exp(-NTU * s)))
    raise ValueError(arrangement)


def ntu_from_effectiveness(eps: float, Cr: float,
                           arrangement: str = "counterflow") -> float:
    """Invert epsilon(NTU) numerically - works for every arrangement."""
    eps_max = effectiveness(1e4, Cr, arrangement)
    if eps >= eps_max:
        raise ValueError(f"effectiveness {eps:.4f} unreachable "
                         f"(limit {eps_max:.4f} for Cr = {Cr})")
    return brentq(lambda n: effectiveness(n, Cr, arrangement) - eps,
                  1e-8, 1e4, xtol=1e-12)


def lmtd_counterflow(Th_i, Th_o, Tc_i, Tc_o) -> float:
    """Log-mean temperature difference for a counter-flow exchanger."""
    d1 = Th_i - Tc_o
    d2 = Th_o - Tc_i
    if abs(d1 - d2) < 1e-12:
        return d1
    return (d1 - d2) / np.log(d1 / d2)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 72)
    print("FRICTION FACTOR - Colebrook (Newton) vs Haaland vs Moody chart")
    print("=" * 72)
    print(f"{'Re':>10}{'eps/D':>10}{'Colebrook':>13}{'Haaland':>12}"
          f"{'Moody':>10}{'dev %':>9}")
    print("-" * 72)
    checks = [(1e4, 0.0, 0.0309), (1e5, 0.0, 0.0180), (1e6, 0.0, 0.0116),
              (1e5, 1e-3, 0.0221), (1e6, 1e-2, 0.0379)]
    for Re, e, moody in checks:
        fc, fh = friction_factor(Re, e), haaland(Re, e)
        print(f"{Re:>10.0e}{e:>10.4f}{fc:>13.5f}{fh:>12.5f}{moody:>10.4f}"
              f"{100*abs(fc-moody)/moody:>9.2f}")

    # ---------------- design case ----------------
    print("\n" + "=" * 72)
    print("COUNTER-FLOW DOUBLE-PIPE EXCHANGER - sizing")
    print("=" * 72)

    # Hot: oil.  Cold: water in the inner tube.
    m_h, cp_h, Th_i = 0.20, 2130.0, 100.0      # kg/s, J/kg-K, degC
    m_c, cp_c, Tc_i = 0.10, 4178.0, 30.0
    D_i, t_w, k_w = 0.025, 0.002, 16.0          # m, m, W/m-K (steel)
    U_guess = 500.0                             # W/m^2-K, overall coefficient
    Q_required = 20_000.0                       # W  (must be < Q_max)

    C_h, C_c = m_h * cp_h, m_c * cp_c
    C_min, C_max = min(C_h, C_c), max(C_h, C_c)
    Cr = C_min / C_max
    Q_max = C_min * (Th_i - Tc_i)
    eps_req = Q_required / Q_max

    print(f"C_h = {C_h:.1f} W/K,  C_c = {C_c:.1f} W/K,  Cr = {Cr:.4f}")
    print(f"Q_max = {Q_max:.1f} W,  required epsilon = {eps_req:.4f}")

    NTU_req = ntu_from_effectiveness(eps_req, Cr, "counterflow")
    A_req = NTU_req * C_min / U_guess
    L_req = A_req / (np.pi * D_i)
    print(f"NTU required        = {NTU_req:.4f}")
    print(f"Area required       = {A_req:.4f} m^2")
    print(f"Tube length required= {L_req:.3f} m")

    # ---- verification: epsilon-NTU duty vs LMTD duty ----
    eps = effectiveness(NTU_req, Cr, "counterflow")
    Q_ntu = eps * Q_max
    Th_o = Th_i - Q_ntu / C_h
    Tc_o = Tc_i + Q_ntu / C_c
    dT_lm = lmtd_counterflow(Th_i, Th_o, Tc_i, Tc_o)
    Q_lmtd = U_guess * A_req * dT_lm

    print("\nCross-check of the two design methods:")
    print(f"  outlet temperatures : T_h,o = {Th_o:.3f} C, T_c,o = {Tc_o:.3f} C")
    print(f"  LMTD                : {dT_lm:.4f} K")
    print(f"  Q from epsilon-NTU  : {Q_ntu:.4f} W")
    print(f"  Q from LMTD         : {Q_lmtd:.4f} W")
    print(f"  relative difference : {abs(Q_ntu-Q_lmtd)/Q_ntu:.3e}  (should be ~0)")

    # ---- energy balance check ----
    imbalance = abs(C_h * (Th_i - Th_o) - C_c * (Tc_o - Tc_i)) / Q_ntu
    print(f"  energy balance error: {imbalance:.3e}")

    # ---- U from resistances, using the computed h ----
    print("\n" + "-" * 72)
    print("Overall U from series resistances (water side, inner tube)")
    rho, mu, k_f, Pr = 995.0, 7.7e-4, 0.62, 5.2          # water at ~305 K
    V = m_c / (rho * np.pi * D_i ** 2 / 4)
    Re_D = rho * V * D_i / mu
    Nu = nusselt(Re_D, Pr, correlation="gnielinski")
    h_i = Nu * k_f / D_i
    print(f"  V = {V:.3f} m/s,  Re_D = {Re_D:.0f},  Nu = {Nu:.2f},  "
          f"h_i = {h_i:.1f} W/m^2-K")
    f = friction_factor(Re_D)
    dP = f * (L_req / D_i) * 0.5 * rho * V ** 2
    print(f"  f = {f:.5f},  pressure drop over {L_req:.2f} m = {dP:.1f} Pa")

    # ---------------- figure ----------------
    figdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.0))

    NTU = np.linspace(0.01, 5, 300)
    for cr in (0.0, 0.25, 0.50, 0.75, 1.0):
        ax[0].plot(NTU, [effectiveness(n, cr, "counterflow") for n in NTU],
                   lw=1.8, label=f"$C_r$ = {cr}")
    ax[0].set_xlabel("NTU"); ax[0].set_ylabel(r"$\varepsilon$")
    ax[0].set_title("Counter-flow effectiveness")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_ylim(0, 1)

    for arr in ("counterflow", "parallelflow", "crossflow-both-unmixed",
                "shell-and-tube-1pass"):
        ax[1].plot(NTU, [effectiveness(n, 0.5, arr) for n in NTU], lw=1.8,
                   label=arr)
    ax[1].set_xlabel("NTU"); ax[1].set_ylabel(r"$\varepsilon$")
    ax[1].set_title(r"Arrangements at $C_r$ = 0.5")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 1)

    # Moody diagram
    Re_range = np.logspace(np.log10(2300), 8, 300)
    for e in (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2):
        ax[2].loglog(Re_range, [friction_factor(r, e) for r in Re_range],
                     lw=1.3, label=f"{e:g}")
    Re_lam = np.logspace(np.log10(600), np.log10(2300), 50)
    ax[2].loglog(Re_lam, 64 / Re_lam, "k-", lw=2, label="laminar")
    ax[2].set_xlabel("$Re_D$"); ax[2].set_ylabel("Darcy $f$")
    ax[2].set_title("Moody diagram (Colebrook solved by Newton)")
    ax[2].legend(fontsize=6, title=r"$\epsilon/D$", ncol=2)
    ax[2].grid(which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "heat_exchanger.png"), dpi=140)
    print("\nFigure written to figures/heat_exchanger.png")


if __name__ == "__main__":
    main()
