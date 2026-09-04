"""
Automated verification & validation suite.

Every test compares a solver against an independent reference: a closed-form
analytical solution, a published benchmark, or a theoretical convergence
rate.  Run with:  pytest -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from blasius_boundary_layer import (solve_blasius, solve_thermal,
                                    boundary_layer_metrics)
from conduction_2d_steady import solve_laplace, exact_sine, exact_uniform
from conduction_1d_transient import (eigenvalues, exact_theta,
                                     solve_transient)
from lid_driven_cavity import solve_cavity, validate, GHIA_U, GHIA_Y
from heat_exchanger import (friction_factor, haaland, effectiveness,
                            ntu_from_effectiveness, lmtd_counterflow)


# ==========================================================================
# Blasius boundary layer
# ==========================================================================

class TestBlasius:

    @staticmethod
    @pytest.fixture(scope="class")
    def sol():
        return solve_blasius()

    def test_wall_shear(self, sol):
        """f''(0) must match Howarth's value 0.3320573."""
        _, _, s = sol
        assert s == pytest.approx(0.332057336, abs=1e-7)

    def test_integral_thicknesses(self, sol):
        eta, y, _ = sol
        m = boundary_layer_metrics(eta, y)
        assert m["delta_99 * sqrt(Re_x)/x"] == pytest.approx(4.910, abs=1e-3)
        assert m["delta_star * sqrt(Re_x)/x"] == pytest.approx(1.7208, abs=1e-3)
        assert m["theta_mom * sqrt(Re_x)/x"] == pytest.approx(0.6641, abs=1e-3)
        assert m["shape_factor_H"] == pytest.approx(2.591, abs=1e-2)

    def test_freestream_recovered(self, sol):
        """f' must asymptote to 1 far from the wall."""
        _, y, _ = sol
        assert y[-1, 1] == pytest.approx(1.0, abs=1e-9)

    def test_unit_prandtl_analogy(self, sol):
        """At Pr = 1 the Reynolds analogy gives theta'(0) = f''(0)."""
        eta, y, s = sol
        _, dtheta0 = solve_thermal(eta, y[:, 0], 1.0)
        assert dtheta0 == pytest.approx(s, rel=1e-4)

    @pytest.mark.parametrize("Pr", [0.7, 7.0, 10.0])
    def test_nusselt_correlation(self, sol, Pr):
        """Nu_x/sqrt(Re_x) within 2% of the 0.332 Pr^(1/3) correlation."""
        eta, y, _ = sol
        _, dtheta0 = solve_thermal(eta, y[:, 0], Pr)
        assert dtheta0 == pytest.approx(0.332 * Pr ** (1 / 3), rel=0.02)


# ==========================================================================
# 2D steady conduction
# ==========================================================================

class TestConduction2D:

    def test_matches_analytical(self):
        L, H = 1.0, 0.5
        X, Y, T = solve_laplace(81, 41, L, H,
                                lambda x: 100 * np.sin(np.pi * x / L))
        Te = exact_sine(X, Y, L, H, 100.0)
        assert np.abs(T - Te).max() < 5e-3

    def test_second_order_convergence(self):
        """Observed order of accuracy must approach the theoretical p = 2."""
        L, H = 1.0, 0.5
        errs, hs = [], []
        for n in (21, 41, 81, 161):
            X, Y, T = solve_laplace(n, n // 2 + 1, L, H,
                                    lambda x: 100 * np.sin(np.pi * x / L))
            errs.append(np.sqrt(np.mean((T - exact_sine(X, Y, L, H, 100.)) ** 2)))
            hs.append(L / (n - 1))
        p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        assert 1.9 < p < 2.1

    def test_maximum_principle(self):
        """A harmonic function attains its extrema on the boundary."""
        _, _, T = solve_laplace(41, 41, 1.0, 1.0,
                                lambda x: np.full_like(x, 100.0))
        interior = T[1:-1, 1:-1]
        assert interior.min() >= -1e-9
        assert interior.max() <= 100.0 + 1e-9

    def test_series_solution_agreement(self):
        """FD solution agrees with the Fourier series away from the corners."""
        L, H = 1.0, 0.5
        X, Y, T = solve_laplace(81, 41, L, H, lambda x: np.full_like(x, 100.))
        Te = exact_uniform(X, Y, L, H, 100.0)
        # exclude the two singular top corners
        assert np.abs(T - Te)[5:-5, 1:-5].max() < 0.5


# ==========================================================================
# 1D transient conduction
# ==========================================================================

class TestTransient1D:

    def test_eigenvalues_bi_unity(self):
        """Roots of zeta tan(zeta) = 1 (Incropera Table 5.1)."""
        z = eigenvalues(1.0)
        expected = [0.8603, 3.4256, 6.4373, 9.5293]
        assert z[:4] == pytest.approx(expected, abs=1e-4)

    def test_initial_condition(self):
        """
        theta -> 1 as Fo -> 0.

        The interior converges rapidly, but the series is evaluated at the
        convective surface where the boundary condition is discontinuous at
        t = 0, so a truncated expansion shows Gibbs ringing there.  The
        surface node is therefore checked with a looser tolerance, and we
        additionally confirm the overshoot shrinks as terms are added.
        """
        xi = np.linspace(0, 1, 21)
        th = exact_theta(xi, 1e-8, 1.0)
        assert th[:-1] == pytest.approx(np.ones(20), abs=1e-3)   # interior
        assert th[-1] == pytest.approx(1.0, abs=1e-2)            # surface

        e_coarse = abs(exact_theta(np.array([1.0]), 1e-8, 1.0,
                                   eigenvalues(1.0, 30))[0] - 1.0)
        e_fine = abs(exact_theta(np.array([1.0]), 1e-8, 1.0,
                                 eigenvalues(1.0, 240))[0] - 1.0)
        assert e_fine < e_coarse

    @pytest.mark.parametrize("scheme",
                             ["explicit", "implicit", "crank-nicolson"])
    def test_schemes_match_exact(self, scheme):
        nx, Bi, Fo = 41, 1.0, 0.2
        dxi = 1.0 / (nx - 1)
        xi, th = solve_transient(scheme, nx, Fo, Bi, 0.4 * dxi ** 2)
        assert np.abs(th - exact_theta(xi, Fo, Bi)).max() < 2e-4

    def test_ftcs_stability_limit(self):
        """FTCS is stable at Fo_mesh = 0.5 and diverges above it."""
        dxi = 1.0 / 20
        _, ok = solve_transient("explicit", 21, 0.05, 1.0, 0.5 * dxi ** 2)
        assert np.abs(ok).max() < 1.01

        _, bad = solve_transient("explicit", 21, 0.05, 1.0, 0.6 * dxi ** 2)
        assert np.abs(bad).max() > 10.0

    def test_implicit_unconditionally_stable(self):
        """Implicit schemes stay bounded at a huge time step."""
        for scheme in ("implicit", "crank-nicolson"):
            _, th = solve_transient(scheme, 41, 0.2, 1.0, 0.05)
            assert np.abs(th).max() < 1.01

    def test_crank_nicolson_second_order_in_time(self):
        ref = solve_transient("crank-nicolson", 81, 0.2, 1.0, 1e-7)[1]
        e = []
        for dFo in (1e-3, 5e-4):
            th = solve_transient("crank-nicolson", 81, 0.2, 1.0, dFo)[1]
            e.append(np.abs(th - ref).max())
        order = np.log(e[0] / e[1]) / np.log(2.0)
        assert 1.8 < order < 2.2

    def test_symmetry_bc_zero_gradient(self):
        """The centreline must have zero temperature gradient."""
        xi, th = solve_transient("crank-nicolson", 81, 0.1, 1.0, 1e-4)
        assert abs(th[1] - th[0]) < 1e-4


# ==========================================================================
# Lid-driven cavity  (slower - marked so it can be deselected)
# ==========================================================================

class TestCavity:

    @pytest.mark.slow
    def test_ghia_benchmark_re100(self):
        x, y, psi, w, u, v, info = solve_cavity(Re=100, N=64, tol=1e-6,
                                                verbose=False)
        assert info["converged"]
        _, _, eu, ev = validate(x, y, u, v, 100)
        assert eu.max() < 0.01
        assert ev.max() < 0.01

    @pytest.mark.slow
    def test_primary_vortex_re100(self):
        """psi_min = -0.1034 at (0.6172, 0.7344) per Ghia et al."""
        x, y, psi, _, _, _, _ = solve_cavity(Re=100, N=64, tol=1e-6,
                                             verbose=False)
        assert psi.min() == pytest.approx(-0.1034, abs=2e-3)
        i, j = np.unravel_index(np.argmin(psi), psi.shape)
        assert x[i] == pytest.approx(0.6172, abs=0.02)
        assert y[j] == pytest.approx(0.7344, abs=0.02)

    @pytest.mark.slow
    def test_incompressibility(self):
        """The streamfunction formulation satisfies div(u) = 0 identically."""
        x, y, psi, _, u, v, _ = solve_cavity(Re=100, N=32, tol=1e-5,
                                             verbose=False)
        h = x[1] - x[0]
        div = ((u[2:, 1:-1] - u[:-2, 1:-1]) +
               (v[1:-1, 2:] - v[1:-1, :-2])) / (2 * h)
        assert np.abs(div[1:-1, 1:-1]).max() < 1e-10


# ==========================================================================
# Heat exchanger
# ==========================================================================

class TestHeatExchanger:

    def test_laminar_friction_exact(self):
        assert friction_factor(1000.0) == pytest.approx(0.064, rel=1e-12)

    @pytest.mark.parametrize("Re,eps_D", [(1e4, 0.0), (1e5, 1e-3),
                                          (1e6, 1e-2), (1e7, 5e-4)])
    def test_colebrook_matches_haaland(self, Re, eps_D):
        """Haaland approximates Colebrook to better than 2%."""
        assert friction_factor(Re, eps_D) == pytest.approx(
            haaland(Re, eps_D), rel=0.02)

    def test_colebrook_satisfies_its_own_equation(self):
        """Substitute the root back into the implicit relation."""
        for Re, e in ((5e4, 0.0), (2e5, 2e-3)):
            f = friction_factor(Re, e)
            lhs = 1.0 / np.sqrt(f)
            rhs = -2.0 * np.log10(e / 3.7 + 2.51 / (Re * np.sqrt(f)))
            assert lhs == pytest.approx(rhs, rel=1e-10)

    def test_effectiveness_limits(self):
        """eps -> 0 as NTU -> 0;  counterflow with Cr=0 -> 1 - exp(-NTU)."""
        assert effectiveness(1e-9, 0.5) == pytest.approx(0.0, abs=1e-8)
        for ntu in (0.5, 1.0, 3.0):
            assert effectiveness(ntu, 0.0) == pytest.approx(
                1 - np.exp(-ntu), rel=1e-12)

    def test_counterflow_beats_parallelflow(self):
        """Counter-flow is always the more effective arrangement."""
        for ntu in (0.5, 1.0, 2.0, 4.0):
            for cr in (0.25, 0.5, 1.0):
                assert (effectiveness(ntu, cr, "counterflow") >=
                        effectiveness(ntu, cr, "parallelflow"))

    def test_cr_unity_counterflow_closed_form(self):
        """The Cr=1 branch must match the limit of the general formula."""
        for ntu in (0.5, 2.0, 5.0):
            assert effectiveness(ntu, 1.0) == pytest.approx(ntu / (1 + ntu))
            assert effectiveness(ntu, 1.0 - 1e-9) == pytest.approx(
                ntu / (1 + ntu), rel=1e-6)

    def test_ntu_inversion_roundtrip(self):
        for arr in ("counterflow", "parallelflow", "crossflow-both-unmixed"):
            for cr in (0.2, 0.6, 1.0):
                for ntu in (0.4, 1.5, 3.0):
                    eps = effectiveness(ntu, cr, arr)
                    assert ntu_from_effectiveness(eps, cr, arr) == \
                        pytest.approx(ntu, rel=1e-6)

    def test_ntu_and_lmtd_are_equivalent(self):
        """The two design methods must give the same duty."""
        C_h, C_c, Th_i, Tc_i = 426.0, 417.8, 100.0, 30.0
        C_min, Cr = min(C_h, C_c), min(C_h, C_c) / max(C_h, C_c)
        U, A = 500.0, 1.77
        NTU = U * A / C_min
        Q = effectiveness(NTU, Cr) * C_min * (Th_i - Tc_i)
        Th_o, Tc_o = Th_i - Q / C_h, Tc_i + Q / C_c
        Q_lmtd = U * A * lmtd_counterflow(Th_i, Th_o, Tc_i, Tc_o)
        assert Q == pytest.approx(Q_lmtd, rel=1e-9)

    def test_second_law_not_violated(self):
        """Outlet temperatures may never cross the inlet of the other stream."""
        C_h, C_c, Th_i, Tc_i = 426.0, 417.8, 100.0, 30.0
        C_min, Cr = min(C_h, C_c), min(C_h, C_c) / max(C_h, C_c)
        for NTU in (0.5, 2.0, 10.0, 50.0):
            Q = effectiveness(NTU, Cr) * C_min * (Th_i - Tc_i)
            Th_o, Tc_o = Th_i - Q / C_h, Tc_i + Q / C_c
            assert Th_o >= Tc_i - 1e-9
            assert Tc_o <= Th_i + 1e-9
