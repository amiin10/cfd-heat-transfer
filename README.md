<div align="center">

# cfd-heat-transfer

### Verified Finite-Difference Solvers for Classical Fluid Mechanics and Heat Transfer

*A small, self-contained solver suite where every result is checked against an
independent reference — a closed-form solution, a published benchmark, or a
theoretical convergence rate — instead of being judged by eye.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-pep8-000000.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22548111.svg)](https://doi.org/10.5281/zenodo.22549009)

<br/>

<img src="2. Lid-driven cavity/Figs/lid_driven_cavity.png" alt="Lid-driven cavity validation against Ghia et al. 1982" width="90%"/>

</div>

---

## Table of Contents

1. [Why this repository?](#why-this-repository)
2. [Method at a glance](#method-at-a-glance)
3. [Repository structure](#repository-structure)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Results](#results)
7. [How it works (deeper dive)](#how-it-works-deeper-dive)
8. [Test suite](#test-suite)
9. [Diagnostics](#diagnostics)
10. [Citation](#citation)
11. [Contact](#contact)
12. [Acknowledgments](#acknowledgments)
13. [License](#license)

---

## Why this repository?

Most student and portfolio CFD projects stop at qualitative agreement: a
contour plot that "looks right." That's not verification. Verification asks
whether the discrete equations are solved *correctly* — established through
order-of-accuracy studies and comparison with exact solutions of the same
governing equations — while validation asks whether the governing equations
are an adequate model of physical reality, established against experiment or
an independently produced benchmark (Ferziger & Perić, 2002). This repo tries
to do both, explicitly, for five classical problems:

| Problem                     | What could go wrong in a naive implementation                                   | How it's checked here                                                     |
| ---------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Blasius boundary layer**   | Shooting method converges to the wrong `f''(0)`, or thermal analogy is broken.    | RK4 + Newton shooting checked to 9 significant figures against Schlichting.  |
| **Lid-driven cavity**        | Wrong vortex structure, non-physical corner eddies, incompressibility violated.   | Matched against Ghia, Ghia & Shin (1982); `∇·u = 0` enforced to 10⁻¹⁰.       |
| **2-D steady conduction**    | Scheme "looks" second order but isn't; error hides everywhere instead of at the true singularity. | Formal grid-convergence study (`p → 2`) plus a 400-term Fourier series check. |
| **Transient conduction**     | Explicit scheme silently unstable past its limit; implicit schemes not actually higher order. | Temporal-order study isolates BTCS (1st order) vs. Crank–Nicolson (2nd order); FTCS blow-up reproduced at `Fo_mesh = 0.6`. |
| **Heat exchanger sizing**    | Two equivalent design formulas (ε-NTU, LMTD) silently disagree due to a bug.      | Cross-checked to machine precision (10⁻¹⁵) on an independent sizing case.    |

Every number reported below is reproducible by running the corresponding
script — nothing here is manually tuned or cherry-picked after the fact.

---

## Method at a glance

Each module follows the same discipline:

```
solve()  →  compare against reference  →  report the error, not just the picture
              │
              ├── closed-form analytical solution   (Blasius, conduction, transient)
              ├── published numerical benchmark      (Ghia, Ghia & Shin, 1982)
              └── theoretical order of accuracy       (grid convergence, temporal order)
```

| Component            | Definition / role                                                                 |
| --------------------- | ----------------------------------------------------------------------------------- |
| `solve_*()`           | The numerical solver for each problem (RK4 shooting, vorticity–streamfunction NS, sparse direct Laplace solve, three time-marching schemes, or the ε-NTU/LMTD design equations). |
| Reference solution     | A closed-form result, tabulated benchmark, or exact series solution, computed independently of the solver under test. |
| Error norm             | `L₂` / `L∞` / relative error against the reference, reported explicitly rather than implied by a plot. |
| Invariant checks       | Physical or mathematical properties (maximum principle, incompressibility, second law, symmetry) that a *wrong* implementation would violate regardless of whether it matches one reference number. |

The `GHIA_ERRATA` flag in the cavity solver is worth calling out: one point in
the original 1982 benchmark table breaks monotonicity and is a known
typesetting error, not a solver bug. It's retained in the dataset for
completeness but excluded from the error norms — see the note in
[Results](#results).

---

## Repository structure

```
cfd-heat-transfer/
├── 1. Blasius/
│   ├── Code/blasius_boundary_layer.py
│   └── Figs/blasius.png
├── 2. Lid-driven cavity/
│   ├── Codes/lid_driven_cavity.py
│   └── Figs/lid_driven_cavity.png
├── 3. 2D steady conduction/
│   ├── Codes/conduction_2d_steady.py
│   └── Figs/conduction_2d.png
├── 4. Transient conduction/
│   ├── Codes/conduction_1d_transient.py
│   └── Figs/conduction_1d_transient.png
├── 5. Heat exchanger/
│   ├── Codes/heat_exchanger.py
│   └── Figs/heat_exchanger.png
├── test_validation.py
├── CITATION.cff
└── README.md
```

---

## Installation

```bash
git clone https://github.com/amiin10/cfd-heat-transfer.git
cd cfd-heat-transfer
pip install -r requirements.txt
```

Tested with Python 3.9–3.12, NumPy, SciPy, Matplotlib, and pytest. No GPU
required — the slowest script (the lid-driven cavity at 129 × 129, Re = 100
and 400) runs in about five minutes on a single CPU core.

---

## Quick start

Each folder is self-contained — pick one and run it:

```bash
python "1. Blasius/Code/blasius_boundary_layer.py"            # ~1 s
python "2. Lid-driven cavity/Codes/lid_driven_cavity.py"       # ~3 min
python "3. 2D steady conduction/Codes/conduction_2d_steady.py" # ~5 s
python "4. Transient conduction/Codes/conduction_1d_transient.py" # ~1 s
python "5. Heat exchanger/Codes/heat_exchanger.py"             # ~5 s
```

Every script prints its own computed-vs-reference table to the console and
writes a figure to a local `figures/` (or `Figs/`) directory — nothing is
hidden in a notebook.

### Using a solver in your own code

The modules are small and unopinionated. For example, the Blasius solver:

```python
from blasius_boundary_layer import solve_blasius, solve_thermal, boundary_layer_metrics

f, fp, fpp = solve_blasius()                 # velocity similarity solution
metrics = boundary_layer_metrics(f, fp, fpp) # delta_99, delta*, theta, H, cf
theta = solve_thermal(f, fpp, Pr=0.7)        # thermal similarity solution
```

Each module exposes its `solve_*` function directly — see the `__main__`
block at the bottom of each script for a complete, runnable example.

---

## Results

> All figures below were regenerated directly from the scripts in this repo.
> Full result tables are printed in the "Method at a glance" companion
> scripts and reproduced in the sections that follow.

| Problem                     | Script                                              | Headline result                                                         |
| ---------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------- |
| Blasius boundary layer       | `1. Blasius/Code/blasius_boundary_layer.py`          | `f''(0) = 0.332057` vs. reference `0.332057`, rel. error `3.6 × 10⁻⁹`      |
| Lid-driven cavity            | `2. Lid-driven cavity/Codes/lid_driven_cavity.py`    | Vortex centre exact vs. Ghia at Re = 100; `ψ_min` within 0.08–0.9 %         |
| 2-D steady conduction        | `3. 2D steady conduction/Codes/conduction_2d_steady.py` | Observed order `p → 1.986` (theory: 2); centre-T within 0.0015 K of series |
| Transient conduction         | `4. Transient conduction/Codes/conduction_1d_transient.py` | BTCS order = 1.00, Crank–Nicolson order = 2.00, FTCS diverges at `Fo_mesh = 0.6` |
| Heat exchanger sizing        | `5. Heat exchanger/Codes/heat_exchanger.py`          | ε-NTU vs. LMTD duty agree to `1.6 × 10⁻¹⁵` (machine precision)             |

<img src="1. Blasius/Figs/blasius.png" alt="Blasius" width="90%"/>

<img src="3. 2D steady conduction/Figs/conduction_2d.png" alt="Conduction 2D" width="90%"/>

<img src="4. Transient conduction/Figs/conduction_1d_transient.png" alt="Transient" width="90%"/>

<img src="5. Heat exchanger/Figs/heat_exchanger.png" alt="Heat exchanger" width="90%"/>

**Note on the Ghia et al. (1982) benchmark.** Table II of the original paper
lists `v = −0.23827` at `x = 0.9063` for Re = 400 — a value that breaks the
monotonicity of the profile relative to its neighbours. This is a known
typesetting error in the 1982 table; this solver obtains `v ≈ −0.382` there,
consistent with later independent reproductions. The point is retained in
the dataset for completeness but excluded from the error norms. The
remaining sixteen tabulated stations all agree to within 0.005.

---

## How it works (deeper dive)

**Blasius (`1. Blasius/`).** The similarity equation `2f''' + f f'' = 0` is a
boundary-value problem in disguise: two initial conditions are known at
`η = 0`, but the third, `f''(0)`, must be found so that `f'(∞) = 1`. The
solver integrates the equivalent first-order system with classical RK4 and
adjusts `f''(0)` by Newton's method, using the sensitivity of `f'` at the
domain edge as the Newton derivative. The thermal problem is linear once `f`
is known and is obtained by direct quadrature — no extra iteration needed.

**Lid-driven cavity (`2. Lid-driven cavity/`).** The incompressible
Navier–Stokes equations are solved in vorticity–streamfunction form, which
eliminates pressure entirely and enforces `∇·u = 0` by construction. The key
trick: the streamfunction Poisson equation `∇²ψ = −ω` is diagonalised
*exactly* at every time step by a discrete sine transform, since a five-point
Laplacian with homogeneous Dirichlet data has a known DST eigenbasis. This
makes each Poisson solve an `O(N² log N)` direct solve with **zero**
iterative error — no SOR, no convergence tolerance to tune.

**2-D steady conduction (`3. 2D steady conduction/`).** Laplace's equation on
a rectangular plate, discretised with the standard five-point stencil and
solved as a sparse linear system (CSR + SuperLU). Because the solve is
direct, any discrepancy from the exact solution is *purely* spatial
truncation error, which is what makes a clean order-of-accuracy study
possible: refine the grid five times, watch the observed order climb
monotonically toward the theoretical `p = 2`.

**Transient conduction (`4. Transient conduction/`).** A plane wall with
convective boundaries has a known eigenfunction-series solution, with
eigenvalues `ζ tan ζ = Bi` located here by Brent's method. Three
time-marching schemes (explicit FTCS, implicit BTCS, Crank–Nicolson) are
compared against the same-mesh, `Δt → 0` reference to isolate *temporal*
error from spatial error — which is how the 1st- vs. 2nd-order behaviour is
confirmed cleanly, rather than being obscured by a fixed spatial grid.

**Heat exchanger sizing (`5. Heat exchanger/`).** Implicit Colebrook–White
friction factor via Newton iteration, Dittus–Boelter/Gnielinski correlations,
and ε-NTU relations for four exchanger arrangements. The self-consistency
check — ε-NTU and LMTD are two independent formulations of the same
steady-flow energy balance — is the sharpest test in the repo: any sign or
indexing bug in either formulation would show up as a duty mismatch, and
here they agree to `10⁻¹⁵`.

---

## Test suite

35 automated tests, all passing, asserting both reference-value agreement
*and* physical invariants that a wrong implementation would violate even if
it happened to match one benchmark number:

- **Maximum principle** — a harmonic temperature field attains its extrema
  only on the boundary.
- **Incompressibility** — the streamfunction formulation satisfies
  `∇·u = 0` to `10⁻¹⁰`.
- **Second law** — exchanger outlet temperatures never cross the opposing
  inlet, checked up to NTU = 50.
- **Counter-flow dominance** — `ε_counter ≥ ε_parallel` for all NTU and `C_r`.
- **Reynolds analogy** — `θ'(0) = f''(0)` at `Pr = 1`.
- **Round-trip inversion** — `NTU → ε → NTU` recovers the input to `10⁻⁶`.
- **Symmetry** — zero temperature gradient at the wall centreline.
- **Removable singularity** — the `C_r = 1` counter-flow branch matches the
  limit of the general formula as `C_r → 1⁻`.

```bash
pytest -v                    # full suite, 35 tests, ~26 s
pytest -v -m "not slow"      # skip the CFD runs, ~1 s
```

---

## Diagnostics

Beyond the headline error tables, each script prints intermediate
diagnostics useful for debugging your own variants:

- **Blasius** — Nusselt-number comparison against the `0.332 Pr^(1/3)`
  correlation across `Pr = 0.7–10`, showing where the correlation itself
  (not the solver) is responsible for the residual gap.
- **Cavity** — full centreline `u`/`v` profiles at both Re = 100 and 400,
  plus the secondary corner-vortex structure.
- **2-D conduction** — a log-scale absolute-error field showing the error is
  localised at the two corner singularities, not spread uniformly (which
  would indicate an implementation bug rather than a known discretisation
  limit).
- **Transient conduction** — an explicit FTCS stability sweep showing bounded
  behaviour at `Fo_mesh = 0.25` and the classic sawtooth divergence at
  `Fo_mesh = 0.6`.
- **Heat exchanger** — a Moody-diagram sweep of the Newton-solved
  Colebrook–White friction factor against Haaland, across five decades of
  Reynolds number and a range of relative roughness.

---

## Citation

If you use this code, please cite both the software and, if relevant, the
verification methodology it demonstrates. The GitHub "Cite this repository"
button reads from [`CITATION.cff`](CITATION.cff).

```bibtex
@software{hosseini_cfd_heat_transfer,
  author       = {Hosseini, Amin},
  title        = {{cfd-heat-transfer: Verified Finite-Difference Solvers
                   for Classical Fluid Mechanics and Heat Transfer Problems}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.22549009},
  url          = {https://doi.org/10.5281/zenodo.22549009}
}
```

---

## Contact

**Amin Hosseini**
✉ &nbsp; *amin.hosseini1@aut.ac.ir*

🐙 &nbsp; [github.com/amiin10](https://github.com/amiin10)

🔬 &nbsp; *Department of Mechanical Engineering, Amirkabir University of Technology (Tehran Polytechnic)*

Bug reports, feature requests, and questions are very welcome via
[GitHub Issues](https://github.com/amiin10/cfd-heat-transfer/issues).

---

## Acknowledgments

- The lid-driven cavity benchmark data is from Ghia, Ghia & Shin (1982),
  *High-Re solutions for incompressible flow using the Navier–Stokes
  equations and a multigrid method*, J. Comput. Phys. 48, 387–411.
- Boundary-layer reference values follow Schlichting & Gersten,
  *Boundary-Layer Theory*, 9th ed., Springer.
- Conduction and heat-exchanger reference formulas follow Bergman & Lavine,
  *Fundamentals of Heat and Mass Transfer*, 8th ed., Wiley.
- The verification/validation framing follows Ferziger & Perić,
  *Computational Methods for Fluid Dynamics*, 3rd ed., Springer.

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

*Built for anyone who wants their CFD code to be right, not just pretty.*

</div>
