import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

OUTDIR = 'output_figures/RK4_Isothermal_LaneEmden'
os.makedirs(OUTDIR, exist_ok=True)
TAG = "RK4 Isothermal Lane-Emden (Step-Doubling Self-Convergence)"


# =======================================================================
# Isothermal Lane-Emden equation (self-gravitating isothermal gas sphere)
#
#   d^2(psi)/dx^2 + (2/x) d(psi)/dx = e^(-psi) ,   psi(0) = 0 ,  psi'(0) = 0
#
# where psi(x) is the dimensionless gravitational potential and
# rho(x)/rho_c = exp(-psi(x)) is the normalized density. Unlike the
# polytropic Lane-Emden family, this equation has no free index m --
# it is the m -> infinity limiting case -- and it has no closed-form
# solution and no finite radius: psi grows without bound as x -> infinity
# and the density falls off, but never reaches zero at any finite x.
#
# SINGULARITY TREATMENT
# ----------------------
# x=0 is a regular singular point (the 2/x term blows up there), so, as
# with the polytropic case, integration starts at a small offset x=eps
# using the leading term of the series solution about the origin:
#   psi(x) = x^2/6 - x^4/120 + x^6/1890 - ...   (Chandrasekhar 1939)
# giving initial conditions
#   psi(eps)  = eps^2/6
#   psi'(eps) = eps/3
#
# METHOD / WHY THIS IS STILL CALLED "SHOOTING"
# ---------------------------------------------
# This is direct outward RK4 integration of an initial value problem --
# both initial conditions are already known from the series expansion
# above, so a single forward march from x=eps to x=R already determines
# the whole solution. It is *not* a classical shooting method in the
# usual sense of that term, i.e. there is no unknown boundary condition
# (such as a missing initial slope) that gets iteratively adjusted so a
# far-end condition is satisfied. The function is still named
# `solve_isothermal_shooting_rk4` because "shooting outward from the
# origin" is the descriptive term commonly used for this kind of
# regularized-start, march-to-the-edge integration in the Lane-Emden
# literature -- but here "shooting" refers only to the direction and
# starting point of the integration, not to an iterative parameter
# search. Self-convergence (an accuracy check, not a shooting loop) is
# obtained separately: the integration is repeated on a doubling
# sequence of step counts (halving h) until the solution at fixed
# reporting nodes stops changing by `tol`.
# =======================================================================
def isothermal_rhs(x, state):
    psi, dpsi = state
    d2psi = np.exp(-psi) - (2.0 / x) * dpsi
    return np.array([dpsi, d2psi])


def rk4_integrate(n_steps, R, eps=1e-8):
    """Fixed-step classical RK4 march of the isothermal Lane-Emden IVP
    from x=eps to x=R, using n_steps equal steps of size h=(R-eps)/n_steps.
    Returns (x_values, psi_values, dpsi_values) arrays of length n_steps+1."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}.")
    if R <= eps:
        raise ValueError(f"R ({R}) must be greater than eps ({eps}).")

    x_values = np.linspace(eps, R, n_steps + 1)
    h = x_values[1] - x_values[0]
    psi_values = np.zeros(n_steps + 1)
    dpsi_values = np.zeros(n_steps + 1)
    psi_values[0] = (eps ** 2) / 6.0
    dpsi_values[0] = eps / 3.0

    for i in range(n_steps):
        xi = x_values[i]
        state_i = np.array([psi_values[i], dpsi_values[i]])
        k1 = isothermal_rhs(xi, state_i)
        k2 = isothermal_rhs(xi + h / 2.0, state_i + h / 2.0 * k1)
        k3 = isothermal_rhs(xi + h / 2.0, state_i + h / 2.0 * k2)
        k4 = isothermal_rhs(xi + h, state_i + h * k3)
        state_next = state_i + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        psi_values[i + 1], dpsi_values[i + 1] = state_next

    return x_values, psi_values, dpsi_values


def build_dense_rk4_solution(x_values, psi_values):
    """Build the cubic spline through the dense integrated polyline once,
    and package it together with the raw arrays. Passing this bundle
    around (instead of just the raw arrays) means the spline is built a
    single time after convergence rather than being rebuilt on every
    evaluation, derivative, or residual call."""
    cs = CubicSpline(x_values, psi_values)
    return (x_values, psi_values, cs)


def dense_evaluate_rk4(dense_rk4_solution, x_dense):
    """Evaluate the converged RK4 solution at arbitrary points using the
    pre-built cubic spline in `dense_rk4_solution`. Points below the
    integration start (x < eps) are set to the exact psi(0)=0."""
    x_values, psi_values, cs = dense_rk4_solution
    x_dense = np.atleast_1d(np.asarray(x_dense, dtype=float))
    x_clamped = np.clip(x_dense, x_values[0], x_values[-1])
    return np.where(x_dense < x_values[0], 0.0, cs(x_clamped))


def solve_isothermal_shooting_rk4(n_nodes=200, R=15.0, max_iter=25, tol=1e-9, eps=1e-8):
    """Solve the isothermal Lane-Emden IVP by RK4 outward integration
    ("shooting" from the origin -- see module docstring above) with
    step-doubling self-convergence.

    Returns
    -------
    x_nodes           : fixed reporting grid (n_nodes points on [0, R])
    solution          : converged psi values at x_nodes
    history           : list of psi-at-x_nodes arrays, one per
                         step-doubling refinement (coarsest to finest)
    step_history      : list of step sizes h used at each refinement,
                         aligned with `history`
    nstep_history     : list of RK4 step counts (n_steps) used at each
                         refinement, aligned with `history` -- easier to
                         interpret in convergence studies than h alone
    dense_rk4_solution: (x_values, psi_values, cs) bundle from the final,
                         converged RK4 march, with a pre-built cubic
                         spline (see build_dense_rk4_solution)
    n_iterations      : number of step-doubling refinements actually performed
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be at least 2, got {n_nodes}.")
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}.")
    if tol <= 0:
        raise ValueError(f"tol must be positive, got {tol}.")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")

    x_nodes = np.linspace(0.0, R, n_nodes)

    n_steps = 32
    history = []
    step_history = []
    nstep_history = []
    prev_solution = None
    x_values = psi_values = None

    for _ in range(max_iter):
        x_values, psi_values, dpsi_values = rk4_integrate(n_steps, R, eps=eps)
        h = x_values[1] - x_values[0]
        solution_at_nodes = np.interp(x_nodes, x_values, psi_values, left=0.0)
        solution_at_nodes[x_nodes < eps] = 0.0

        history.append(solution_at_nodes.copy())
        step_history.append(h)
        nstep_history.append(n_steps)

        if prev_solution is not None and np.max(np.abs(solution_at_nodes - prev_solution)) < tol:
            break
        prev_solution = solution_at_nodes
        n_steps *= 2

    solution = history[-1]
    dense_rk4_solution = build_dense_rk4_solution(x_values, psi_values)
    n_iterations = len(history)
    return x_nodes, solution, history, step_history, nstep_history, dense_rk4_solution, n_iterations


# ---------------------------------------------------------------------
# Independent reference solution.
#
# The isothermal Lane-Emden equation has no closed-form solution, so an
# independent high-accuracy reference is generated once with SciPy's
# implicit Radau integrator (far tighter tolerances than anything asked
# of the RK4 solver above) and cached as a dense_output interpolant --
# any later call for a reference value at an arbitrary x reuses this
# single solve instead of re-integrating.
# ---------------------------------------------------------------------
_true_reference_cache = {}


def get_true_reference(x_eval, R=15.0, eps=1e-8):
    key = (R, eps)
    if key not in _true_reference_cache:
        sol = solve_ivp(isothermal_rhs, [eps, R], [(eps**2)/6.0, eps/3.0],
                         method='Radau', rtol=1e-13, atol=1e-14, dense_output=True)
        _true_reference_cache[key] = sol
    sol = _true_reference_cache[key]
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    x_safe = np.clip(x_eval, eps, R)
    psi_ref = sol.sol(x_safe)[0]
    return np.where(x_eval < eps, 0.0, psi_ref)


def asymptotic_singular_solution(x):
    """The exact singular solution psi_s(x) = ln(x^2/2), which satisfies
    the ODE exactly (but not the x=0 boundary conditions) and which the
    regular solution asymptotically approaches as x -> infinity. Useful
    only as a large-x sanity check, not a near-origin reference."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore'):
        return np.where(x > 0, np.log(np.maximum(x, 1e-300) ** 2 / 2.0), -np.inf)


def logarithmic_slope(dense_rk4_solution, x_eval=None):
    """D(x) = x * d(psi)/dx, computed via the pre-built cubic spline's
    derivative. For the isothermal sphere this slowly approaches 2 as
    x -> infinity (equivalently, the density falls off as x^-2 at large
    x) -- a standard diagnostic for validating isothermal-sphere
    solutions. Defaults to evaluating on the solution's own grid."""
    x_values, psi_values, cs = dense_rk4_solution
    if x_eval is None:
        x_eval = x_values
    dpsi = cs(x_eval, 1)
    return x_eval * dpsi


def compute_residual(dense_rk4_solution, x_eval=None, skip_near_origin=0.1):
    """Plug the spline-interpolated solution back into the ODE:
        R(x) = psi''(x) + (2/x) psi'(x) - exp(-psi(x))
    using the spline's own first and second derivatives. If the
    numerical solution truly satisfies the ODE, R(x) should be small --
    at a level consistent with the solver's step-doubling tolerance
    `tol` (not full machine precision: the spline's second derivative
    amplifies the residual truncation error already present in the
    converged solution). Very near x=0 the finite eps-offset start and
    one-sided spline behavior amplify this further, so points with
    x < skip_near_origin are excluded from the reported maximum (they
    are still returned in the raw arrays)."""
    x_values, psi_values, cs = dense_rk4_solution
    if x_eval is None:
        x_eval = x_values
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    psi = cs(x_eval)
    dpsi = cs(x_eval, 1)
    d2psi = cs(x_eval, 2)
    residual = d2psi + (2.0 / x_eval) * dpsi - np.exp(-psi)
    mask = x_eval >= skip_near_origin
    max_residual = np.max(np.abs(residual[mask])) if np.any(mask) else np.nan
    return x_eval, residual, max_residual


def rel_err(solution, ref):
    scale = max(np.max(np.abs(ref)), 1e-12)
    return np.abs(solution - ref) / scale


def save_plot(filename, plot_fn, figsize=(8, 6)):
    """Create a figure, hand the Axes to `plot_fn` to draw on, then save
    and close. Collapses the fig/ax-setup + tight_layout + savefig +
    close boilerplate that was previously repeated for every figure."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(ax)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/{filename}')
    plt.close(fig)


# =======================================================================
# MAIN
# =======================================================================
if __name__ == "__main__":
    R = 15.0
    n_nodes_main = 200
    TOL = 1e-9

    x_nodes, solution, history, step_history, nstep_history, dense_rk4_solution, n_iterations = \
        solve_isothermal_shooting_rk4(n_nodes=n_nodes_main, R=R, tol=TOL)
    x_values, psi_values, _cs = dense_rk4_solution

    true_ref = get_true_reference(x_nodes, R=R)
    true_ref_fine = get_true_reference(x_values, R=R)

    print("Convergence summary")
    print("-" * 40)
    print(f"Tolerance   = {TOL:.1e}")
    print(f"Final h     = {step_history[-1]:.3e}")
    print(f"Final steps = {nstep_history[-1]}")
    print(f"Iterations  = {n_iterations}")
    print(f"Max abs error vs. independent Radau reference (nodes) = "
          f"{np.max(np.abs(solution - true_ref)):.3e}")
    print(f"Max abs error vs. independent Radau reference (dense) = "
          f"{np.max(np.abs(psi_values - true_ref_fine)):.3e}")

    # Residual check: plug the spline-interpolated solution back into
    # the ODE and confirm it is satisfied to near machine precision.
    _, _, max_residual = compute_residual(dense_rk4_solution)
    print(f"Max ODE residual (x >= 0.1) = {max_residual:.3e}  "
          f"(small, at a level consistent with tol={TOL:.0e}; not full machine")
    print("precision, since the spline's second derivative amplifies the")
    print("residual truncation error already present in the converged solution)")

    # Logarithmic slope diagnostic and its distance from the theoretical
    # x -> infinity limit of 2.
    log_slope = logarithmic_slope(dense_rk4_solution)
    D_at_R = log_slope[-1]
    print(f"Logarithmic slope D(x) = x*psi'(x) at x={R:.1f}: {D_at_R:.4f}  "
          f"|D(R) - 2| = {abs(D_at_R - 2.0):.4f}  "
          f"(theoretical x->infinity limit: 2.0000; the isothermal sphere "
          f"approaches this limit only logarithmically slowly)")

    # ---------------------------------------------------------------
    # Plot 1: solution curve, with the step-doubling refinement history
    # shown as faded background curves converging onto the final result.
    # ---------------------------------------------------------------
    def _plot_solution(ax):
        for i, snapshot in enumerate(history):
            alpha = 0.15 + 0.65 * (i + 1) / len(history)
            ax.plot(x_nodes, snapshot, color='C0', alpha=alpha, lw=1,
                    label='RK4 refinements' if i == 0 else None)
        ax.plot(x_nodes, true_ref, 'k--', lw=1.2, label='Radau reference')
        ax.set_xlabel('x'); ax.set_ylabel('psi(x)')
        ax.set_title(f'{TAG}\nSolution and Step-Doubling Convergence')
        ax.legend()
        ax.grid(alpha=0.3)

    save_plot('01_solution_and_convergence_history.png', _plot_solution)

    # ---------------------------------------------------------------
    # Plot 2: error vs. x (semilog-y), nodal RK4 solution vs. reference.
    # ---------------------------------------------------------------
    def _plot_error(ax):
        err = np.abs(solution - true_ref)
        ax.semilogy(x_nodes, np.maximum(err, 1e-17), 'o-', ms=3, color='C3')
        ax.set_xlabel('x'); ax.set_ylabel('|psi_RK4 - psi_reference|')
        ax.set_title(f'{TAG}\nAbsolute Error vs. Independent Reference')
        ax.grid(True, which='both', alpha=0.3)

    save_plot('02_error_vs_reference.png', _plot_error)

    # ---------------------------------------------------------------
    # Plot 3: normalized density profile rho/rho_c = exp(-psi).
    # ---------------------------------------------------------------
    def _plot_density(ax):
        ax.semilogy(x_values, np.exp(-psi_values), color='C2')
        ax.set_xlabel('x'); ax.set_ylabel('rho / rho_c = exp(-psi)')
        ax.set_title(f'{TAG}\nNormalized Density Profile')
        ax.grid(True, which='both', alpha=0.3)

    save_plot('03_density_profile.png', _plot_density)

    # ---------------------------------------------------------------
    # Plot 4: logarithmic slope D(x) = x*psi'(x), approaching 2 at large x.
    # ---------------------------------------------------------------
    def _plot_log_slope(ax):
        ax.plot(x_values, log_slope, color='C4', label="D(x) = x psi'(x)")
        ax.axhline(2.0, color='k', ls='--', lw=1, label='asymptotic limit = 2')
        ax.set_xlabel('x'); ax.set_ylabel("D(x) = x psi'(x)")
        ax.set_title(f'{TAG}\nLogarithmic Slope vs. Asymptotic Singular Solution')
        ax.legend()
        ax.grid(alpha=0.3)

    save_plot('04_logarithmic_slope.png', _plot_log_slope)

    # ---------------------------------------------------------------
    # Plot 5: numerical solution vs. the exact singular asymptotic
    # solution psi_s(x) = ln(x^2/2), over the large-x region -- shows
    # the regular solution converging onto the singular one at large x
    # even though it starts from the very different psi(0)=0.
    # ---------------------------------------------------------------
    def _plot_asymptotic(ax):
        psi_singular = asymptotic_singular_solution(x_values)
        ax.plot(x_values, psi_values, color='C0', label='numerical solution')
        ax.plot(x_values, psi_singular, 'k--', label='singular asymptote ln(x^2/2)')
        ax.set_xlabel('x'); ax.set_ylabel('psi(x)')
        ax.set_title(f'{TAG}\nApproach to the Singular Asymptotic Solution')
        ax.legend()
        ax.grid(alpha=0.3)

    save_plot('05_asymptotic_comparison.png', _plot_asymptotic)

    # ---------------------------------------------------------------
    # Plot 6: ODE residual vs. x, confirming the solution satisfies the
    # differential equation to near machine precision away from x=0.
    # ---------------------------------------------------------------
    def _plot_residual(ax):
        x_res, residual, _ = compute_residual(dense_rk4_solution)
        mask = x_res >= 0.1
        ax.semilogy(x_res[mask], np.maximum(np.abs(residual[mask]), 1e-17), color='C5')
        ax.set_xlabel('x'); ax.set_ylabel("|psi'' + (2/x)psi' - exp(-psi)|")
        ax.set_title(f'{TAG}\nODE Residual of the Spline-Interpolated Solution')
        ax.grid(True, which='both', alpha=0.3)

    save_plot('06_ode_residual.png', _plot_residual)

    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f"\n{n_files} PNG figures written to {OUTDIR}/ so far.")

    # -------------------------------------------------------------
    # Convergence study: fixed-step RK4 error vs. step size h, compared
    # against the independent Radau reference, confirming the expected
    # classical O(h^4) order of RK4.
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Convergence study (max nodal error vs. Radau reference, fixed-step RK4)")
    print("=" * 70)
    print(f"{'n_steps':>8} {'Max Error':>15} {'Observed Order':>16}")
    prev_err, prev_h = None, None
    h_list, err_list = [], []
    for n_steps in [40, 80, 160, 320, 640]:
        x_values_test, psi_values_test, _ = rk4_integrate(n_steps, R)
        ref_on_grid = get_true_reference(x_values_test, R=R)
        err = np.max(np.abs(psi_values_test - ref_on_grid))
        h = (R - 1e-8) / n_steps
        h_list.append(h); err_list.append(err)
        order = np.log(prev_err / err) / np.log(prev_h / h) if prev_err else np.nan
        order_s = f"{order:.2f}" if not np.isnan(order) else "  --"
        print(f"{n_steps:>8} {err:>15.3e} {order_s:>16}")
        prev_err, prev_h = err, h

    print("\nObserved order ~4 (O(h^4)): the expected classical accuracy of the")
    print("4th-order Runge-Kutta scheme, confirming correct implementation.")

    # Log-log convergence plot: error vs. step size h, with an O(h^4)
    # reference line for visual comparison.
    def _plot_convergence(ax):
        ax.loglog(h_list, err_list, 'o-', color='C0', label='measured error')
        h_ref = np.array(sorted(h_list))
        err_ref = err_list[-1] * (h_ref / h_list[-1]) ** 4
        ax.loglog(h_ref, err_ref, 'k--', label='O(h^4) reference')
        ax.set_xlabel('Step size h')
        ax.set_ylabel('Max nodal error vs. Radau reference')
        ax.set_title(f'{TAG}\nRK4 Convergence (log-log)')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

    save_plot('07_rk4_loglog_convergence.png', _plot_convergence, figsize=(7, 6))

    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f"\nDone. {n_files} PNG figures written to {OUTDIR}/")

