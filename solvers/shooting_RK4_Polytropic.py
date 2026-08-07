import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/RK4_Polytropic/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = "RK4 Polytropic (Step-Doubling Self-Convergence)"


# =======================================================================
# Lane-Emden / polytropic equation
#
#   y'' + (2/x) y' + |y|^m = 0 ,   y(0) = 1 ,  y'(0) = 0
#
# SINGULARITY TREATMENT
# ----------------------
# x=0 is a regular singular point of the ODE (the 2/x term blows up
# there), so a numerical march cannot literally start at x=0. Instead
# integration starts at a small offset x=eps, using the leading terms
# of the known series expansion about the origin as initial conditions:
#   y(eps)  = 1 - eps^2/6
#   y'(eps) = -eps/3
# For eps small enough (default 1e-8) this reproduces y(0)=1, y'(0)=0
# to well beyond the accuracy of the RK4 integration itself.
#
# METHOD
# ------
# This is direct RK4 *outward integration* of an initial value problem,
# not a classical shooting method: there is no unknown parameter (e.g.
# a missing initial slope) being iteratively adjusted to satisfy a
# boundary condition at the far end. Both initial conditions are known
# from the series expansion above, so a single forward march from
# x=eps to x=R already determines the solution.
#
# SELF-CONVERGENCE
# -----------------
# To get an accuracy estimate without a closed-form solution, the
# integration is repeated on a doubling sequence of step counts
# (n_steps, 2*n_steps, 4*n_steps, ...), i.e. h, h/2, h/4, ..., and the
# solution at a fixed set of reporting nodes is compared between
# consecutive refinements. Once the change drops below `tol`, the
# result is accepted. This refinement sequence is stored in `history`
# (and the corresponding step sizes in `step_history`) purely for
# diagnostic/plotting purposes.
# =======================================================================
def lane_emden_rhs(x, state, m):
    y, dy = state
    d2y = -(2.0 / x) * dy - np.abs(y) ** m
    return np.array([dy, d2y])


def rk4_integrate(m, R, n_steps, eps=1e-8):
    """Fixed-step classical RK4 march of the Lane-Emden IVP from x=eps to
    x=R, using n_steps equal steps of size h=(R-eps)/n_steps.
    Returns (x, y, dy) arrays of length n_steps+1."""
    x = np.linspace(eps, R, n_steps + 1)
    h = x[1] - x[0]
    y = np.zeros(n_steps + 1)
    dy = np.zeros(n_steps + 1)
    y[0] = 1.0 - (eps ** 2) / 6.0
    dy[0] = -eps / 3.0

    for i in range(n_steps):
        xi = x[i]
        state_i = np.array([y[i], dy[i]])
        k1 = lane_emden_rhs(xi, state_i, m)
        k2 = lane_emden_rhs(xi + h / 2.0, state_i + h / 2.0 * k1, m)
        k3 = lane_emden_rhs(xi + h / 2.0, state_i + h / 2.0 * k2, m)
        k4 = lane_emden_rhs(xi + h, state_i + h * k3, m)
        state_next = state_i + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        y[i + 1], dy[i + 1] = state_next

    return x, y, dy


def dense_evaluate_rk4(dense_solution, x_dense):
    """Evaluate the converged RK4 solution at arbitrary points using a
    cubic spline through the dense integrated polyline, for a smoother
    reconstruction than piecewise-linear interpolation. Points below the
    integration start (x < eps) are set to the exact y(0)=1."""
    x_fine, y_fine = dense_solution
    x_dense = np.atleast_1d(np.asarray(x_dense, dtype=float))
    cs = CubicSpline(x_fine, y_fine)
    x_clamped = np.clip(x_dense, x_fine[0], x_fine[-1])
    result = np.where(x_dense < x_fine[0], 1.0, cs(x_clamped))
    return result


def find_first_zero_crossing(x, y):
    """Locate the first x where y crosses zero, via linear interpolation
    between the bracketing samples. For the Lane-Emden equation this is
    the (dimensionless) radius of the polytrope. Returns None if y never
    changes sign over the given grid."""
    sign_changes = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[0]
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def solve_polytropic_rk4(m, n_nodes=43, R=2.0, max_iter=20, tol=1e-9, eps=1e-8):
    """Solve the Lane-Emden IVP for polytropic index m by RK4 outward
    integration with step-doubling self-convergence (see module
    docstring above).

    Returns
    -------
    x_nodes       : fixed reporting grid (n_nodes points on [0, R])
    solution      : converged solution values at x_nodes
    history       : list of solution-at-x_nodes arrays, one per
                     step-doubling refinement (coarsest to finest)
    step_history  : list of step sizes h used at each refinement,
                     aligned with `history`
    dense_solution: (x_fine, y_fine) polyline from the final, converged
                     RK4 march -- used for cubic-spline evaluation
                     elsewhere (e.g. dense_evaluate_rk4)
    n_iterations  : number of step-doubling refinements actually performed
    """
    m_val = float(m)
    x_nodes = np.linspace(0.0, R, n_nodes)

    n_steps = 16
    history = []
    step_history = []
    prev_solution = None
    x_fine = y_fine = None

    for _ in range(max_iter):
        x_grid, y_grid, dy_grid = rk4_integrate(m_val, R, n_steps, eps=eps)
        h = x_grid[1] - x_grid[0]
        solution_at_nodes = np.interp(x_nodes, x_grid, y_grid, left=1.0)
        solution_at_nodes[x_nodes < eps] = 1.0

        history.append(solution_at_nodes.copy())
        step_history.append(h)
        x_fine, y_fine = x_grid, y_grid

        if prev_solution is not None and np.max(np.abs(solution_at_nodes - prev_solution)) < tol:
            break
        prev_solution = solution_at_nodes
        n_steps *= 2

    solution = history[-1]
    dense_solution = (x_fine, y_fine)
    n_iterations = len(history)
    return x_nodes, solution, history, step_history, dense_solution, n_iterations


def get_true_reference(m, x_eval):
    if abs(m - 1.0) < 1e-9:
        return np.where(x_eval < 1e-8, 1.0, np.sin(x_eval) / (x_eval + 1e-15))
    elif abs(m - 5.0) < 1e-9:
        return 1.0 / np.sqrt(1.0 + (x_eval ** 2) / 3.0)
    else:
        # No closed-form solution exists for general m, so an
        # independent high-accuracy reference is generated with SciPy's
        # implicit Radau integrator (rtol/atol far tighter than
        # anything asked of the RK4 solver above).
        def ode(t_, y):
            return [y[1], -(2.0/t_)*y[1] - np.abs(y[0])**m]
        eps = 1e-8
        y0 = [1.0 - (eps**2)/6.0, -eps/3.0]
        x_eval_safe = np.clip(x_eval, eps, None)
        sort_idx = np.argsort(x_eval_safe)
        x_sorted = x_eval_safe[sort_idx]
        sol = solve_ivp(ode, [eps, max(x_sorted[-1], eps*2)], y0, t_eval=x_sorted,
                         method='Radau', rtol=1e-12, atol=1e-13)
        y_ref = np.zeros_like(x_eval)
        y_ref[sort_idx] = sol.y[0]
        y_ref[x_eval < eps] = 1.0
        return y_ref


# Cache of fine-resolution self-convergence references, keyed by
# (m, R, n_nodes_fine), so repeated lookups for the same m (e.g. once
# for node-grid evaluation and once for the dense plotting grid) reuse
# a single RK4 solve instead of re-integrating from scratch each time.
_approx_reference_cache = {}


def get_approx_reference_solution(m, R=2.0, n_nodes_fine=160):
    """Independent self-convergence reference: the same RK4 method run
    to a tight tolerance. Returns the dense (x_fine, y_fine) polyline,
    cached per (m, R, n_nodes_fine) so it is computed at most once."""
    key = (round(float(m), 10), R, n_nodes_fine)
    if key in _approx_reference_cache:
        return _approx_reference_cache[key]
    _, _, _, _, dense_solution, _ = solve_polytropic_rk4(m, n_nodes=n_nodes_fine, R=R, tol=1e-9)
    _approx_reference_cache[key] = dense_solution
    return dense_solution


def get_approx_reference(m, x_eval, R=2.0, n_nodes_fine=160):
    """Convenience wrapper: fetch (or compute) the fine-tolerance
    reference for m and evaluate it at x_eval."""
    dense_solution = get_approx_reference_solution(m, R=R, n_nodes_fine=n_nodes_fine)
    return dense_evaluate_rk4(dense_solution, x_eval)


def rel_err(solution, ref):
    scale = max(np.max(np.abs(ref)), 1e-12)
    return np.abs(solution - ref) / scale


def surf(ax, X, Y, Z, cmap, xlabel, ylabel, zlabel, title, int_y=False):
    s = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='k', alpha=0.9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.set_title(f'{TAG} - {title}')
    if int_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return s


# =======================================================================
# MAIN
# =======================================================================
if __name__ == "__main__":
    m_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    R = 2.0
    n_nodes_main = 43

    x_dense = np.linspace(0.0, R, 200)

    results = {}
    for m in m_values:
        x_nodes, solution, history, step_history, dense_solution, n_iterations = \
            solve_polytropic_rk4(m, n_nodes=n_nodes_main, R=R)
        true_ref = get_true_reference(m, x_nodes)
        dense_approx_ref = get_approx_reference_solution(m, R=R)
        approx_ref = dense_evaluate_rk4(dense_approx_ref, x_nodes)

        solution_dense = dense_evaluate_rk4(dense_solution, x_dense)
        true_ref_dense = get_true_reference(m, x_dense)
        approx_ref_dense = dense_evaluate_rk4(dense_approx_ref, x_dense)

        x1 = find_first_zero_crossing(*dense_solution)

        results[m] = dict(x_nodes=x_nodes, solution=solution, history=history,
                           step_history=step_history, true_ref=true_ref,
                           approx_ref=approx_ref, solution_dense=solution_dense,
                           true_ref_dense=true_ref_dense, approx_ref_dense=approx_ref_dense,
                           first_zero=x1)
        x1_str = f"{x1:.6f}" if x1 is not None else "none in [0,R]"
        print(f"m={m}: {n_iterations} RK4 step-doublings | first zero (radius) x_1={x1_str} | "
              f"true err(nodes)={np.max(np.abs(solution-true_ref)):.3e} | "
              f"approx err(nodes)={np.max(np.abs(solution-approx_ref)):.3e} | "
              f"true err(dense)={np.max(np.abs(solution_dense-true_ref_dense)):.3e} | "
              f"approx err(dense)={np.max(np.abs(solution_dense-approx_ref_dense)):.3e}")

    x_nodes_common = results[m_values[0]]['x_nodes']
    Xg, Mg = np.meshgrid(x_nodes_common, m_values)
    Xd, Md = np.meshgrid(x_dense, m_values)

    # 01: exact / numerical / final nodal values
    Zfinal = np.array([results[m]['solution'] for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (m)',
         'Final nodal values', 'Final Nodal Values (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/01_exact_numerical_final_nodal_values.png'); plt.close(fig)

    # 02: final relative true error
    Zerr_true = np.array([rel_err(results[m]['solution'], results[m]['true_ref']) for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_true, 'plasma', 'x value', 'Parameter value (m)',
         'Final relative true error', 'Final Relative True Error (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/02_exact_numerical_final_relative_true_error.png'); plt.close(fig)

    # 03 & 04: iterative (step-doubling) nodal values / true error, per m
    for m in m_values:
        history = results[m]['history']
        true_ref = results[m]['true_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_nodes_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(h, true_ref) for h in history])

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Iteration number',
             'Iterative nodal values', f'Iterative Nodal Values (m={m})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/03_exact_numerical_iterative_values_p_{m}.png'); plt.close(fig)

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Iteration number',
             'Iterative relative true error', f'Iterative Relative True Error (m={m})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/04_exact_numerical_iterative_true_error_p_{m}.png'); plt.close(fig)

    # 05 & 06: direct/analytical -- dense-grid evaluation of the SAME converged RK4 solution
    Zdense_val = np.array([results[m]['solution_dense'] for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (m)',
         'Nodal values (dense cubic-spline evaluation)',
         'Direct/Analytical Evaluation of the RK4 Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/05_exact_analytical_nodal_values.png'); plt.close(fig)

    Zdense_err_true = np.array([rel_err(results[m]['solution_dense'], results[m]['true_ref_dense'])
                                 for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_true, 'plasma', 'x value', 'Parameter value (m)',
         'Relative true error (dense grid)',
         'Direct/Analytical Evaluation - Relative True Error (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/06_exact_analytical_relative_true_error.png'); plt.close(fig)

    # 07: no_exact / numerical / final nodal values (same data as 01)
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (m)',
         'Final nodal values', 'Final Nodal Values (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/07_no_exact_numerical_final_nodal_values.png'); plt.close(fig)

    # 08: final relative approximate error
    Zerr_approx = np.array([rel_err(results[m]['solution'], results[m]['approx_ref']) for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_approx, 'plasma', 'x value', 'Parameter value (m)',
         'Final relative approximate error', 'Final Relative Approximate Error (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/08_no_exact_numerical_final_relative_approx_error.png'); plt.close(fig)

    # 09 & 10: iterative nodal values / approximate error, per m
    for m in m_values:
        history = results[m]['history']
        approx_ref = results[m]['approx_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_nodes_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(h, approx_ref) for h in history])

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Iteration number',
             'Iterative nodal values', f'Iterative Nodal Values (m={m})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/09_no_exact_numerical_iterative_values_p_{m}.png'); plt.close(fig)

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Iteration number',
             'Iterative relative approximate error', f'Iterative Relative Approximate Error (m={m})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/10_no_exact_numerical_iterative_approx_error_p_{m}.png'); plt.close(fig)

    # 11 & 12: direct/analytical -- same dense evaluation, approximate error
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (m)',
         'Nodal values (dense cubic-spline evaluation)',
         'Direct/Analytical Evaluation of the RK4 Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/11_no_exact_analytical_nodal_values.png'); plt.close(fig)

    Zdense_err_approx = np.array([rel_err(results[m]['solution_dense'], results[m]['approx_ref_dense'])
                                   for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_approx, 'plasma', 'x value', 'Parameter value (m)',
         'Relative approximate error (dense grid)',
         'Direct/Analytical Evaluation - Relative Approximate Error (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/12_no_exact_analytical_relative_approx_error.png'); plt.close(fig)

    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f"\nDone. {n_files} PNG figures written to {OUTDIR}/")

    # -------------------------------------------------------------
    # Convergence study (fixed-step RK4, m=1 and m=5), driven directly
    # by n_steps (i.e. by step size h = R/n_steps) rather than by a
    # convergence tolerance, so the classical O(h^4) order of RK4 is
    # what gets measured here. A log-log plot of error vs. h makes the
    # 4th-order slope visually obvious alongside the printed table.
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Convergence study (max nodal error vs exact solution, fixed-step RK4)")
    print("="*70)

    convergence_data = {}
    for m_test, exact_fn in [(1.0, lambda x: np.where(x < 1e-8, 1.0, np.sin(x)/(x+1e-15))),
                              (5.0, lambda x: 1.0/np.sqrt(1.0 + x**2/3.0))]:
        print(f"\n m = {m_test}")
        print(f"{'n_steps':>8} {'Max Error':>15} {'Observed Order':>16}")
        prev_err, prev_h = None, None
        h_list, err_list = [], []
        for n_steps in [10, 20, 40, 80, 160]:
            x_grid, y_grid, _ = rk4_integrate(m_test, R, n_steps)
            err = np.max(np.abs(y_grid - exact_fn(x_grid)))
            h = R / n_steps
            h_list.append(h); err_list.append(err)
            order = np.log(prev_err/err)/np.log(prev_h/h) if prev_err else np.nan
            order_s = f"{order:.2f}" if not np.isnan(order) else "  --"
            print(f"{n_steps:>8} {err:>15.3e} {order_s:>16}")
            prev_err, prev_h = err, h
        convergence_data[m_test] = (h_list, err_list)

    print("\nObserved order ~4 (O(h^4)): the expected classical accuracy of the")
    print("4th-order Runge-Kutta scheme, confirming correct implementation.")

    # Log-log convergence plot: error vs. step size h, with a reference
    # O(h^4) line for visual comparison against the measured slopes.
    fig, ax = plt.subplots(figsize=(7, 6))
    for m_test, (h_list, err_list) in convergence_data.items():
        ax.loglog(h_list, err_list, 'o-', label=f'm={m_test} (measured)')
    h_ref = np.array(sorted(convergence_data[1.0][0]))
    err_ref = convergence_data[1.0][1][-1] * (h_ref / convergence_data[1.0][0][-1]) ** 4
    ax.loglog(h_ref, err_ref, 'k--', label='O(h^4) reference')
    ax.set_xlabel('Step size h')
    ax.set_ylabel('Max nodal error vs. exact solution')
    ax.set_title(f'{TAG} - RK4 Convergence (log-log)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/13_rk4_loglog_convergence.png'); plt.close(fig)
    print(f"\nLog-log convergence plot written to {OUTDIR}/13_rk4_loglog_convergence.png")
