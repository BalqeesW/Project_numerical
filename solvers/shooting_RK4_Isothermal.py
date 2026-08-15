"""
RK4 Outward Shooting (Step-Doubling Self-Convergence) -- Isothermal
Lane-Emden gas sphere model (lambda-parametrized, 28-figure output
mirroring the FDM script)

        psi''(x) + (2/x) psi'(x) = lambda * exp(-psi(x)) ,  psi(0) = 0 ,  psi'(0) = 0

This is the lambda-parametrized isothermal-gas-sphere form of the Lane-Emden
equation. It is solved by direct outward RK4 integration of the initial
value problem -- both initial conditions are already known from the series
expansion about the origin, so a single forward march from x=eps to x=R
already determines the whole solution. The function is still named
`solve_isothermal_shooting_rk4` because "shooting outward from the origin"
is the descriptive term commonly used for this kind of regularized-start,
march-to-the-edge integration in the Lane-Emden literature -- but here
"shooting" refers only to the direction and starting point of the
integration, not to an iterative parameter search.

SINGULARITY TREATMENT
----------------------
x=0 is a regular singular point (the 2/x term blows up there), so
integration starts at a small offset x=eps using the leading term of the
lambda-parametrized series solution about the origin:
    psi(x) = (lambda/6) x^2 - ...
giving initial conditions
    psi(eps)  = (lambda/6) eps^2
    psi'(eps) = (lambda/3) eps

SELF-CONVERGENCE ("Step-Doubling Refinements")
------------------------------------------------
For a fixed lambda, the integration is repeated on a doubling sequence of
step counts (halving h) until the solution at fixed reporting nodes stops
changing by `tol`. In this script, that sequence of increasingly refined
nodal solutions -- coarsest to finest -- is the "history" that plays
exactly the role the FDM script's Newton-iteration history plays: figures
03, 04, 09, and 10 plot it against a "Step-doubling refinement" axis in
place of "Iteration number".

--------------------------------------------------------------------------
Output structure
--------------------------------------------------------------------------
This script reproduces the exact 28-figure template, file-naming
convention, 3D surface-plotting logic, and true/self-convergence
analytical-comparison structure of the FDM isothermal script
(finite_difference_isothermal.py), with these substitutions:
    * Newton-iteration history                                  for  step-doubling refinement history (coarsest to finest)
    * "Iteration number" axis label (figs 03/04/09/10)          for  "Step-doubling refinement"
    * n_elem_fine mesh-refinement self-convergence reference    for  a very fine step-doubling RK4 solve (n_nodes=160),
                                                                       evaluated via the pre-built cubic-spline dense_evaluate_rk4
"""
import os
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/RK4_Isothermal/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = 'RK4 Isothermal'
DEFAULT_R = 2.0


# =======================================================================
# Core RK4 machinery (lambda-parametrized)
# =======================================================================
def isothermal_rhs(x, state, lam):
    """psi'' = lambda*exp(-psi) - (2/x)*psi'"""
    psi, dpsi = state
    d2psi = lam * np.exp(-psi) - (2.0 / x) * dpsi
    return np.array([dpsi, d2psi])


def rk4_integrate(n_steps, R, lam, eps=1e-8):
    """Fixed-step classical RK4 march of the isothermal Lane-Emden IVP
    from x=eps to x=R, using n_steps equal steps of size h=(R-eps)/n_steps.
    Returns (x_values, psi_values, dpsi_values) arrays of length n_steps+1."""
    if n_steps < 1:
        raise ValueError(f'n_steps must be at least 1, got {n_steps}.')
    if R <= eps:
        raise ValueError(f'R ({R}) must be greater than eps ({eps}).')

    x_values = np.linspace(eps, R, n_steps + 1)
    h = x_values[1] - x_values[0]
    psi_values = np.zeros(n_steps + 1)
    dpsi_values = np.zeros(n_steps + 1)
    psi_values[0] = (lam / 6.0) * eps ** 2
    dpsi_values[0] = (lam / 3.0) * eps

    for i in range(n_steps):
        xi = x_values[i]
        state_i = np.array([psi_values[i], dpsi_values[i]])
        k1 = isothermal_rhs(xi, state_i, lam)
        k2 = isothermal_rhs(xi + h / 2.0, state_i + h / 2.0 * k1, lam)
        k3 = isothermal_rhs(xi + h / 2.0, state_i + h / 2.0 * k2, lam)
        k4 = isothermal_rhs(xi + h, state_i + h * k3, lam)
        state_next = state_i + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        psi_values[i + 1], dpsi_values[i + 1] = state_next

    return x_values, psi_values, dpsi_values


def build_dense_rk4_solution(x_values, psi_values):
    """Build the cubic spline through the dense integrated polyline once,
    and package it together with the raw arrays."""
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


def solve_isothermal_shooting_rk4(n_nodes=200, R=DEFAULT_R, lam=1.0, max_iter=25, tol=1e-9, eps=1e-8):
    """Solve the lambda-parametrized isothermal Lane-Emden IVP by RK4
    outward integration with step-doubling self-convergence.

    Returns
    -------
    x_nodes           : fixed reporting grid (n_nodes points on [0, R])
    solution          : converged psi values at x_nodes
    history           : list of psi-at-x_nodes arrays, one per
                         step-doubling refinement (coarsest to finest)
    step_history      : list of step sizes h used at each refinement
    nstep_history     : list of RK4 step counts (n_steps) used at each refinement
    dense_rk4_solution: (x_values, psi_values, cs) bundle from the final,
                         converged RK4 march, with a pre-built cubic spline
    n_iterations      : number of step-doubling refinements actually performed
    converged         : True if the step-doubling loop met `tol` before max_iter
    """
    if n_nodes < 2:
        raise ValueError(f'n_nodes must be at least 2, got {n_nodes}.')
    if R <= 0:
        raise ValueError(f'R must be positive, got {R}.')
    if tol <= 0:
        raise ValueError(f'tol must be positive, got {tol}.')
    if eps <= 0:
        raise ValueError(f'eps must be positive, got {eps}.')
    if max_iter <= 0:
        raise ValueError(f'max_iter must be positive, got {max_iter}.')

    x_nodes = np.linspace(0.0, R, n_nodes)

    n_steps = 32
    history = []
    step_history = []
    nstep_history = []
    prev_solution = None
    x_values = psi_values = None
    converged = False

    for _ in range(max_iter):
        x_values, psi_values, dpsi_values = rk4_integrate(n_steps, R, lam, eps=eps)
        h = x_values[1] - x_values[0]
        solution_at_nodes = np.interp(x_nodes, x_values, psi_values, left=0.0)
        solution_at_nodes[x_nodes < eps] = 0.0

        history.append(solution_at_nodes.copy())
        step_history.append(h)
        nstep_history.append(n_steps)

        if prev_solution is not None and np.max(np.abs(solution_at_nodes - prev_solution)) < tol:
            converged = True
            break
        prev_solution = solution_at_nodes
        n_steps *= 2

    solution = history[-1]
    dense_rk4_solution = build_dense_rk4_solution(x_values, psi_values)
    n_iterations = len(history)
    return x_nodes, solution, history, step_history, nstep_history, dense_rk4_solution, n_iterations, converged


def solve_isothermal_rk4(lam, n_nodes=41, R=DEFAULT_R, max_iter=25, tol=1e-9, eps=1e-8, verbose=False):
    """RK4 analogue of solve_isothermal_fdm: builds the step-doubling
    refinement "history" in place of the Newton-iteration history, and
    returns a tuple whose first five entries mirror the FDM script's
    (nodes, c, history, it, converged) shape, plus the dense RK4 spline
    bundle needed for the "Direct/Analytical" figures.
    """
    x_nodes, solution, history, step_history, nstep_history, dense_rk4_solution, n_iterations, converged = \
        solve_isothermal_shooting_rk4(n_nodes=n_nodes, R=R, lam=lam, max_iter=max_iter, tol=tol, eps=eps)

    if verbose:
        print(f'    lam={lam}: {n_iterations} step-doubling refinements  final h={step_history[-1]:.3e}  '
              f'final n_steps={nstep_history[-1]}')
    if not converged:
        warnings.warn(f'RK4 step-doubling refinement for lam={lam}, n_nodes={n_nodes} did not reach '
                       f'tol={tol:.1e} within {max_iter} refinements.')

    return x_nodes, solution, history, n_iterations, converged, dense_rk4_solution


def get_true_reference(x_eval, lam, R=DEFAULT_R, eps=1e-8):
    """High-accuracy reference: implicit Radau integration of the exact
    ODE (independent of the RK4 step-doubling construction), lambda-
    parametrized. Cached per (lam, R, eps) as a dense_output interpolant."""
    key = (float(lam), R, eps)
    if key not in _true_reference_cache:
        sol = solve_ivp(lambda x, y: isothermal_rhs(x, y, lam), [eps, R],
                         [(lam / 6.0) * eps ** 2, (lam / 3.0) * eps],
                         method='Radau', rtol=1e-13, atol=1e-14, dense_output=True)
        _true_reference_cache[key] = sol
    sol = _true_reference_cache[key]
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    x_safe = np.clip(x_eval, eps, R)
    psi_ref = sol.sol(x_safe)[0]
    return np.where(x_eval < eps, 0.0, psi_ref)


_true_reference_cache = {}
_self_conv_cache = {}


def get_self_convergence_reference(lam, x_eval, R=DEFAULT_R, n_nodes_fine=160, tol=1e-9, eps=1e-8):
    """Approximate reference: the RK4 solver run at a very fine
    resolution (n_nodes_fine=160, its own step-doubling refinement to
    convergence), evaluated on the requested grid via the pre-built
    cubic-spline `dense_evaluate_rk4` (rather than a mesh-refined FDM
    solve). Cached per lambda since the fine solve is reused across the
    nodal grid and the dense grid."""
    key = (float(lam), n_nodes_fine, R, tol, eps)
    if key not in _self_conv_cache:
        _, _, _, _, _, dense_rk4_solution_fine, _, _ = \
            solve_isothermal_shooting_rk4(n_nodes=n_nodes_fine, R=R, lam=lam, tol=tol, eps=eps)
        _self_conv_cache[key] = dense_rk4_solution_fine
    dense_rk4_solution_fine = _self_conv_cache[key]
    return dense_evaluate_rk4(dense_rk4_solution_fine, x_eval)


def rel_err(Y, ref):
    scale = max(np.max(np.abs(ref)), 1e-12)
    return np.abs(Y - ref) / scale


def surf(ax, X, Y, Z, cmap, xlabel, ylabel, zlabel, title, int_y=False):
    s = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='k', alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f'{TAG} - {title}')
    if int_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return s


if __name__ == '__main__':
    lam_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    R = DEFAULT_R
    print(f'Fixed domain: R = {R:.4f}')

    n_nodes_main = 41  # matches FDM's n_elem=40 -> 41 nodes
    x_dense = np.linspace(0.0, R, 200)
    results = {}

    for lam in lam_values:
        nodes, c, history, nit, converged, dense_rk4_solution = solve_isothermal_rk4(lam, n_nodes=n_nodes_main, R=R)
        ref_hi = get_true_reference(nodes, lam, R=R)
        ref_self = get_self_convergence_reference(lam, nodes, R=R)

        Y_dense = dense_evaluate_rk4(dense_rk4_solution, x_dense)
        ref_hi_dense = get_true_reference(x_dense, lam, R=R)
        ref_self_dense = get_self_convergence_reference(lam, x_dense, R=R)

        results[lam] = dict(nodes=nodes, Y=c, history=history, ref_hi=ref_hi, ref_self=ref_self,
                             Y_dense=Y_dense, ref_hi_dense=ref_hi_dense, ref_self_dense=ref_self_dense)

        conv_tag = 'OK' if converged else 'NOT CONVERGED'
        print(f'lam={lam}: {nit} step-doubling refinements [{conv_tag}] | err(nodes) vs hi-acc ref={np.max(np.abs(c - ref_hi)):.3e} '
              f'| err(nodes) vs self-conv ref={np.max(np.abs(c - ref_self)):.3e} '
              f'| err(dense) vs hi-acc ref={np.max(np.abs(Y_dense - ref_hi_dense)):.3e} '
              f'| err(dense) vs self-conv ref={np.max(np.abs(Y_dense - ref_self_dense)):.3e}')

    nodes_common = results[lam_values[0]]['nodes']
    Xg, Mg = np.meshgrid(nodes_common, lam_values)
    Xd, Md = np.meshgrid(x_dense, lam_values)
    Zfinal = np.array([results[lam]['Y'] for lam in lam_values])

    # 01: Final Nodal Values
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (lambda)', 'Final nodal values', 'Final Nodal Values (RK4)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    fig01_path = f'{OUTDIR}/01_exact_numerical_final_nodal_values.png'
    plt.savefig(fig01_path)
    plt.close(fig)

    # 02: Final Relative True Error
    Zerr_hi = np.array([rel_err(results[lam]['Y'], results[lam]['ref_hi']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_hi, 'plasma', 'x value', 'Parameter value (lambda)', 'Final relative error vs. reference', 'Final Relative Error vs. Reference Solution (RK4)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/02_exact_numerical_final_relative_true_error.png')
    plt.close(fig)

    # 03 & 04: Step-Doubling Refinement Nodal Values and True Error (10 figures generated here)
    for lam in lam_values:
        history = results[lam]['history']
        ref_hi = results[lam]['ref_hi']
        refinements = np.arange(len(history))
        Xi, Ii = np.meshgrid(nodes_common, refinements)
        Zval = np.array(history)
        Zerr = np.array([rel_err(hh, ref_hi) for hh in history])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Step-doubling refinement', 'Iterative nodal values', f'Step-Doubling Nodal Values (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/03_exact_numerical_iterative_values_p_{lam}.png')
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Step-doubling refinement', 'Iterative relative error vs. reference', f'Step-Doubling Relative Error vs. Reference (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/04_exact_numerical_iterative_true_error_p_{lam}.png')
        plt.close(fig)

    # 05: Direct Analytical Nodal Values
    Zdense_val = np.array([results[lam]['Y_dense'] for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (lambda)', 'Nodal values (direct cubic-spline reconstruction)', 'Direct/Analytical Evaluation of the RK4 Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    fig05_path = f'{OUTDIR}/05_exact_analytical_nodal_values.png'
    plt.savefig(fig05_path)
    plt.close(fig)

    # 06: Direct Analytical Relative True Error
    Zdense_err_hi = np.array([rel_err(results[lam]['Y_dense'], results[lam]['ref_hi_dense']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_hi, 'plasma', 'x value', 'Parameter value (lambda)', 'Relative error vs. reference (dense grid)', 'Direct/Analytical Evaluation - Relative Error vs. Reference (dense grid) (RK4)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    fig06_path = f'{OUTDIR}/06_exact_analytical_relative_true_error.png'
    plt.savefig(fig06_path)
    plt.close(fig)

    # 07: No-Exact Final Nodal Values (Copy of 01)
    fig07_path = f'{OUTDIR}/07_no_exact_numerical_final_nodal_values.png'
    shutil.copyfile(fig01_path, fig07_path)

    # 08: Final Relative Approx Error
    Zerr_self = np.array([rel_err(results[lam]['Y'], results[lam]['ref_self']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_self, 'plasma', 'x value', 'Parameter value (lambda)', 'Final relative error vs. self-convergence reference', 'Final Relative Error vs. Self-Convergence Reference (RK4)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/08_no_exact_numerical_final_relative_approx_error.png')
    plt.close(fig)

    # 09 & 10: Step-Doubling Refinement Approx Error (10 figures generated here)
    for lam in lam_values:
        history = results[lam]['history']
        ref_self = results[lam]['ref_self']
        refinements = np.arange(len(history))
        Xi, Ii = np.meshgrid(nodes_common, refinements)
        Zval = np.array(history)
        Zerr = np.array([rel_err(hh, ref_self) for hh in history])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Step-doubling refinement', 'Iterative nodal values', f'Step-Doubling Nodal Values (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/09_no_exact_numerical_iterative_values_p_{lam}.png')
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Step-doubling refinement', 'Iterative relative error vs. self-convergence reference', f'Step-Doubling Relative Error vs. Self-Convergence Reference (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/10_no_exact_numerical_iterative_approx_error_p_{lam}.png')
        plt.close(fig)

    # 11: No-Exact Analytical Nodal Values (Copy of 05)
    fig11_path = f'{OUTDIR}/11_no_exact_analytical_nodal_values.png'
    shutil.copyfile(fig05_path, fig11_path)

    # 12: Direct Analytical Relative Approx Error
    Zdense_err_self = np.array([rel_err(results[lam]['Y_dense'], results[lam]['ref_self_dense']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_self, 'plasma', 'x value', 'Parameter value (lambda)', 'Relative error vs. self-convergence reference (dense grid)', 'Direct/Analytical Evaluation - Relative Error vs. Self-Convergence Reference (dense grid) (RK4)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/12_no_exact_analytical_relative_approx_error.png')
    plt.close(fig)

    # Explicitly calculate and print the total figure count
    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f'\nDone. The FULL {n_files}-figure set has been successfully written to {OUTDIR}/ (figures 07 and 11 are file copies of 01 and 05).')

    print('\n' + '=' * 70)
    print('Convergence study (max nodal error vs. Radau reference, fixed-step RK4)')
    print('=' * 70)

    for lam_test in [1.0, 2.5]:
        print(f'\n lambda = {lam_test}')
        print(f"{'n_steps':>8} {'h':>12} {'Max Error':>15} {'Observed Order':>16}")
        prev_err, prev_h = None, None
        for n_steps in [40, 80, 160, 320, 640]:
            x_values_test, psi_values_test, _ = rk4_integrate(n_steps, R, lam_test)
            ref_on_grid = get_true_reference(x_values_test, lam_test, R=R)
            err = np.max(np.abs(psi_values_test - ref_on_grid))
            h = (R - 1e-8) / n_steps
            order = np.log(prev_err / err) / np.log(prev_h / h) if prev_err else np.nan
            order_s = f'{order:.2f}' if not np.isnan(order) else '  --'
            print(f'{n_steps:>8} {h:>12.5f} {err:>15.3e} {order_s:>16}')
            prev_err, prev_h = err, h

    print('\nObserved order ~4 (O(h^4)): the expected classical accuracy of the')
    print('4th-order Runge-Kutta scheme, confirming correct implementation across lambda.')