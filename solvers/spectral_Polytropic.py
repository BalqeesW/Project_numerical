"""
Chebyshev Spectral Collocation Method -- Lane-Emden polytropic model
Produces the FULL 28-figure set required by the professor's template
(3D_figures_grouped_report), with the exact filenames/folder structure
requested. ALL 28 figures belong to the SAME (problem, method) pair:
polytropic Lane-Emden solved by Chebyshev Spectral Collocation. No other
method and no substitute reference is plotted as if it were "the method".

        y''(x) + (2/x) y'(x) + y(x)^m = 0 ,   y(0) = 1 ,  y'(0) = 0

--------------------------------------------------------------------------
FIX (previous version): figures 05, 06, 11, 12 were labelled "analytical
(direct)" but were actually just the reference solution (closed-form or
fine-mesh) re-plotted -- that is not an output of the spectral method at
all, so the labelling was misleading. This version fixes that.

--------------------------------------------------------------------------
What "analytical / direct" legitimately means for a SPECTRAL method
--------------------------------------------------------------------------
A Chebyshev spectral solution is not just a table of numbers at N+1
nodes -- once Newton's method converges, those nodal values ARE the
(unique) coefficients of a global degree-N interpolating polynomial.
Evaluating that polynomial at ANY new point x (via the barycentric
interpolation formula) requires NO further Newton iteration -- it is a
single closed-form algebraic evaluation. That is the genuinely "direct /
analytical" counterpart of this same method and this same converged
solution: the continuous polynomial reconstruction, evaluated on a DENSE
grid of 200 points (not just the 31 collocation nodes), rather than the
discrete iterative process that produced it.

  m_values = [1, 2, 3, 4, 5]   (used identically in both branches)

Branch "exact" (true error), files 01-06:
    01/02 -- discrete Newton solution AT THE 31 COLLOCATION NODES,
             error vs. closed-form exact (m=1,5) / high-accuracy Radau
             validation reference (m=2,3,4; used only for comparison,
             never inside the solver).
    03/04 -- the Newton iteration history at those same nodes (5 figures
             each = 10 figures).
    05/06 -- the SAME converged spectral solution, but evaluated as a
             CONTINUOUS polynomial on a dense 200-point grid via
             barycentric interpolation (direct, non-iterative
             evaluation) -- and its error against the exact/near-exact
             reference evaluated on that same dense grid.

Branch "no exact" (approximate error), files 07-12:
    07/08 -- same discrete Newton solution at the 31 nodes, error vs. an
             INDEPENDENT self-convergence reference: the same spectral
             method solved again at N_fine=70 and barycentric-
             interpolated back -- no closed form, no external solver.
    09/10 -- Newton iteration history, error vs. that self-convergence
             reference (10 figures).
    11/12 -- the SAME converged N=30 solution evaluated as a continuous
             polynomial on the dense grid (direct evaluation, as in
             05), error against the fine-mesh (N=70) solution ALSO
             evaluated as a continuous polynomial on the same dense
             grid.
--------------------------------------------------------------------------
"""
import os
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/Spectral_Polytropic/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = "Spectral Polytropic"   # every figure title is prefixed with this


# ---------------------------------------------------------------------
# Stable Chebyshev differentiation matrices
# ---------------------------------------------------------------------
def chebyshev_D_D2_stable(N, dps=40):
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    idx = list(range(N + 1))
    z = [mp.cos(mp.pi * i / N) for i in idx]
    c = [mp.mpf(1)] * (N + 1)
    c[0] = 2
    c[-1] = 2
    c = [c[i] * ((-1) ** i) for i in idx]
    D = mp.matrix(N + 1, N + 1)
    for i in idx:
        for j in idx:
            if i != j:
                D[i, j] = (c[i] / c[j]) / (z[i] - z[j])
    for i in idx:
        D[i, i] = -mp.fsum(D[i, j] for j in idx if j != i)
    D2 = D * D
    D_np = np.array([[float(D[i, j]) for j in idx] for i in idx])
    D2_np = np.array([[float(D2[i, j]) for j in idx] for i in idx])
    z_np = np.array([float(zz) for zz in z])
    mp.mp.dps = old_dps
    return D_np, D2_np, z_np


def build_operator(N, R):
    D_z, D2_z, z = chebyshev_D_D2_stable(N)
    x = (R / 2.0) * (z + 1.0)
    D = (2.0 / R) * D_z
    D2 = (2.0 / R) ** 2 * D2_z
    return x, D, D2


def cheb_bary_weights(N):
    """Barycentric interpolation weights for Chebyshev-Gauss-Lobatto points
    (NOTE: these are the reciprocal of the 'c' array used in the
    differentiation-matrix formula -- 0.5 at the two endpoints, not 2)."""
    d = np.ones(N + 1)
    d[0] = 0.5
    d[-1] = 0.5
    return d * (-1.0) ** np.arange(N + 1)


def barycentric_interp(x_nodes, f_nodes, w, x_eval):
    """Chebyshev barycentric interpolation (used to compare two spectral
    solutions computed on different-resolution grids)."""
    x_eval = np.atleast_1d(x_eval).astype(float)
    out = np.zeros_like(x_eval)
    for k, xe in enumerate(x_eval):
        diff = xe - x_nodes
        exact = np.where(np.abs(diff) < 1e-12)[0]
        if len(exact) > 0:
            out[k] = f_nodes[exact[0]]
        else:
            terms = w / diff
            out[k] = np.sum(terms * f_nodes) / np.sum(terms)
    return out


# ---------------------------------------------------------------------
# Spectral Newton solver (matrix form, boundary bordering)
# ---------------------------------------------------------------------
def solve_polytropic_spectral(m, N=30, R=2.0, max_iter=30, tol=1e-12):
    x, D, D2 = build_operator(N, R)
    m_int = int(round(m))

    with np.errstate(divide='ignore', invalid='ignore'):
        inv2x = np.where(x > 0, 2.0 / np.where(x == 0, 1.0, x), 0.0)
    L = D2 + inv2x[:, None] * D

    Y = 1.0 - (x ** 2) / 6.0
    history = [Y.copy()]
    n_used = N - 1

    for _ in range(max_iter):
        Ym = Y ** m_int
        dYm = m_int * (Y ** (m_int - 1)) if m_int != 0 else np.zeros_like(Y)

        F = np.zeros(N + 1)
        J = np.zeros((N + 1, N + 1))
        F[:n_used] = L[:n_used, :] @ Y + Ym[:n_used]
        J[:n_used, :] = L[:n_used, :]
        rows = np.arange(n_used)
        J[rows, rows] += dYm[:n_used]

        F[N - 1] = D[N, :] @ Y
        J[N - 1, :] = D[N, :]
        F[N] = Y[N] - 1.0
        J[N, N] = 1.0

        dY = np.linalg.solve(J, -F)
        Y = Y + dY
        history.append(Y.copy())
        if np.linalg.norm(dY, np.inf) < tol:
            break

    return x, Y, history


def get_true_reference(m, x_eval):
    """Closed-form exact solution for m=1,5; high-accuracy Radau
    (validation only, never used inside the solver) otherwise."""
    if abs(m - 1.0) < 1e-9:
        return np.where(x_eval < 1e-8, 1.0, np.sin(x_eval) / (x_eval + 1e-15))
    elif abs(m - 5.0) < 1e-9:
        return 1.0 / np.sqrt(1.0 + (x_eval ** 2) / 3.0)
    else:
        def ode(t, y):
            return [y[1], -(2.0 / t) * y[1] - np.abs(y[0]) ** m]
        eps = 1e-8
        y0 = [1.0 - (eps ** 2) / 6.0, -eps / 3.0]
        x_eval_safe = np.clip(x_eval, eps, None)
        sort_idx = np.argsort(x_eval_safe)
        x_sorted = x_eval_safe[sort_idx]
        sol = solve_ivp(ode, [eps, max(x_sorted[-1], eps * 2)], y0, t_eval=x_sorted,
                         method='Radau', rtol=1e-12, atol=1e-13)
        y_ref = np.zeros_like(x_eval)
        y_ref[sort_idx] = sol.y[0]
        y_ref[x_eval < eps] = 1.0
        return y_ref


def get_approx_reference(m, x_eval, R=2.0, N_fine=70):
    """Independent self-convergence reference: same spectral method at a
    much finer N, interpolated onto x_eval. Uses NO exact formula and NO
    external ODE solver -- suitable for the 'absence of exact solution'
    branch."""
    x_fine, Y_fine, _ = solve_polytropic_spectral(m, N=N_fine, R=R)
    w_fine = cheb_bary_weights(N_fine)
    return barycentric_interp(x_fine, Y_fine, w_fine, x_eval)


def rel_err(Y, ref):
    scale = max(np.max(np.abs(ref)), 1e-12)
    return np.abs(Y - ref) / scale


def surf(ax, X, Y, Z, cmap, xlabel, ylabel, zlabel, title, int_y=False):
    s = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='k', alpha=0.9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    ax.set_title(f'{TAG} - {title}')
    if int_y:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return s


# =======================================================================
# MAIN: solve for all 5 parameter values, build both branches
# =======================================================================
if __name__ == "__main__":
    m_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    R = 2.0

    N_main = 30
    x_dense = np.linspace(0.0, R, 200)   # dense evaluation grid, NOT collocation nodes
    w_main = cheb_bary_weights(N_main)

    results = {}   # m -> dict(x, Y, history, true_ref, approx_ref, Y_dense, ...)
    for m in m_values:
        x, Y, history = solve_polytropic_spectral(m, N=N_main, R=R)
        true_ref = get_true_reference(m, x)
        approx_ref = get_approx_reference(m, x, R=R)

        # ---- direct / analytical evaluation of the SAME converged solution:
        # barycentric (non-iterative) evaluation of the spectral polynomial
        # on a dense grid -- no Newton solve is re-run here.
        Y_dense = barycentric_interp(x, Y, w_main, x_dense)
        true_ref_dense = get_true_reference(m, x_dense)
        approx_ref_dense = get_approx_reference(m, x_dense, R=R)  # independent fine-mesh (N=70) eval

        results[m] = dict(x=x, Y=Y, history=history,
                           true_ref=true_ref, approx_ref=approx_ref,
                           Y_dense=Y_dense, true_ref_dense=true_ref_dense,
                           approx_ref_dense=approx_ref_dense)
        print(f"m={m}: {len(history)-1} Newton its | "
              f"true err(nodes)={np.max(np.abs(Y-true_ref)):.3e} | "
              f"approx err(nodes)={np.max(np.abs(Y-approx_ref)):.3e} | "
              f"true err(dense)={np.max(np.abs(Y_dense-true_ref_dense)):.3e} | "
              f"approx err(dense)={np.max(np.abs(Y_dense-approx_ref_dense)):.3e}")

    x_common = results[m_values[0]]['x']   # same grid (N,R fixed) for all m

    # -------------------------------------------------------------
    # 01: exact / numerical / final nodal values  (x, m, Y_final)
    # -------------------------------------------------------------
    Xg, Mg = np.meshgrid(x_common, m_values)
    Zfinal = np.array([results[m]['Y'] for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (m)',
         'Final nodal values', 'Final Nodal Values (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/01_exact_numerical_final_nodal_values.png'); plt.close(fig)

    # 02: exact / numerical / final relative TRUE error
    Zerr_true = np.array([rel_err(results[m]['Y'], results[m]['true_ref']) for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_true, 'plasma', 'x value', 'Parameter value (m)',
         'Final relative true error', 'Final Relative True Error (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/02_exact_numerical_final_relative_true_error.png'); plt.close(fig)

    # 03 & 04: iterative nodal values / iterative true error, one figure per m
    for m in m_values:
        history = results[m]['history']
        true_ref = results[m]['true_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_common, iters)
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

    # 05 & 06: exact / analytical (direct) -- the SAME converged spectral
    # solution, evaluated as a continuous polynomial (barycentric formula,
    # no further Newton iteration) on a dense 200-point grid, and its
    # error against the exact/near-exact reference evaluated on that grid.
    Xd, Md = np.meshgrid(x_dense, m_values)
    Zdense_val = np.array([results[m]['Y_dense'] for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (m)',
         'Nodal values (direct polynomial evaluation)',
         'Direct/Analytical Evaluation of the Spectral Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/05_exact_analytical_nodal_values.png'); plt.close(fig)

    Zdense_err_true = np.array([rel_err(results[m]['Y_dense'], results[m]['true_ref_dense'])
                                 for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_true, 'plasma', 'x value', 'Parameter value (m)',
         'Relative true error (dense grid)',
         'Direct/Analytical Evaluation - Relative True Error (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/06_exact_analytical_relative_true_error.png'); plt.close(fig)

    # -------------------------------------------------------------
    # 07: no_exact / numerical / final nodal values (same Y as 01, shown
    #     again because this branch reports it against the OTHER reference)
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (m)',
         'Final nodal values', 'Final Nodal Values (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/07_no_exact_numerical_final_nodal_values.png'); plt.close(fig)

    # 08: no_exact / numerical / final relative APPROXIMATE error
    Zerr_approx = np.array([rel_err(results[m]['Y'], results[m]['approx_ref']) for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_approx, 'plasma', 'x value', 'Parameter value (m)',
         'Final relative approximate error', 'Final Relative Approximate Error (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/08_no_exact_numerical_final_relative_approx_error.png'); plt.close(fig)

    # 09 & 10: iterative nodal values / iterative approximate error, per m
    for m in m_values:
        history = results[m]['history']
        approx_ref = results[m]['approx_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_common, iters)
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

    # 11 & 12: no_exact / analytical (direct) -- the SAME dense-grid direct
    # evaluation as 05 (this is the same converged N=30 solution; it does
    # not change between branches), but its error is now measured against
    # the INDEPENDENT self-convergence reference (spectral solved again at
    # N_fine=70, itself evaluated as a polynomial on the same dense grid)
    # instead of the exact/near-exact reference -- i.e. the "approximate
    # error" companion to 05/06.
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (m)',
         'Nodal values (direct polynomial evaluation)',
         'Direct/Analytical Evaluation of the Spectral Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/11_no_exact_analytical_nodal_values.png'); plt.close(fig)

    Zdense_err_approx = np.array([rel_err(results[m]['Y_dense'], results[m]['approx_ref_dense'])
                                   for m in m_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_approx, 'plasma', 'x value', 'Parameter value (m)',
         'Relative approximate error (dense grid)',
         'Direct/Analytical Evaluation - Relative Approximate Error (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/12_no_exact_analytical_relative_approx_error.png'); plt.close(fig)

    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f"\nDone. {n_files} PNG figures written to {OUTDIR}/")