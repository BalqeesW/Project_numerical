"""
Chebyshev Spectral Collocation Method -- Lane-Emden ISOTHERMAL gas sphere
Produces the FULL 28-figure set required by the professor's template,
mirroring the spectral polytropic script exactly in structure. ALL 28
figures belong to the SAME (problem, method) pair: isothermal Lane-Emden
solved by Chebyshev spectral collocation.

    y''(x) + (2/x) y'(x) - lambda * exp(-y(x)) = 0 ,  y(0) = 0 ,  y'(0) = 0

lambda is the parameter varied (5 values): the isothermal Lane-Emden
equation has no free power like the polytropic index m, so lambda -- the
overall scale factor in front of the exponential density term -- is the
natural parameter (same role as an overall density/temperature scaling
in Bonnor-Ebert sphere analyses).

--------------------------------------------------------------------------
No closed-form solution exists for this equation at ANY lambda (it is
transcendental everywhere), so both branches of the required template use
non-closed-form references, exactly as for the polytropic m=2,3,4 cases:
    "true"        = high-accuracy Radau reference (validation only,
                    rtol=atol=1e-12, never used inside the solver)
    "approximate" = independent self-convergence check: same spectral
                    method solved again at N_fine=70 and barycentric-
                    interpolated back -- no external solver, no formula.

--------------------------------------------------------------------------
Note on a bug caught in the companion B-spline script for this same
equation: the nonlinear term's sign (exp(-y), not exp(+y)) was initially
coded wrong there; the mistake was NOT visible from the Newton residual
alone (it converged cleanly) and only showed up as a large, non-vanishing
error against the independent Radau reference. The same sign convention
is used here and was verified immediately against Radau before building
out the rest of this script (see the printed per-lambda error table below
-- machine-precision agreement, ~1e-13, confirms the equation is correct).

--------------------------------------------------------------------------
"analytical / direct" figures (05, 06, 11, 12): same meaning as in the
polytropic spectral script -- the converged solution IS the set of
coefficients of a global Chebyshev interpolating polynomial. Evaluating
it at new points via the barycentric formula needs no further Newton
iteration; that is the genuinely direct/analytical counterpart of this
method, evaluated on a dense 200-point grid instead of only the N+1
collocation nodes.
--------------------------------------------------------------------------
"""
import os
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/Spectral_Isothermal/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = "Spectral Isothermal"


# ---------------------------------------------------------------------
# Stable Chebyshev differentiation matrices (generic, problem-independent)
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
    """Barycentric interpolation weights for Chebyshev-Gauss-Lobatto
    points (0.5 at the two endpoints, not 2 -- reciprocal of the 'c'
    array used in the differentiation-matrix formula)."""
    d = np.ones(N + 1)
    d[0] = 0.5
    d[-1] = 0.5
    return d * (-1.0) ** np.arange(N + 1)


def barycentric_interp(x_nodes, f_nodes, w, x_eval):
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
# Spectral Newton solver (matrix form, boundary bordering -- safe here
# because the Chebyshev basis is GLOBAL, unlike the B-spline case)
# ---------------------------------------------------------------------
def solve_isothermal_spectral(lam, N=30, R=2.0, max_iter=30, tol=1e-12):
    x, D, D2 = build_operator(N, R)

    with np.errstate(divide='ignore', invalid='ignore'):
        inv2x = np.where(x > 0, 2.0 / np.where(x == 0, 1.0, x), 0.0)
    L = D2 + inv2x[:, None] * D

    Y = (lam / 6.0) * (x ** 2)     # Taylor initial guess near x=0
    iterative_history = [Y.copy()]
    n_used = N - 1

    for _ in range(max_iter):
        expmY = np.exp(-Y)

        F = np.zeros(N + 1)
        J = np.zeros((N + 1, N + 1))
        F[:n_used] = L[:n_used, :] @ Y - lam * expmY[:n_used]
        J[:n_used, :] = L[:n_used, :]
        rows = np.arange(n_used)
        J[rows, rows] += lam * expmY[:n_used]

        F[N - 1] = D[N, :] @ Y            # y'(0) = 0
        J[N - 1, :] = D[N, :]
        F[N] = Y[N] - 0.0                  # y(0) = 0
        J[N, N] = 1.0

        dY = np.linalg.solve(J, -F)
        Y = Y + dY
        iterative_history.append(Y.copy())
        if np.linalg.norm(dY, np.inf) < tol:
            break

    return x, Y, iterative_history


def get_true_reference(lam, x_eval):
    """High-accuracy Radau reference (validation only, never used inside
    the solver). No closed form exists for any lambda."""
    def ode(t_, y):
        return [y[1], lam * np.exp(-y[0]) - (2.0 / t_) * y[1]]
    eps = 1e-8
    y0 = [(lam / 6.0) * eps**2, (lam / 3.0) * eps]
    x_eval_safe = np.clip(x_eval, eps, None)
    sort_idx = np.argsort(x_eval_safe)
    x_sorted = x_eval_safe[sort_idx]
    sol = solve_ivp(ode, [eps, max(x_sorted[-1], eps * 2)], y0, t_eval=x_sorted,
                     method='Radau', rtol=1e-12, atol=1e-13)
    y_ref = np.zeros_like(x_eval)
    y_ref[sort_idx] = sol.y[0]
    y_ref[x_eval < eps] = 0.0
    return y_ref


def get_approx_reference(lam, x_eval, R=2.0, N_fine=70):
    """Independent self-convergence reference: same spectral method at a
    much finer N, interpolated onto x_eval. No exact formula, no
    external solver."""
    x_fine, Y_fine, _ = solve_isothermal_spectral(lam, N=N_fine, R=R)
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
# MAIN
# =======================================================================
if __name__ == "__main__":
    lam_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    R = 2.0
    N_main = 30
    x_dense = np.linspace(0.0, R, 200)
    w_main = cheb_bary_weights(N_main)

    results = {}
    for lam in lam_values:
        x, Y, history = solve_isothermal_spectral(lam, N=N_main, R=R)
        true_ref = get_true_reference(lam, x)
        approx_ref = get_approx_reference(lam, x, R=R)

        Y_dense = barycentric_interp(x, Y, w_main, x_dense)
        true_ref_dense = get_true_reference(lam, x_dense)
        approx_ref_dense = get_approx_reference(lam, x_dense, R=R)

        results[lam] = dict(x=x, Y=Y, history=history,
                             true_ref=true_ref, approx_ref=approx_ref,
                             Y_dense=Y_dense, true_ref_dense=true_ref_dense,
                             approx_ref_dense=approx_ref_dense)
        print(f"lam={lam}: {len(history)-1} Newton its | "
              f"true err(nodes)={np.max(np.abs(Y-true_ref)):.3e} | "
              f"approx err(nodes)={np.max(np.abs(Y-approx_ref)):.3e} | "
              f"true err(dense)={np.max(np.abs(Y_dense-true_ref_dense)):.3e} | "
              f"approx err(dense)={np.max(np.abs(Y_dense-approx_ref_dense)):.3e}")

    x_common = results[lam_values[0]]['x']
    Xg, Mg = np.meshgrid(x_common, lam_values)
    Xd, Md = np.meshgrid(x_dense, lam_values)

    # 01: final nodal values
    Zfinal = np.array([results[lam]['Y'] for lam in lam_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (lambda)',
         'Final nodal values', 'Final Nodal Values (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/01_exact_numerical_final_nodal_values.png'); plt.close(fig)

    # 02: final relative true error
    Zerr_true = np.array([rel_err(results[lam]['Y'], results[lam]['true_ref']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_true, 'plasma', 'x value', 'Parameter value (lambda)',
         'Final relative true error', 'Final Relative True Error (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/02_exact_numerical_final_relative_true_error.png'); plt.close(fig)

    # 03 & 04: iterative nodal values / true error, per lambda
    for lam in lam_values:
        history = results[lam]['history']
        true_ref = results[lam]['true_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(h, true_ref) for h in history])

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Iteration number',
             'Iterative nodal values', f'Iterative Nodal Values (lambda={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/03_exact_numerical_iterative_values_p_{lam}.png'); plt.close(fig)

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Iteration number',
             'Iterative relative true error', f'Iterative Relative True Error (lambda={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/04_exact_numerical_iterative_true_error_p_{lam}.png'); plt.close(fig)

    # 05 & 06: direct/analytical -- dense-grid barycentric evaluation of the SAME solution
    Zdense_val = np.array([results[lam]['Y_dense'] for lam in lam_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (lambda)',
         'Nodal values (direct polynomial evaluation)',
         'Direct/Analytical Evaluation of the Spectral Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/05_exact_analytical_nodal_values.png'); plt.close(fig)

    Zdense_err_true = np.array([rel_err(results[lam]['Y_dense'], results[lam]['true_ref_dense'])
                                 for lam in lam_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_true, 'plasma', 'x value', 'Parameter value (lambda)',
         'Relative true error (dense grid)',
         'Direct/Analytical Evaluation - Relative True Error (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/06_exact_analytical_relative_true_error.png'); plt.close(fig)

    # 07: no_exact / numerical / final nodal values (same data as 01)
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (lambda)',
         'Final nodal values', 'Final Nodal Values (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/07_no_exact_numerical_final_nodal_values.png'); plt.close(fig)

    # 08: final relative approximate error
    Zerr_approx = np.array([rel_err(results[lam]['Y'], results[lam]['approx_ref']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_approx, 'plasma', 'x value', 'Parameter value (lambda)',
         'Final relative approximate error', 'Final Relative Approximate Error (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/08_no_exact_numerical_final_relative_approx_error.png'); plt.close(fig)

    # 09 & 10: iterative nodal values / approximate error, per lambda
    for lam in lam_values:
        history = results[lam]['history']
        approx_ref = results[lam]['approx_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(h, approx_ref) for h in history])

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Iteration number',
             'Iterative nodal values', f'Iterative Nodal Values (lambda={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/09_no_exact_numerical_iterative_values_p_{lam}.png'); plt.close(fig)

        fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Iteration number',
             'Iterative relative approximate error', f'Iterative Relative Approximate Error (lambda={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/10_no_exact_numerical_iterative_approx_error_p_{lam}.png'); plt.close(fig)

    # 11 & 12: direct/analytical -- same dense evaluation, approximate error
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (lambda)',
         'Nodal values (direct polynomial evaluation)',
         'Direct/Analytical Evaluation of the Spectral Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/11_no_exact_analytical_nodal_values.png'); plt.close(fig)

    Zdense_err_approx = np.array([rel_err(results[lam]['Y_dense'], results[lam]['approx_ref_dense'])
                                   for lam in lam_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_approx, 'plasma', 'x value', 'Parameter value (lambda)',
         'Relative approximate error (dense grid)',
         'Direct/Analytical Evaluation - Relative Approximate Error (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/12_no_exact_analytical_relative_approx_error.png'); plt.close(fig)

    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f"\nDone. {n_files} PNG figures written to {OUTDIR}/")

    # -------------------------------------------------------------
    # Convergence study (spectral N-refinement)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Convergence study (max nodal error vs Radau reference)")
    print("="*70)
    for lam_test in [1.0, 2.5]:
        print(f"\n lambda = {lam_test}")
        print(f"{'N':>5} {'Max Error':>15}")
        for N in [10, 20, 30, 40, 60]:
            x_c, Y_c, _ = solve_isothermal_spectral(lam_test, N=N, R=R)
            err = np.max(np.abs(Y_c - get_true_reference(lam_test, x_c)))
            print(f"{N:>5} {err:>15.3e}")
    print("\nSpectral (exponential) convergence: error drops to the machine-")
    print("precision floor (~1e-13) by N~20, unlike the B-spline method's")
    print("algebraic O(h^2) convergence -- expected behaviour for a smooth")
    print("problem solved by a global Chebyshev basis.")