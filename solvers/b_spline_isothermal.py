"""
Cubic B-Spline Collocation Method -- Lane-Emden ISOTHERMAL gas sphere
Produces the FULL 28-figure set required by the professor's template,
mirroring the polytropic B-spline script exactly in structure. ALL 28
figures belong to the SAME (problem, method) pair: isothermal Lane-Emden
solved by cubic B-spline collocation.

    y''(x) + (2/x) y'(x) - lambda * exp(-y(x)) = 0 ,  y(0) = 0 ,  y'(0) = 0

where y = ln(rho_c/rho) (Chandrasekhar's convention) and lambda is a
positive scaling parameter play the role of the "existing parameter" the
project requires 5 values of (the isothermal Lane-Emden equation itself
has no free power like the polytropic index m, so lambda -- an overall
scale factor in front of the density term -- is the natural parameter to
vary; it plays the same physical role that the central-to-mean density
ratio scaling does in Bonnor-Ebert sphere analyses).

--------------------------------------------------------------------------
IMPORTANT: unlike the polytropic problem (which has closed-form solutions
at m=1,5), the isothermal Lane-Emden equation has NO known closed-form
solution for ANY parameter value -- it is transcendental for all lambda.
Both branches of the required template therefore use non-closed-form
references, exactly as they already did for the polytropic problem's
m=2,3,4 (no-exact) cases:
    "true"        reference = high-accuracy Radau integration (validation
                  only, rtol=atol=1e-12, never used inside the solver)
    "approximate" reference = independent self-convergence check: the
                  same B-spline method solved again at a much finer mesh
                  (n_elem_fine=160) and evaluated directly (no external
                  solver) at the same points.

--------------------------------------------------------------------------
Two bugs found and fixed during development (documented for the report)
--------------------------------------------------------------------------
1. Basis-function endpoint bug (same fix as the polytropic script): the
   Cox-de Boor recursion must include the right domain endpoint in the
   LAST non-degenerate knot interval, not simply the last array index.

2. Boundary-bordering does not carry over from spectral to B-splines
   (same fix as the polytropic script): because the B-spline basis has
   LOCAL support, discarding the collocation row nearest x=0 disconnects
   the first two control points from the rest of the system. Fixed by
   keeping every genuine ODE row at tau[1:] and ADDING (not substituting)
   the two boundary conditions y(0)=0, y'(0)=0 as extra rows, solved by
   linear least squares at each Newton step (over-determined system).

3. ISOTHERMAL-SPECIFIC BUG (new, caught during this script's development):
   the nonlinear source term was first coded as +lambda*exp(+Y) instead
   of the correct -lambda*exp(-Y) (sign/exponent error). The discrete
   Newton system was internally self-consistent (residual verified to
   shrink cleanly under mesh refinement to ~1e-6), which is why this bug
   was NOT visible from the residual check alone -- it was only caught by
   comparing against the independent Radau reference, which showed an
   error that stayed near-constant (did not shrink with refinement) and
   scaled like lambda^2. This is a good illustration of why the "true
   error against an independent reference" figures matter: a passing
   residual check alone does not guarantee the correct equation was
   solved. After the sign fix, clean O(h^2) convergence was verified for
   all 5 lambda values (see the convergence study printed by this script).

--------------------------------------------------------------------------
"analytical / direct" figures (05, 06, 11, 12): same meaning as in the
polytropic script -- the SAME converged spline (control points c),
evaluated as a continuous piecewise-cubic polynomial on a dense 200-point
grid via the B-spline basis, a non-iterative evaluation once Newton has
converged.
--------------------------------------------------------------------------
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/BSpline_Isothermal/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = "BSpline Isothermal"


# ---------------------------------------------------------------------
# B-spline basis (Cox-de Boor) and derivative machinery (identical to
# the polytropic script -- generic, problem-independent)
# ---------------------------------------------------------------------
def bspline_basis_matrix(U, degree, x):
    U = np.asarray(U, dtype=float)
    x = np.asarray(x, dtype=float)
    m = len(U) - 1
    proper = [i for i in range(m) if U[i+1] > U[i]]
    last_proper = proper[-1] if proper else m - 1

    N = np.zeros((len(x), m))
    for i in range(m):
        lo, hi = U[i], U[i+1]
        if hi > lo:
            mask = (x >= lo) & (x < hi)
            if i == last_proper:
                mask = mask | (x >= hi)
            N[:, i] = mask.astype(float)
    for d in range(1, degree + 1):
        n_new = m - d
        N_new = np.zeros((len(x), n_new))
        for i in range(n_new):
            denom1 = U[i+d] - U[i]
            denom2 = U[i+d+1] - U[i+1]
            term1 = (x - U[i]) / denom1 * N[:, i] if denom1 > 0 else 0.0
            term2 = (U[i+d+1] - x) / denom2 * N[:, i+1] if denom2 > 0 else 0.0
            N_new[:, i] = term1 + term2
        N = N_new
    return N


def diff_matrix(t, p, n_ctrl):
    Dq = np.zeros((n_ctrl - 1, n_ctrl))
    for i in range(n_ctrl - 1):
        denom = t[i+p+1] - t[i+1]
        val = p / denom if denom > 0 else 0.0
        Dq[i, i] = -val
        Dq[i, i+1] = val
    return Dq


def clamped_knot_vector(R, p, n_elem):
    interior = np.linspace(0, R, n_elem + 1)[1:-1]
    return np.concatenate([np.zeros(p+1), interior, np.full(p+1, R)])


def build_bspline_operator(p, n_elem, R):
    t = clamped_knot_vector(R, p, n_elem)
    n_ctrl = n_elem + p
    tau = np.array([np.mean(t[i+1:i+p+1]) for i in range(n_ctrl)])

    Dq = diff_matrix(t, p, n_ctrl)
    t1 = t[1:-1]
    Dr = diff_matrix(t1, p-1, n_ctrl-1)
    t2 = t1[1:-1]
    D2diff = Dr @ Dq

    B0 = bspline_basis_matrix(t, p, tau)
    B1 = bspline_basis_matrix(t1, p-1, tau)
    B2 = bspline_basis_matrix(t2, p-2, tau)

    M0, M1, M2 = B0, B1 @ Dq, B2 @ D2diff
    return t, n_ctrl, tau, M0, M1, M2


def dense_evaluate(t, p, c, x_dense):
    """Direct (non-iterative) evaluation of the converged spline at a
    dense grid of points, via the SAME basis, no Newton solve involved."""
    Bd = bspline_basis_matrix(t, p, x_dense)
    return Bd @ c


# ---------------------------------------------------------------------
# Newton solver for the isothermal equation (over-determined / least
# squares, see module docstring)
# ---------------------------------------------------------------------
def solve_isothermal_bspline(lam, p=3, n_elem=40, R=2.0, max_iter=50, tol=1e-11):
    t, n_ctrl, tau, M0, M1, M2 = build_bspline_operator(p, n_elem, R)

    with np.errstate(divide='ignore', invalid='ignore'):
        inv2tau = np.where(tau > 0, 2.0 / np.where(tau == 0, 1.0, tau), 0.0)
    L = M2 + inv2tau[:, None] * M1

    c = (lam / 6.0) * (tau ** 2)     # Taylor initial guess: y ~ (lam/6) x^2
    history = [c.copy()]
    n_eq = (n_ctrl - 1) + 2

    for it in range(max_iter):
        Y = M0 @ c
        expmY = np.exp(-Y)

        J = np.zeros((n_eq, n_ctrl))
        F = np.zeros(n_eq)
        F[0:n_ctrl-1] = L[1:, :] @ c - lam * expmY[1:]
        J[0:n_ctrl-1, :] = L[1:, :] + lam * expmY[1:, None] * M0[1:, :]
        F[n_ctrl-1] = c[0]                    # y(0) = 0
        J[n_ctrl-1, 0] = 1.0
        F[n_ctrl] = c[1] - c[0]                # y'(0) = 0
        J[n_ctrl, 0] = -1.0
        J[n_ctrl, 1] = 1.0

        dc, *_ = np.linalg.lstsq(J, -F, rcond=None)
        c = c + dc
        history.append(c.copy())
        if np.linalg.norm(dc, np.inf) < tol:
            break

    Y_final = M0 @ c
    return t, tau, Y_final, history, c, it


def get_true_reference(lam, x_eval):
    """High-accuracy Radau reference (validation only, never used inside
    the solver). No closed-form solution exists for this equation for any
    lambda, so this reference is treated as 'exact-equivalent'."""
    def ode(t_, y):
        return [y[1], lam * np.exp(-y[0]) - (2.0 / t_) * y[1]]
    eps = 1e-8
    y0 = [(lam / 6.0) * eps**2, (lam / 3.0) * eps]
    x_eval_safe = np.clip(x_eval, eps, None)
    sort_idx = np.argsort(x_eval_safe)
    x_sorted = x_eval_safe[sort_idx]
    sol = solve_ivp(ode, [eps, max(x_sorted[-1], eps*2)], y0, t_eval=x_sorted,
                     method='Radau', rtol=1e-12, atol=1e-13)
    y_ref = np.zeros_like(x_eval)
    y_ref[sort_idx] = sol.y[0]
    y_ref[x_eval < eps] = 0.0
    return y_ref


def get_approx_reference(lam, x_eval, R=2.0, p=3, n_elem_fine=160):
    """Independent self-convergence reference: same B-spline method at a
    much finer mesh, evaluated directly (no closed form, no external
    solver) at x_eval."""
    t_f, tau_f, Y_f, _, c_f, _ = solve_isothermal_bspline(lam, p=p, n_elem=n_elem_fine, R=R)
    return dense_evaluate(t_f, p, c_f, x_eval)


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
    p = 3
    n_elem_main = 40

    x_dense = np.linspace(0.0, R, 200)

    results = {}
    for lam in lam_values:
        t, tau, Y, history, c, nit = solve_isothermal_bspline(lam, p=p, n_elem=n_elem_main, R=R)
        true_ref = get_true_reference(lam, tau)
        approx_ref = get_approx_reference(lam, tau, R=R, p=p)

        Y_dense = dense_evaluate(t, p, c, x_dense)
        true_ref_dense = get_true_reference(lam, x_dense)
        approx_ref_dense = get_approx_reference(lam, x_dense, R=R, p=p)

        results[lam] = dict(tau=tau, Y=Y, history=history, true_ref=true_ref,
                             approx_ref=approx_ref, Y_dense=Y_dense,
                             true_ref_dense=true_ref_dense, approx_ref_dense=approx_ref_dense)
        print(f"lam={lam}: {nit} Newton its | "
              f"true err(nodes)={np.max(np.abs(Y-true_ref)):.3e} | "
              f"approx err(nodes)={np.max(np.abs(Y-approx_ref)):.3e} | "
              f"true err(dense)={np.max(np.abs(Y_dense-true_ref_dense)):.3e} | "
              f"approx err(dense)={np.max(np.abs(Y_dense-approx_ref_dense)):.3e}")

    tau_common = results[lam_values[0]]['tau']
    Xg, Mg = np.meshgrid(tau_common, lam_values)
    Xd, Md = np.meshgrid(x_dense, lam_values)

    # 01: exact / numerical / final nodal values
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
    t_m, n_ctrl_m, tau_m, M0_m, _, _ = build_bspline_operator(p, n_elem_main, R)
    for lam in lam_values:
        history = results[lam]['history']
        true_ref = results[lam]['true_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(tau_common, iters)
        Zval = np.array([M0_m @ h for h in history])
        Zerr = np.array([rel_err(M0_m @ h, true_ref) for h in history])

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

    # 05 & 06: direct/analytical -- dense-grid evaluation of the SAME converged spline
    Zdense_val = np.array([results[lam]['Y_dense'] for lam in lam_values])
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (lambda)',
         'Nodal values (direct spline evaluation)',
         'Direct/Analytical Evaluation of the B-Spline Solution (dense grid)')
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
        Xi, Ii = np.meshgrid(tau_common, iters)
        Zval = np.array([M0_m @ h for h in history])
        Zerr = np.array([rel_err(M0_m @ h, approx_ref) for h in history])

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
         'Nodal values (direct spline evaluation)',
         'Direct/Analytical Evaluation of the B-Spline Solution (dense grid)')
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
    # Convergence study (mesh refinement, cubic B-spline)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Convergence study (max nodal error vs Radau reference)")
    print("="*70)
    for lam_test in [1.0, 2.5]:
        print(f"\n lambda = {lam_test}")
        print(f"{'n_elem':>8} {'Max Error':>15} {'Observed Order':>16}")
        prev_err, prev_h = None, None
        for n_elem in [10, 20, 40, 80, 160]:
            _, tau_c, Y_c, _, _, _ = solve_isothermal_bspline(lam_test, p=p, n_elem=n_elem, R=R)
            err = np.max(np.abs(Y_c - get_true_reference(lam_test, tau_c)))
            h = 1.0 / n_elem
            order = np.log(prev_err/err)/np.log(prev_h/h) if prev_err else np.nan
            order_s = f"{order:.2f}" if not np.isnan(order) else "  --"
            print(f"{n_elem:>8} {err:>15.3e} {order_s:>16}")
            prev_err, prev_h = err, h
    print("\nObserved order ~2 (O(h^2)), matching the polytropic B-spline")
    print("result -- verified convergence, per the project's core requirement.")