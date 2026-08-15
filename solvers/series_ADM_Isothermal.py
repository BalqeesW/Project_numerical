"""
Adomian Decomposition Method (ADM) -- Isothermal Lane-Emden gas sphere model
(lambda-parametrized, 28-figure output mirroring the FDM script)

        psi''(x) + (2/x) psi'(x) = lambda * exp(-psi(x)) ,  psi(0) = 0 ,  psi'(0) = 0

This is the lambda-parametrized isothermal-gas-sphere form of the Lane-Emden
equation. It is solved entirely with the series-based Adomian Decomposition
Method (ADM): no discretization of x, no linear solve, and no Newton
iteration is ever performed. In this script, "iteration number" in the FDM
sense is replaced by "ADM truncation order N": the ADM partial sum S_N(x)
plays the role that the N-th Newton iterate plays in the FDM script.

--------------------------------------------------------------------------
Method summary (derivation)
--------------------------------------------------------------------------
Multiply by x^2 to remove the coordinate singularity at the origin and
write the equation in self-adjoint (Adomian-operator) form:

        L[psi] := (x^2 psi')' = lambda * x^2 exp(-psi) =: x^2 N(psi)

L is trivially invertible on power series vanishing at x=0 together with
their first derivative:

        L^{-1}[f](x) = Integral_0^x  s^{-2} Integral_0^s  t^2 f(t) dt ds

which, applied to a single monomial, has the closed form used throughout
this script:

        L^{-1}[x^k] = x^(k+2) / [(k+2)(k+3)]

ADM writes the unknown as a series psi = sum_{n=0}^inf psi_n and the
nonlinearity as N(psi) = lambda * sum_{n=0}^inf A_n, where the A_n are the
"Adomian polynomials" of psi_0,...,psi_n for f(u) = exp(-u):

        A_n = (1/n!) d^n/dmu^n [ exp(-sum_i psi_i mu^i) ]  at mu=0

The two initial conditions fix psi_0 = 0. The ADM recursion is already
triangular and exactly satisfies psi(0)=0, psi'(0)=0 term by term, because
L^{-1} always raises the polynomial degree by 2, starting from a genuine
constant/linear-free term:

        psi_0 = 0
        psi_{n+1} = L^{-1}[lambda * A_n] ,   n = 0, 1, 2, ...

which is evaluated in exact rational arithmetic (sympy) so every psi_n is
a closed-form polynomial in x, and the N-term partial sum

        S_N(x) = sum_{n=0}^{N} psi_n(x)

is the ADM approximation to psi(x).

--------------------------------------------------------------------------
Fast recursion for the Adomian polynomials (lambda-parametrized)
--------------------------------------------------------------------------
Since f(u) = exp(-u) satisfies f'(u) = -f(u), if
G(mu) = lambda * f( sum_{i=0}^n psi_i mu^i ) then
G'(mu) = -G(mu) * U'(mu), and matching powers of mu gives the compact,
purely-recursive (Duan-type) formula used below:

        G_0 = lambda                              ( = lambda * A_0 = lambda * exp(-psi_0) )
        n * G_n = - sum_{q=0}^{n-1} (q+1) psi_{q+1} G_{n-1-q} ,   n >= 1

Because f'(u) = -f(u) does not involve lambda, the recursive *shape* is
identical to the lambda=1 case; lambda enters only through the seed G_0,
which is exactly the adaptation required to parametrize the classical
(lambda=1) ADM solution by lambda.

--------------------------------------------------------------------------
Output structure
--------------------------------------------------------------------------
This script reproduces the exact 28-figure template, file-naming
convention, 3D surface-plotting logic, and true/self-convergence
analytical-comparison structure of the FDM isothermal script
(finite_difference_isothermal.py), with these substitutions:
    * n_elem / mesh-refinement families of Newton iterations for  ADM truncation order N (partial sums S_0..S_Nmax)
    * "Iteration number" axis label (figs 03/04/09/10)          for  "ADM truncation order N"
    * dense-grid spline reconstruction (Y_dense)                for  direct ADM series evaluation (no interpolation needed)
    * mesh-refinement self-convergence reference (n_elem_fine)  for  higher-truncation-order self-convergence reference (N_fine)
"""
import os
import shutil
import warnings
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/ADM_Isothermal/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = 'ADM Isothermal'
DEFAULT_R = 2.0

N_MAX = 20    # highest ADM truncation order reported in the "history" figures (mirrors Newton-iteration history)
N_FINE = 30   # extra-order self-convergence reference (mirrors the FDM n_elem_fine mesh-refinement reference)

xsym = sp.symbols('x')


# =======================================================================
# Core ADM machinery (lambda-parametrized)
# =======================================================================
def Linv(poly):
    """L^{-1}[x^k] = x^(k+2) / [(k+2)(k+3)], applied termwise.
    (L[y] = (x^2 y')', the self-adjoint form of y'' + (2/x) y'.)"""
    poly = sp.expand(poly)
    if poly == 0:
        return sp.Integer(0)
    p = sp.Poly(poly, xsym)
    result = sp.Integer(0)
    for monom, coeff in p.terms():
        k = monom[0]
        result += coeff * xsym ** (k + 2) / sp.Rational((k + 2) * (k + 3))
    return sp.expand(result)


def adm_series_isothermal(N_max, lam):
    """
    Builds psi_0..psi_{N_max+1} for
        (x^2 psi')' = lambda * x^2 exp(-psi),  psi(0)=0, psi'(0)=0
    using the fast recursion for the Adomian polynomials of f(u)=exp(-u),
    seeded by lambda:
        G_0 = lambda
        n*G_n = -sum_{q=0}^{n-1} (q+1) psi_{q+1} G_{n-1-q}      (n >= 1)
        psi_{n+1} = L^{-1}[G_n]
    Returns (psi, adomian_polys):
        psi           -- [psi_0, psi_1, ..., psi_{N_max+1}] (sympy polynomials)
        adomian_polys -- [G_0, G_1, ..., G_{N_max}]          (sympy polynomials, = lambda * A_n)
    """
    if N_max < 0:
        raise ValueError('N_max must be non-negative.')

    lam_r = sp.Rational(lam)
    psi = [sp.Integer(0)]              # psi_0 = 0 (from the two ICs)
    adomian_polys = [lam_r]            # G_0 = lambda * A_0 = lambda * exp(-psi_0) = lambda
    psi.append(Linv(adomian_polys[0]))  # psi_1 = L^{-1}[G_0]

    for n in range(1, N_max + 1):
        s = sp.Integer(0)
        for q in range(n):
            s += (q + 1) * psi[q + 1] * adomian_polys[n - 1 - q]
        Gn = sp.expand(-s / n)
        adomian_polys.append(Gn)
        psi.append(Linv(Gn))           # psi_{n+1} = L^{-1}[G_n]

    return psi, adomian_polys


def partial_sums(psi_list):
    """Cumulative sums S_N = sum_{n=0}^N psi_n, N = 0..len(psi_list)-1."""
    out = []
    running = sp.Integer(0)
    for p in psi_list:
        running = sp.expand(running + p)
        out.append(running)
    return out


def build_partial_sum_functions(S_all):
    """Lambdify each partial sum S_N and wrap so every call returns an
    array matching the input shape. S_0 = psi_0 = 0 is a bare sympy
    Integer, so plain lambdify would otherwise return a scalar instead of
    broadcasting when evaluated on an array of x."""
    raw_funcs = [sp.lambdify(xsym, s, 'numpy') for s in S_all]

    def make_eval(f):
        def eval_fn(x_vals):
            x_vals = np.asarray(x_vals, dtype=float)
            return np.broadcast_to(np.asarray(f(x_vals), dtype=float), x_vals.shape).copy()
        return eval_fn

    return [make_eval(f) for f in raw_funcs]


# ---------------------------------------------------------------------
# Per-lambda series cache: builds each lambda's ADM series (up to N_FINE)
# exactly once and reuses it for the main history AND the self-convergence
# reference, so a single sympy build serves every figure for that lambda.
# ---------------------------------------------------------------------
_series_cache = {}


def build_adm_for_lambda(lam, N_max=N_FINE):
    key = (float(lam), N_max)
    if key not in _series_cache:
        psi_list, adomian_polys = adm_series_isothermal(N_max, lam)
        S_all = partial_sums(psi_list)
        S_funcs = build_partial_sum_functions(S_all)
        _series_cache[key] = (psi_list, S_all, S_funcs)
    return _series_cache[key]


def make_mesh(R, n_elem):
    return np.linspace(0.0, R, n_elem + 1)


def solve_isothermal_adm(lam, n_elem=40, R=DEFAULT_R, N_max=N_MAX, N_fine=N_FINE, verbose=False):
    """
    ADM analogue of solve_isothermal_fdm: builds the ADM partial-sum
    "history" S_0(x_nodes), S_1(x_nodes), ..., S_Nmax(x_nodes) in place of
    the Newton-iteration history, and returns the same tuple shape
    (nodes, c, history, it, converged) so the figure-generation block
    below can stay a near-verbatim copy of the FDM script's.
    """
    nodes = make_mesh(R, n_elem)
    _, _, S_funcs = build_adm_for_lambda(lam, N_max=N_fine)

    history = [S_funcs[n](nodes) for n in range(N_max + 1)]
    c = history[-1]
    it = N_max

    # Convergence heuristic (mirrors the FDM residual/step-size check):
    # the ADM partial-sum increments should shrink with N inside the
    # series' radius of convergence. If the last increment is not smaller
    # than the previous one, the chosen domain R may exceed that radius
    # for this lambda, so we warn rather than silently reporting a
    # possibly-divergent value (same spirit as the FDM non-convergence
    # warning, adapted from "residual not shrinking" to "increment not
    # shrinking").
    if N_max >= 2:
        last_inc = np.max(np.abs(history[-1] - history[-2]))
        prev_inc = np.max(np.abs(history[-2] - history[-3]))
        converged = bool(last_inc <= prev_inc)
    else:
        last_inc = prev_inc = np.nan
        converged = True

    if verbose:
        print(f'    lam={lam}: N_max={N_max}  last_increment={last_inc:.3e}  prev_increment={prev_inc:.3e}')
    if not converged:
        warnings.warn(f'ADM partial-sum increments are not shrinking for lam={lam} on R={R} at N_max={N_max} '
                       f'(last={last_inc:.3e}, prev={prev_inc:.3e}); domain may exceed the series radius of convergence.')

    return (nodes, c, history, it, converged)


def get_reference_solution(lam, x_eval):
    """High-accuracy reference: implicit Radau integration of the exact
    ODE (independent of the ADM series construction), lambda-parametrized."""
    def ode(t_, y):
        return [y[1], lam * np.exp(-y[0]) - (2.0 / t_) * y[1]]

    eps = 1e-08
    y0 = [(lam / 6.0) * eps ** 2, (lam / 3.0) * eps]
    x_eval_safe = np.clip(x_eval, eps, None)
    sort_idx = np.argsort(x_eval_safe)
    x_sorted = x_eval_safe[sort_idx]
    sol = solve_ivp(ode, [eps, max(x_sorted[-1], eps * 2)], y0, t_eval=x_sorted, method='Radau', rtol=1e-12, atol=1e-13)

    y_ref = np.zeros_like(x_eval)
    y_ref[sort_idx] = sol.y[0]
    y_ref[x_eval < eps] = 0.0
    return y_ref


def get_self_convergence_reference(lam, x_eval, R=DEFAULT_R, N_fine=N_FINE):
    """Approximate reference: the same ADM series carried to a much
    higher truncation order N_fine (rather than a mesh-refined FDM
    solve), evaluated directly (no interpolation needed)."""
    _, _, S_funcs = build_adm_for_lambda(lam, N_max=N_fine)
    return S_funcs[N_fine](x_eval)


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

    n_elem_main = 40
    x_dense = np.linspace(0.0, R, 200)
    results = {}

    for lam in lam_values:
        nodes, c, history, nit, converged = solve_isothermal_adm(lam, n_elem=n_elem_main, R=R, N_max=N_MAX, N_fine=N_FINE)
        ref_hi = get_reference_solution(lam, nodes)
        ref_self = get_self_convergence_reference(lam, nodes, R=R, N_fine=N_FINE)

        Y_dense = get_self_convergence_reference(lam, x_dense, R=R, N_fine=N_MAX)  # direct ADM eval at N_MAX (dense grid)
        ref_hi_dense = get_reference_solution(lam, x_dense)
        ref_self_dense = get_self_convergence_reference(lam, x_dense, R=R, N_fine=N_FINE)

        results[lam] = dict(nodes=nodes, Y=c, history=history, ref_hi=ref_hi, ref_self=ref_self,
                             Y_dense=Y_dense, ref_hi_dense=ref_hi_dense, ref_self_dense=ref_self_dense)

        conv_tag = 'OK' if converged else 'INCREMENTS NOT SHRINKING'
        print(f'lam={lam}: N_max={nit} ADM terms [{conv_tag}] | err(nodes) vs hi-acc ref={np.max(np.abs(c - ref_hi)):.3e} '
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
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (lambda)', 'Final nodal values', 'Final Nodal Values (ADM)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    fig01_path = f'{OUTDIR}/01_exact_numerical_final_nodal_values.png'
    plt.savefig(fig01_path)
    plt.close(fig)

    # 02: Final Relative True Error
    Zerr_hi = np.array([rel_err(results[lam]['Y'], results[lam]['ref_hi']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_hi, 'plasma', 'x value', 'Parameter value (lambda)', 'Final relative error vs. reference', 'Final Relative Error vs. Reference Solution (ADM)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/02_exact_numerical_final_relative_true_error.png')
    plt.close(fig)

    # 03 & 04: ADM-order Nodal Values and True Error (10 figures generated here)
    for lam in lam_values:
        history = results[lam]['history']
        ref_hi = results[lam]['ref_hi']
        orders = np.arange(len(history))
        Xi, Ii = np.meshgrid(nodes_common, orders)
        Zval = np.array(history)
        Zerr = np.array([rel_err(hh, ref_hi) for hh in history])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'ADM truncation order N', 'ADM partial-sum nodal values', f'ADM Partial-Sum Nodal Values (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/03_exact_numerical_iterative_values_p_{lam}.png')
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'ADM truncation order N', 'ADM relative error vs. reference', f'ADM Relative Error vs. Reference (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/04_exact_numerical_iterative_true_error_p_{lam}.png')
        plt.close(fig)

    # 05: Direct Analytical Nodal Values
    Zdense_val = np.array([results[lam]['Y_dense'] for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (lambda)', 'Nodal values (direct ADM series evaluation)', 'Direct/Analytical Evaluation of the ADM Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    fig05_path = f'{OUTDIR}/05_exact_analytical_nodal_values.png'
    plt.savefig(fig05_path)
    plt.close(fig)

    # 06: Direct Analytical Relative True Error
    Zdense_err_hi = np.array([rel_err(results[lam]['Y_dense'], results[lam]['ref_hi_dense']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_hi, 'plasma', 'x value', 'Parameter value (lambda)', 'Relative error vs. reference (dense grid)', 'Direct/Analytical Evaluation - Relative Error vs. Reference (dense grid) (ADM)')
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
    surf(ax, Xg, Mg, Zerr_self, 'plasma', 'x value', 'Parameter value (lambda)', 'Final relative error vs. self-convergence reference', 'Final Relative Error vs. Self-Convergence Reference (ADM)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/08_no_exact_numerical_final_relative_approx_error.png')
    plt.close(fig)

    # 09 & 10: ADM-order Approx Error (10 figures generated here)
    for lam in lam_values:
        history = results[lam]['history']
        ref_self = results[lam]['ref_self']
        orders = np.arange(len(history))
        Xi, Ii = np.meshgrid(nodes_common, orders)
        Zval = np.array(history)
        Zerr = np.array([rel_err(hh, ref_self) for hh in history])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'ADM truncation order N', 'ADM partial-sum nodal values', f'ADM Partial-Sum Nodal Values (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/09_no_exact_numerical_iterative_values_p_{lam}.png')
        plt.close(fig)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'ADM truncation order N', 'ADM relative error vs. self-convergence reference', f'ADM Relative Error vs. Self-Convergence Reference (lam={lam})', int_y=True)
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
    surf(ax, Xd, Md, Zdense_err_self, 'plasma', 'x value', 'Parameter value (lambda)', 'Relative error vs. self-convergence reference (dense grid)', 'Direct/Analytical Evaluation - Relative Error vs. Self-Convergence Reference (dense grid) (ADM)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/12_no_exact_analytical_relative_approx_error.png')
    plt.close(fig)

    # Explicitly calculate and print the total figure count
    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f'\nDone. The FULL {n_files}-figure set has been successfully written to {OUTDIR}/ (figures 07 and 11 are file copies of 01 and 05).')

    print('\n' + '=' * 70)
    print('Convergence study (max nodal error vs. Radau ODE reference, increasing ADM truncation order N)')
    print('=' * 70)

    for lam_test in [1.0, 2.5]:
        print(f'\n lambda = {lam_test}')
        print(f"{'N (order)':>10} {'Max Error':>15}")
        _, _, S_funcs_test = build_adm_for_lambda(lam_test, N_max=N_FINE)
        ref_test = get_reference_solution(lam_test, nodes_common)
        for N_test in [4, 8, 12, 16, 20]:
            c_test = S_funcs_test[N_test](nodes_common)
            err = np.max(np.abs(c_test - ref_test))
            print(f'{N_test:>10} {err:>15.3e}')

    print("\nErrors should shrink monotonically with N inside the series' radius")
    print('of convergence for a given lambda; growth (rather than shrinkage) at')
    print('the highest N values signals that R=2.0 exceeds the radius of')
    print('convergence for that lambda, which is expected, reported behaviour')
    print('for the isothermal Lane-Emden power series rather than a defect of')
    print('the ADM implementation.')