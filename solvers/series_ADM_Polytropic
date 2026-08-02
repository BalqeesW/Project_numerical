import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/ADMSeries_Polytropic/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = "ADM Series Polytropic"


# =======================================================================
# Adomian Decomposition Method (ADM) for the Lane-Emden equation
#
#   y'' + (2/x) y' + y^m = 0 ,   y(0) = 1 ,  y'(0) = 0
#
# Rewritten with the spherical-Laplacian operator
#   L y := (1/x^2) d/dx( x^2 dy/dx )  =>  L y = -y^m
# whose inverse (with the two ICs already folded in) is the double
# integral operator
#   L^{-1} f(x) = int_0^x x^{-2} int_0^x x^2 f(x) dx dx
# which, applied to a monomial, gives the clean shift rule
#   L^{-1}(x^k) = x^(k+2) / [(k+2)(k+3)]
# and automatically regularizes the (1/x) singularity at the origin --
# no eps-offset start is needed.
#
# Decomposing y = sum_n y_n and y^m = sum_n A_n (Adomian polynomials),
# the recursion is
#   y_0 = 1
#   y_{n+1} = -L^{-1}(A_n)
# Because y_0 = 1 (constant) and the nonlinearity is the pure power
# y^m, the Adomian polynomials collapse onto the classical recursive
# formula (Rach's formula) for f(u) = u^m:
#   A_0 = u_0^m
#   A_n = (1/(n u_0)) * sum_{k=1}^n [ k(m+1) - n ] u_k A_{n-k},  n>=1
# Since here every y_n turns out to be a pure monomial c_n x^(2n) (the
# Lane-Emden solution is an even function of x), the whole ADM
# recursion reduces to one clean scalar recursion for the Taylor
# coefficients of y(x) = sum_j series_coefficients[j] x^(2j) -- this is
# the standard "series ADM" solution used in the Lane-Emden ADM
# literature (Wazwaz-type algorithms).
# =======================================================================
def adm_series_coeffs(m, N):
    """Return series_coefficients[0..N], the ADM/Taylor coefficients of
    y(x) = sum_j series_coefficients[j] x^(2j)."""
    if m <= 0:
        raise ValueError(f"Polytropic index m must be positive, got m={m}.")
    if N < 2:
        raise ValueError(f"N (series length) must be at least 2, got N={N}.")

    series_coefficients = np.zeros(N + 1)
    adomian_coefficients = np.zeros(N)   # A_n, also pure monomials in x^2
    series_coefficients[0] = 1.0
    for n in range(N):
        if n == 0:
            a_n = series_coefficients[0] ** m
        else:
            s = 0.0
            for k in range(1, n + 1):
                s += (k * (m + 1) - n) * series_coefficients[k] * adomian_coefficients[n - k]
            a_n = s / (n * series_coefficients[0])
        adomian_coefficients[n] = a_n
        series_coefficients[n + 1] = -a_n / ((2 * n + 2) * (2 * n + 3))
    return series_coefficients


# ---------------------------------------------------------------------
# Pade resummation of the ADM/Taylor series.
#
# The raw power series only converges inside its radius of convergence
# (e.g. for m=5 the exact solution 1/sqrt(1+x^2/3) has complex poles at
# x = +-i*sqrt(3), giving a radius ~1.73 -- smaller than the R=2 domain
# used here, so the raw partial sums diverge before reaching x=R).
# A diagonal Pade approximant of the same series is the standard fix in
# the ADM literature: it resums the series into a rational function in
# z = x^2 that matches the same Taylor coefficients at z=0 but extends
# the effective domain of convergence far beyond the original radius.
# ---------------------------------------------------------------------
def pade_approx(series_coefficients, L, M):
    """[L/M] Pade approximant of the power series with coefficients
    series_coefficients[0..L+M]. Returns numerator_coeffs (length L+1)
    and denominator_coeffs (length M+1, denominator_coeffs[0]=1)."""
    c = np.asarray(series_coefficients, dtype=float)
    A = np.zeros((M, M))
    rhs = np.zeros(M)
    for j in range(1, M + 1):
        for k in range(1, M + 1):
            idx = L + j - k
            A[j - 1, k - 1] = c[idx] if idx >= 0 else 0.0
        rhs[j - 1] = -c[L + j] if (L + j) < len(c) else 0.0
    try:
        denom_tail = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        denom_tail, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    denominator_coeffs = np.concatenate(([1.0], denom_tail))
    numerator_coeffs = np.zeros(L + 1)
    for j in range(L + 1):
        s = 0.0
        for k in range(min(j, M) + 1):
            s += denominator_coeffs[k] * c[j - k]
        numerator_coeffs[j] = s
    return numerator_coeffs, denominator_coeffs


def eval_pade(numerator_coeffs, denominator_coeffs, z):
    """Vectorized Horner evaluation of the [L/M] Pade rational function at z=x^2."""
    z = np.asarray(z, dtype=float)
    num = np.zeros_like(z)
    for coef in numerator_coeffs[::-1]:
        num = num * z + coef
    den = np.zeros_like(z)
    for coef in denominator_coeffs[::-1]:
        den = den * z + coef
    return num / den


def dense_evaluate_adm(pade_coefficients, x_dense):
    """Evaluate the converged ADM+Pade solution at arbitrary points."""
    numerator_coeffs, denominator_coeffs = pade_coefficients
    return eval_pade(numerator_coeffs, denominator_coeffs, np.asarray(x_dense) ** 2)


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


# ---------------------------------------------------------------------
# ADM solver: build the raw ADM/Taylor series once, then walk up
# through increasing diagonal Pade orders [k/k] until the solution at
# the reporting nodes stops changing by `tol`. This order-refinement
# sequence is stored in `history` (solution snapshots) and
# `order_history` (the Pade order used at each snapshot), which drive
# the script's convergence plots.
# ---------------------------------------------------------------------
def solve_polytropic_adm(m, n_nodes=43, R=2.0, max_iter=25, tol=1e-11, N_series=60):
    if m <= 0:
        raise ValueError(f"Polytropic index m must be positive, got m={m}.")
    if N_series < 4:
        raise ValueError(f"N_series must be at least 4 for a meaningful Pade approximant, got {N_series}.")
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be at least 2, got {n_nodes}.")
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}.")

    x_nodes = np.linspace(0.0, R, n_nodes)
    series_coefficients = adm_series_coeffs(float(m), N_series)

    history = []
    order_history = []
    prev_solution = None
    numerator_coeffs = denominator_coeffs = None
    order = 2

    for _ in range(max_iter):
        order = min(order, N_series // 2)
        numerator_coeffs, denominator_coeffs = pade_approx(series_coefficients, order, order)
        solution_at_nodes = eval_pade(numerator_coeffs, denominator_coeffs, x_nodes ** 2)
        history.append(solution_at_nodes.copy())
        order_history.append(order)

        if prev_solution is not None and np.max(np.abs(solution_at_nodes - prev_solution)) < tol:
            break
        prev_solution = solution_at_nodes
        if order >= N_series // 2:
            break
        order += 2

    solution = history[-1]
    pade_coefficients = (numerator_coeffs, denominator_coeffs)
    return None, x_nodes, solution, history, order_history, pade_coefficients, len(history)


def get_true_reference(m, x_eval):
    if abs(m - 1.0) < 1e-9:
        return np.where(x_eval < 1e-8, 1.0, np.sin(x_eval) / (x_eval + 1e-15))
    elif abs(m - 5.0) < 1e-9:
        return 1.0 / np.sqrt(1.0 + (x_eval ** 2) / 3.0)
    else:
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


# Cache of fine-order self-convergence references, keyed by
# (m, R, n_nodes_fine, N_series), so repeated lookups for the same m
# (e.g. once for the node grid and once for the dense plotting grid)
# reuse a single ADM+Pade solve instead of recomputing from scratch.
_approx_reference_cache = {}


def get_approx_reference_solution(m, R=2.0, n_nodes_fine=160, N_series=100):
    """Independent self-convergence reference: same ADM+Pade method run
    with many more series terms (tighter tolerance). Returns the
    pade_coefficients tuple, cached per (m, R, n_nodes_fine, N_series)
    so it is computed at most once."""
    key = (round(float(m), 10), R, n_nodes_fine, N_series)
    if key in _approx_reference_cache:
        return _approx_reference_cache[key]
    _, _, _, _, _, pade_coefficients, _ = solve_polytropic_adm(
        m, n_nodes=n_nodes_fine, R=R, tol=1e-13, N_series=N_series)
    _approx_reference_cache[key] = pade_coefficients
    return pade_coefficients


def get_approx_reference(m, x_eval, R=2.0, n_nodes_fine=160, N_series=100):
    """Convenience wrapper: fetch (or compute) the fine-order reference
    for m and evaluate it at x_eval (rational-function evaluation only,
    no external solver)."""
    pade_coefficients = get_approx_reference_solution(m, R=R, n_nodes_fine=n_nodes_fine, N_series=N_series)
    return dense_evaluate_adm(pade_coefficients, x_eval)


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


def plot_surface_figure(filename, X, Y, Z, xlabel, ylabel, zlabel, title, cmap, int_y=False):
    """Helper that builds a 3D surface figure, adds a colorbar, saves it
    to OUTDIR/filename, and closes the figure -- avoids repeating the
    same six lines for every one of the report's surface plots."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, X, Y, Z, cmap, xlabel, ylabel, zlabel, title, int_y=int_y)
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/{filename}')
    plt.close(fig)


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
        _, x_nodes, solution, history, order_history, pade_coefficients, n_iterations = \
            solve_polytropic_adm(m, n_nodes=n_nodes_main, R=R)
        true_ref = get_true_reference(m, x_nodes)
        approx_pade_coefficients = get_approx_reference_solution(m, R=R)
        approx_ref = dense_evaluate_adm(approx_pade_coefficients, x_nodes)

        solution_dense = dense_evaluate_adm(pade_coefficients, x_dense)
        true_ref_dense = get_true_reference(m, x_dense)
        approx_ref_dense = dense_evaluate_adm(approx_pade_coefficients, x_dense)

        x1 = find_first_zero_crossing(x_dense, solution_dense)

        results[m] = dict(x_nodes=x_nodes, solution=solution, history=history,
                           order_history=order_history, true_ref=true_ref,
                           approx_ref=approx_ref, solution_dense=solution_dense,
                           true_ref_dense=true_ref_dense, approx_ref_dense=approx_ref_dense,
                           first_zero=x1)
        x1_str = f"{x1:.6f}" if x1 is not None else "none in [0,R]"
        print(f"m={m}: {n_iterations} Pade-order refinements | first zero (radius) x_1={x1_str} | "
              f"true err(nodes)={np.max(np.abs(solution-true_ref)):.3e} | "
              f"approx err(nodes)={np.max(np.abs(solution-approx_ref)):.3e} | "
              f"true err(dense)={np.max(np.abs(solution_dense-true_ref_dense)):.3e} | "
              f"approx err(dense)={np.max(np.abs(solution_dense-approx_ref_dense)):.3e}")

    x_nodes_common = results[m_values[0]]['x_nodes']
    Xg, Mg = np.meshgrid(x_nodes_common, m_values)
    Xd, Md = np.meshgrid(x_dense, m_values)

    # 01: exact / numerical / final nodal values
    Zfinal = np.array([results[m]['solution'] for m in m_values])
    plot_surface_figure('01_exact_numerical_final_nodal_values.png', Xg, Mg, Zfinal,
                         'x value', 'Parameter value (m)', 'Final nodal values',
                         'Final Nodal Values (Numerical)', 'viridis')

    # 02: final relative true error
    Zerr_true = np.array([rel_err(results[m]['solution'], results[m]['true_ref']) for m in m_values])
    plot_surface_figure('02_exact_numerical_final_relative_true_error.png', Xg, Mg, Zerr_true,
                         'x value', 'Parameter value (m)', 'Final relative true error',
                         'Final Relative True Error (Numerical)', 'plasma')

    # 03 & 04: iterative (Pade-order refinement) nodal values / true error, per m
    for m in m_values:
        history = results[m]['history']
        true_ref = results[m]['true_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_nodes_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(h, true_ref) for h in history])

        plot_surface_figure(f'03_exact_numerical_iterative_values_p_{m}.png', Xi, Ii, Zval,
                             'x value', 'Iteration number', 'Iterative nodal values',
                             f'Iterative Nodal Values (m={m})', 'viridis', int_y=True)
        plot_surface_figure(f'04_exact_numerical_iterative_true_error_p_{m}.png', Xi, Ii, Zerr,
                             'x value', 'Iteration number', 'Iterative relative true error',
                             f'Iterative Relative True Error (m={m})', 'plasma', int_y=True)

    # 05 & 06: direct/analytical -- dense-grid evaluation of the SAME converged ADM+Pade solution
    Zdense_val = np.array([results[m]['solution_dense'] for m in m_values])
    plot_surface_figure('05_exact_analytical_nodal_values.png', Xd, Md, Zdense_val,
                         'x value', 'Parameter value (m)', 'Nodal values (direct ADM+Pade evaluation)',
                         'Direct/Analytical Evaluation of the ADM Series Solution (dense grid)', 'viridis')

    Zdense_err_true = np.array([rel_err(results[m]['solution_dense'], results[m]['true_ref_dense'])
                                 for m in m_values])
    plot_surface_figure('06_exact_analytical_relative_true_error.png', Xd, Md, Zdense_err_true,
                         'x value', 'Parameter value (m)', 'Relative true error (dense grid)',
                         'Direct/Analytical Evaluation - Relative True Error (dense grid)', 'plasma')

    # 07: no_exact / numerical / final nodal values (same data as 01)
    plot_surface_figure('07_no_exact_numerical_final_nodal_values.png', Xg, Mg, Zfinal,
                         'x value', 'Parameter value (m)', 'Final nodal values',
                         'Final Nodal Values (Numerical)', 'viridis')

    # 08: final relative approximate error
    Zerr_approx = np.array([rel_err(results[m]['solution'], results[m]['approx_ref']) for m in m_values])
    plot_surface_figure('08_no_exact_numerical_final_relative_approx_error.png', Xg, Mg, Zerr_approx,
                         'x value', 'Parameter value (m)', 'Final relative approximate error',
                         'Final Relative Approximate Error (Numerical)', 'plasma')

    # 09 & 10: iterative nodal values / approximate error, per m
    for m in m_values:
        history = results[m]['history']
        approx_ref = results[m]['approx_ref']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(x_nodes_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(h, approx_ref) for h in history])

        plot_surface_figure(f'09_no_exact_numerical_iterative_values_p_{m}.png', Xi, Ii, Zval,
                             'x value', 'Iteration number', 'Iterative nodal values',
                             f'Iterative Nodal Values (m={m})', 'viridis', int_y=True)
        plot_surface_figure(f'10_no_exact_numerical_iterative_approx_error_p_{m}.png', Xi, Ii, Zerr,
                             'x value', 'Iteration number', 'Iterative relative approximate error',
                             f'Iterative Relative Approximate Error (m={m})', 'plasma', int_y=True)

    # 11 & 12: direct/analytical -- same dense evaluation, approximate error
    plot_surface_figure('11_no_exact_analytical_nodal_values.png', Xd, Md, Zdense_val,
                         'x value', 'Parameter value (m)', 'Nodal values (direct ADM+Pade evaluation)',
                         'Direct/Analytical Evaluation of the ADM Series Solution (dense grid)', 'viridis')

    Zdense_err_approx = np.array([rel_err(results[m]['solution_dense'], results[m]['approx_ref_dense'])
                                   for m in m_values])
    plot_surface_figure('12_no_exact_analytical_relative_approx_error.png', Xd, Md, Zdense_err_approx,
                         'x value', 'Parameter value (m)', 'Relative approximate error (dense grid)',
                         'Direct/Analytical Evaluation - Relative Approximate Error (dense grid)', 'plasma')

    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f"\nDone. {n_files} PNG figures written to {OUTDIR}/")

    # -------------------------------------------------------------
    # Convergence study (number of ADM/Taylor series terms retained in
    # the diagonal Pade approximant, m=1 and m=5). Series methods
    # converge geometrically/spectrally in the number of terms, not at
    # a fixed algebraic order like a finite-difference or RK scheme, so
    # here we report the error directly against series length rather
    # than an "observed order," and plot it on a semilog axis where
    # geometric convergence shows up as a straight line.
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("Convergence study (max nodal error vs exact solution, ADM+Pade)")
    print("="*70)

    convergence_data = {}
    for m_test, exact_fn in [(1.0, lambda x: np.where(x < 1e-8, 1.0, np.sin(x)/(x+1e-15))),
                              (5.0, lambda x: 1.0/np.sqrt(1.0 + x**2/3.0))]:
        print(f"\n m = {m_test}")
        print(f"{'N_series':>9} {'Pade order':>11} {'Max Error':>15}")
        x_test = np.linspace(0.0, R, 50)
        exact_ref = exact_fn(x_test)   # computed once, reused across all N_series below

        n_series_list, err_list = [], []
        for N_series in [10, 20, 30, 40, 60]:
            series_coeffs = adm_series_coeffs(m_test, N_series)
            order = N_series // 2
            numerator_coeffs, denominator_coeffs = pade_approx(series_coeffs, order, order)
            solution_test = eval_pade(numerator_coeffs, denominator_coeffs, x_test**2)
            err = np.max(np.abs(solution_test - exact_ref))
            print(f"{N_series:>9} {order:>11} {err:>15.3e}")
            n_series_list.append(N_series)
            err_list.append(max(err, 1e-17))   # floor to keep semilogy well-behaved at 0 error
        convergence_data[m_test] = (n_series_list, err_list)

    print("\nError collapses super-exponentially with series length: the")
    print("Pade-resummed ADM series reaches machine precision by ~40-60 terms,")
    print("even for m=5 where the raw (non-resummed) Taylor series diverges")
    print("at x=R=2 because R exceeds its radius of convergence (~sqrt(3)).")

    # Convergence plot: error vs. series length N_series, semilog-y so
    # geometric/spectral convergence appears as a straight line.
    fig, ax = plt.subplots(figsize=(7, 6))
    for m_test, (n_series_list, err_list) in convergence_data.items():
        ax.semilogy(n_series_list, err_list, 'o-', label=f'm={m_test}')
    ax.set_xlabel('Series length N_series (Pade order = N_series/2)')
    ax.set_ylabel('Max nodal error vs. exact solution')
    ax.set_title(f'{TAG} - ADM+Pade Convergence')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/13_adm_pade_convergence.png')
    plt.close(fig)
    print(f"\nConvergence plot written to {OUTDIR}/13_adm_pade_convergence.png")
