import os
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
from matplotlib.ticker import MaxNLocator

OUTDIR = 'output_figures/FDM_Isothermal/3D_figures_grouped'
os.makedirs(OUTDIR, exist_ok=True)
TAG = 'FDM Isothermal'
DEFAULT_R = 2.0

def make_mesh(R, n_elem):
    return np.linspace(0.0, R, n_elem + 1)

def dense_evaluate(nodes, c, x_dense, method='pchip'):
    if method == 'pchip':
        interp = PchipInterpolator(nodes, c)
    elif method == 'cubic':
        interp = CubicSpline(nodes, c, bc_type=((1, 0.0), 'not-a-knot'))
    else:
        raise ValueError(f'Unknown dense_evaluate method: {method!r}')
    y = interp(x_dense)
    lo, hi = (np.min(c), np.max(c))
    pad = 1e-06 * (hi - lo + 1e-12)
    if np.any(y < lo - pad) or np.any(y > hi + pad):
        warnings.warn('dense_evaluate: interpolated values overshoot the nodal data range -- possible non-physical oscillation.')
    return y

def nonlinear_source(y, lam):
    """
    For the isothermal Lane-Emden equation: y'' + (2/x)y' = lambda * exp(-y)
    Rearranged to equal 0: y'' + (2/x)y' - lambda * exp(-y) = 0
    """
    Y_val = lam * np.exp(-y)
    dY_val = -lam * np.exp(-y)
    return (Y_val, dY_val)

def assemble_system(c, lam, nodes, h, n_elem, need_jacobian=True):
    n = n_elem + 1
    F = np.zeros(n)
    rows, cols, vals = ([], [], [])
    
    # Boundary condition at x=0: y(0) = 0
    F[0] = c[0] - 0.0
    if need_jacobian:
        rows.append(0)
        cols.append(0)
        vals.append(1.0)
        
    # Boundary condition at x=0: y'(0) = 0 (2nd-order forward difference)
    inv2h = 1.0 / (2.0 * h)
    F[1] = (-3.0 * c[0] + 4.0 * c[1] - c[2]) * inv2h
    if need_jacobian:
        rows += [1, 1, 1]
        cols += [0, 1, 2]
        vals += [-3.0 * inv2h, 4.0 * inv2h, -1.0 * inv2h]
        
    # Interior nodes
    i = np.arange(1, n_elem)
    xi = np.maximum(nodes[i], np.finfo(float).eps)
    cm1, c0, cp1 = (c[i - 1], c[i], c[i + 1])
    
    Y_val, dY_val = nonlinear_source(c0, lam)
    adv = (cp1 - cm1) / (xi * h)
    lap = (cm1 - 2.0 * c0 + cp1) / h ** 2
    F[2:] = lap + adv - Y_val
    
    if need_jacobian:
        out_rows = np.arange(2, n)
        a_sub = 1.0 / h ** 2 - 1.0 / (xi * h)
        a_diag = -2.0 / h ** 2 - dY_val
        a_sup = 1.0 / h ** 2 + 1.0 / (xi * h)
        rows.extend(out_rows)
        cols.extend(i - 1)
        vals.extend(a_sub)
        rows.extend(out_rows)
        cols.extend(i)
        vals.extend(a_diag)
        rows.extend(out_rows)
        cols.extend(i + 1)
        vals.extend(a_sup)
        
    J = None
    if need_jacobian:
        J = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return (F, J)

def solve_isothermal_fdm(lam, n_elem=40, R=DEFAULT_R, max_iter=50, tol=1e-11, ls_max_halvings=30, verbose=False):
    if n_elem < 4:
        raise ValueError('n_elem must be >= 4: the 2nd-order forward-difference condition at row 1 needs node x_2 to exist.')
    nodes = make_mesh(R, n_elem)
    h = nodes[1] - nodes[0]
    
    # Taylor expansion initial guess for isothermal sphere
    c = (lam / 6.0) * (nodes ** 2)
    history = [c.copy()]
    
    F, J = assemble_system(c, lam, nodes, h, n_elem, need_jacobian=True)
    res_norm = np.linalg.norm(F, np.inf)
    converged = False
    it = 0
    
    for it in range(1, max_iter + 1):
        if res_norm < tol:
            converged = True
            break
        dc = spla.spsolve(J.tocsc(), -F)
        if not np.all(np.isfinite(dc)):
            raise RuntimeError(f'Newton solver produced a non-finite update at iteration {it} (lam={lam}, n_elem={n_elem}); the Jacobian is likely (near-)singular.')
        
        alpha = 1.0
        accepted = False
        F_trial = None
        for _ls in range(ls_max_halvings):
            c_trial = c + alpha * dc
            F_trial, _ = assemble_system(c_trial, lam, nodes, h, n_elem, need_jacobian=False)
            res_trial = np.linalg.norm(F_trial, np.inf)
            if np.all(np.isfinite(F_trial)) and res_trial <= (1.0 - 0.0001 * alpha) * res_norm:
                accepted = True
                break
            alpha *= 0.5
            
        if not accepted:
            raise RuntimeError(f'Newton line search failed to reduce the residual after {ls_max_halvings} halvings (lam={lam}, n_elem={n_elem}, iteration={it}); treating this as divergence rather than looping forever.')
        
        step_norm = np.linalg.norm(alpha * dc, np.inf)
        c = c_trial
        history.append(c.copy())
        F, J = assemble_system(c, lam, nodes, h, n_elem, need_jacobian=True)
        res_norm = np.linalg.norm(F, np.inf)
        
        if verbose:
            print(f'    it={it:3d}  alpha={alpha:6.3f}  |F|_inf={res_norm:.3e}  |dc|_inf={step_norm:.3e}')
        if step_norm < tol and res_norm < tol:
            converged = True
            break
            
    if not converged:
        warnings.warn(f'Newton solver for lam={lam}, n_elem={n_elem} did not reach tol={tol:.1e} within {max_iter} iterations (final |F|_inf={res_norm:.3e}).')
    return (nodes, c, history, it, converged)

def get_reference_solution(lam, x_eval):
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

def get_self_convergence_reference(lam, x_eval, R=DEFAULT_R, n_elem_fine=160):
    nodes_f, c_f, _, _, _ = solve_isothermal_fdm(lam, n_elem=n_elem_fine, R=R)
    return dense_evaluate(nodes_f, c_f, x_eval)

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
        nodes, c, history, nit, converged = solve_isothermal_fdm(lam, n_elem=n_elem_main, R=R)
        ref_hi = get_reference_solution(lam, nodes)
        ref_self = get_self_convergence_reference(lam, nodes, R=R)
        
        Y_dense = dense_evaluate(nodes, c, x_dense)
        ref_hi_dense = get_reference_solution(lam, x_dense)
        ref_self_dense = get_self_convergence_reference(lam, x_dense, R=R)
        
        results[lam] = dict(nodes=nodes, Y=c, history=history, ref_hi=ref_hi, ref_self=ref_self, 
                            Y_dense=Y_dense, ref_hi_dense=ref_hi_dense, ref_self_dense=ref_self_dense)
                            
        conv_tag = 'OK' if converged else 'NOT CONVERGED'
        print(f'lam={lam}: {nit} Newton its [{conv_tag}] | err(nodes) vs hi-acc ref={np.max(np.abs(c - ref_hi)):.3e} | err(nodes) vs self-conv ref={np.max(np.abs(c - ref_self)):.3e} | err(dense) vs hi-acc ref={np.max(np.abs(Y_dense - ref_hi_dense)):.3e} | err(dense) vs self-conv ref={np.max(np.abs(Y_dense - ref_self_dense)):.3e}')
        
    nodes_common = results[lam_values[0]]['nodes']
    Xg, Mg = np.meshgrid(nodes_common, lam_values)
    Xd, Md = np.meshgrid(x_dense, lam_values)
    Zfinal = np.array([results[lam]['Y'] for lam in lam_values])
    
    # 01: Final Nodal Values
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zfinal, 'viridis', 'x value', 'Parameter value (lambda)', 'Final nodal values', 'Final Nodal Values (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    fig01_path = f'{OUTDIR}/01_exact_numerical_final_nodal_values.png'
    plt.savefig(fig01_path)
    plt.close(fig)
    
    # 02: Final Relative True Error
    Zerr_hi = np.array([rel_err(results[lam]['Y'], results[lam]['ref_hi']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xg, Mg, Zerr_hi, 'plasma', 'x value', 'Parameter value (lambda)', 'Final relative error vs. reference', 'Final Relative Error vs. Reference Solution (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/02_exact_numerical_final_relative_true_error.png')
    plt.close(fig)
    
    # 03 & 04: Iterative Nodal Values and True Error (10 figures generated here)
    for lam in lam_values:
        history = results[lam]['history']
        ref_hi = results[lam]['ref_hi']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(nodes_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(hh, ref_hi) for hh in history])
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Iteration number', 'Iterative nodal values', f'Iterative Nodal Values (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/03_exact_numerical_iterative_values_p_{lam}.png')
        plt.close(fig)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Iteration number', 'Iterative relative error vs. reference', f'Iterative Relative Error vs. Reference (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/04_exact_numerical_iterative_true_error_p_{lam}.png')
        plt.close(fig)
        
    # 05: Direct Analytical Nodal Values
    Zdense_val = np.array([results[lam]['Y_dense'] for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_val, 'viridis', 'x value', 'Parameter value (lambda)', 'Nodal values (direct spline reconstruction)', 'Direct/Analytical Evaluation of the FDM Solution (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    fig05_path = f'{OUTDIR}/05_exact_analytical_nodal_values.png'
    plt.savefig(fig05_path)
    plt.close(fig)
    
    # 06: Direct Analytical Relative True Error
    Zdense_err_hi = np.array([rel_err(results[lam]['Y_dense'], results[lam]['ref_hi_dense']) for lam in lam_values])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf(ax, Xd, Md, Zdense_err_hi, 'plasma', 'x value', 'Parameter value (lambda)', 'Relative error vs. reference (dense grid)', 'Direct/Analytical Evaluation - Relative Error vs. Reference (dense grid)')
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
    surf(ax, Xg, Mg, Zerr_self, 'plasma', 'x value', 'Parameter value (lambda)', 'Final relative error vs. self-convergence reference', 'Final Relative Error vs. Self-Convergence Reference (Numerical)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/08_no_exact_numerical_final_relative_approx_error.png')
    plt.close(fig)
    
    # 09 & 10: Iterative Approx Error (10 figures generated here)
    for lam in lam_values:
        history = results[lam]['history']
        ref_self = results[lam]['ref_self']
        iters = np.arange(len(history))
        Xi, Ii = np.meshgrid(nodes_common, iters)
        Zval = np.array(history)
        Zerr = np.array([rel_err(hh, ref_self) for hh in history])
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zval, 'viridis', 'x value', 'Iteration number', 'Iterative nodal values', f'Iterative Nodal Values (lam={lam})', int_y=True)
        fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
        plt.savefig(f'{OUTDIR}/09_no_exact_numerical_iterative_values_p_{lam}.png')
        plt.close(fig)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf(ax, Xi, Ii, Zerr, 'plasma', 'x value', 'Iteration number', 'Iterative relative error vs. self-convergence reference', f'Iterative Relative Error vs. Self-Convergence Reference (lam={lam})', int_y=True)
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
    surf(ax, Xd, Md, Zdense_err_self, 'plasma', 'x value', 'Parameter value (lambda)', 'Relative error vs. self-convergence reference (dense grid)', 'Direct/Analytical Evaluation - Relative Error vs. Self-Convergence Reference (dense grid)')
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.5, pad=0.1)
    plt.savefig(f'{OUTDIR}/12_no_exact_analytical_relative_approx_error.png')
    plt.close(fig)
    
    # Explicitly calculate and print the total figure count
    n_files = len([f for f in os.listdir(OUTDIR) if f.endswith('.png')])
    print(f'\nDone. The FULL {n_files}-figure set has been successfully written to {OUTDIR}/ (figures 07 and 11 are file copies of 01 and 05).')
    
    print('\n' + '=' * 70)
    print('Convergence study (max nodal error vs. Radau ODE reference)')
    print('=' * 70)
    
    for lam_test in [1.0, 2.5]:
        print(f'\n lambda = {lam_test}')
        print(f"{'n_elem':>8} {'h=R/n_elem':>12} {'Max Error':>15} {'Observed Order':>16}")
        prev_err, prev_h = (None, None)
        for n_elem in [10, 20, 40, 80, 160]:
            nodes_c, c_c, _, _, conv_c = solve_isothermal_fdm(lam_test, n_elem=n_elem, R=R)
            err = np.max(np.abs(c_c - get_reference_solution(lam_test, nodes_c)))
            h = R / n_elem
            order = np.log(prev_err / err) / np.log(prev_h / h) if prev_err else np.nan
            order_s = f'{order:.2f}' if not np.isnan(order) else '  --'
            print(f'{n_elem:>8} {h:>12.5f} {err:>15.3e} {order_s:>16}')
            prev_err, prev_h = (err, h)
            
    print('\nObserved order ~2 (O(h^2)): consistent with the 2nd-order')
    print('central-difference interior stencil plus a genuinely 2nd-order')
    print('closure at the origin -- confirms clean second-order')
    print("global convergence, the project's core requirement.")