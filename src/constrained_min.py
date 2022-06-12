import numpy as np

def func_with_log_barr(func, ineq_constraints, t, x):
    f, df, h = func(x)
    f, df, h = t * f, t * df, t * h

    for ineq in ineq_constraints:
        f_temp, g_temp, h_temp = ineq(x)

        f = f - np.log(-f_temp)
        df = df - (1 / f_temp) * g_temp
        h = h + (1 / f_temp ** 2) * g_temp @ g_temp.T - (1 / f_temp) * h_temp

    return f, df, h


def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, obj_tol, param_tol, max_iter, eps,
                mu):
    """
    Inputs:
    func               - the function to minimize
    ineq_constraints   - list of inequality constraint functions to be observed
    eq_constraints_mat - A of Ax = b
    eq_constraints_rhs - b of Ax = b
    x0                 - the starting point
    obj_tol            - the numeric tolerance for successful termination in terms of small enough change in objective function
                         values, between two consecutive iterations (𝑓(𝑥𝑖+1) and 𝑓(𝑥𝑖)),
                         or in the Newton Decrement based approximation of the objective decrease.
    param_tol          - the numeric tolerance for successful termination in terms of small enough distance between two consecutive
                         iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖).
    max_iter           - the maximum allowed number of iterations
    eps                - tolerance measure for t
    mu                 - scalar value for outer iteration increase in t
    """
    # important vals
    if eq_constraints_mat is not None:
        l = len(eq_constraints_mat)
    m = len(ineq_constraints)
    n = len(x0)
    # initial values
    t = 1
    x_prev = x0

    # outer loop
    j = 0
    if eq_constraints_mat is None:
        iterations = [list(x_prev) + [func(x_prev)[0], j] + list(np.zeros((1)))]
    else:
        iterations = [list(x_prev) + [func(x_prev)[0], j] + list(np.zeros((len(eq_constraints_rhs))))]

    while m / t > eps:
        j += 1
        # update t if not first outer loop
        if j == 1:
            pass
        else:
            t *= mu
        # get initial values of barrier functon
        f_prev, df_prev, h_prev = func_with_log_barr(func, ineq_constraints, t, x_prev)

        # inner loop
        i = 0
        success = False

        while not success and i < max_iter:
            # calculate Newton Direction
            if eq_constraints_mat is None:
                search_dir = -np.linalg.solve(h_prev, np.identity(len(h_prev))) @ df_prev
                search_dir = search_dir.reshape((-1))
                w_k = np.zeros((1))
            else:
                left = np.bmat([[h_prev, eq_constraints_mat.T],
                                [eq_constraints_mat, np.zeros((l, l))]])
                right = np.vstack((-df_prev, np.zeros((l, l))))

                lin_set_solution = np.linalg.solve(left, right)
                search_dir = lin_set_solution[:n].reshape((-1))
                w_k = lin_set_solution[n:].reshape((-1))

            # calculate alpha with wolfe conditions
            alpha = 1
            # calculating wolfe condition inequality values
            next_iter_f, _, _ = func_with_log_barr(func, ineq_constraints, t, x_prev + alpha * search_dir)
            if np.isnan(next_iter_f):
                next_iter_f = np.inf
            comparison = f_prev + .5 * alpha * df_prev.T @ search_dir
            # loop for updating alpha according to wolfe condition
            while next_iter_f > comparison:
                alpha = .9 * alpha
                next_iter_f, _, _ = func_with_log_barr(func, ineq_constraints, t, x_prev + alpha * search_dir)
                if np.isnan(next_iter_f):
                    next_iter_f = np.inf
                comparison = f_prev + .5 * alpha * df_prev.T @ search_dir

            # updating step values
            x_next = x_prev + (alpha * search_dir)
            f_next, df_next, h_next = func_with_log_barr(func, ineq_constraints, t, x_next)

            # checking convergence conditions
            if abs(f_next - f_prev) < obj_tol:
                success = True
                objective_val, _, _ = func(x_next)
                iterations.append(list(x_next) + [objective_val, j] + list(w_k / t))
            elif abs(x_next - x_prev).any() < param_tol:
                success = True
                objective_val, _, _ = func(x_next)
                iterations.append(list(x_next) + [objective_val, j] + list(w_k / t))
            # logging step values in current iteration
            else:
                i += 1

            # preparing values for next iteration
            x_prev, f_prev, df_prev, h_prev = x_next, f_next, df_next, h_next

    iterations = np.array(iterations)
    return x_prev, iterations