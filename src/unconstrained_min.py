import numpy as np


def line_search_min(f, x0, step_len, obj_tol, param_tol, max_iter, min_type):
    """
    Inputs:
    f         - the function to minimize
    x0        - the starting point
    step_len  - the float constant step length or the string 'wolfe'.
    max_iter  - the maximum allowed number of iterations
    obj_tol   - the numeric tolerance for successful termination in terms of small enough change in objective function 
                values, between two consecutive iterations (𝑓(𝑥𝑖+1) and 𝑓(𝑥𝑖)), 
                or in the Newton Decrement based approximation of the objective decrease. 
    param_tol - the numeric tolerance for successful termination in terms of small enough distance between two consecutive 
                iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖).
    min_type  - the implementation type to be used for minimization. Accepts 'gradient' or 'newton' as argument
    """
    # sanity check for acceptable min_type
    acceptable_min_types = ['gradient', 'newton']
    if min_type not in acceptable_min_types:
        raise Exception(f'{min_type} is not an accepted min_type input')

    # calculating initial values for x0
    x_prev = x0

    if min_type == 'gradient':
        f_prev, df_prev = f(x0, hess=False)
    else:
        f_prev, df_prev, h_prev = f(x0, hess=True)

    # logging intial values in initial iteration
    iterations = []
    i = 0
    success = False

    iterations.append(list(x_prev) + [f_prev, i])

    # line search iteration loop
    while not success and i < max_iter:
        # step calculation for float step value
        if step_len != 'wolfe':
            x_next = x_prev - (step_len * df_prev)
        # step calculation for wolfe condition
        else:
            alpha = 1
            # calculate search direction
            if min_type == 'gradient':
                search_dir = -np.identity(len(df_prev)) @ df_prev
            else:
                try:
                    search_dir = -np.linalg.solve(h_prev, np.identity(len(h_prev))) @ df_prev
                except np.linalg.LinAlgError:
                    print('Error in calculating Hessian inverse, substituting with 0 matrix')
                    search_dir = np.zeros(h_prev.shape) @ df_prev

            # calculating wolfe condition inequality values
            next_iter_f, next_iter_df = f(x_prev + alpha * search_dir, hess=False)
            comparison = f_prev + .5 * alpha * df_prev.T @ search_dir

            # loop for updating alpha according to wolfe condition
            while next_iter_f > comparison:
                alpha = .9 * alpha
                next_iter_f, next_iter_df = f(x_prev + alpha * search_dir, hess=False)
                comparison = f_prev + .5 * alpha * df_prev.T @ search_dir
            x_next = x_prev + (alpha * search_dir)

        # updating step values based on line search method type (gradient or newton)    
        if min_type == 'gradient':
            f_next, df_next = f(x_next, hess=False)
        else:
            f_next, df_next, h_next = f(x_next, hess=True)

        # checking convergence conditions
        if abs(f_next - f_prev) < obj_tol:
            success = True
            print('output stop condition met')
        elif abs(x_next - x_prev).any() < param_tol:
            succes = True
            print('input stop condition met')
        # logging step values in current iteration
        else:
            i += 1
            iterations.append(list(x_next) + [f_next, i])

        # preparing values for next iteration
        x_prev, f_prev, df_prev = x_next, f_next, df_next
        if min_type != 'gradient':
            h_prev = h_next

    # handling non-convergence
    if success == False:
        print('max iter condition met')

    iterations = np.array(iterations)
    print(f'final iteration {i}: x = {x_next}, f = {f_next}')
    final_iter = iterations[-1]

    return iterations, final_iter, success 