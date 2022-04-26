from examples import *
from src.unconstrained_min import *
from src.utils import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import latex

functions = [f1, f2, f3, f4, f5, f6]

for f in functions:
    if f == f4:
        x0 = np.array([-1, 2]).T
        max_iter = 10000
    else:
        x0 = np.array([1, 1]).T
        max_iter = 100
    step_len = 'wolfe'
    obj_tol = 10 ** -12
    param_tol = 10 ** -8

    print(f'Function - {f.__name__}')
    print('Gradient Descent:')
    grad_iterations, _, _ = line_searech_min(f, x0, step_len, obj_tol, param_tol, max_iter, min_type='gradient')
    print("Newton's Method:")
    newton_iterations, _, _ = line_searech_min(f, x0, step_len, obj_tol, param_tol, max_iter, min_type='newton')
    fig = line_search_plots(grad_iterations, newton_iterations, f)
    plt.savefig(f'{f.__name__}_plots')