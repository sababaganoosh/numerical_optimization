import numpy as np
import matplotlib.pyplot as plt

def line_search_plots(grad_its, new_its, f):
    # extracting values
    x_g, y_g, f_g, i_g = grad_its[:, 0], grad_its[:, 1], grad_its[:, 2], grad_its[:, 3]
    x_n, y_n, f_n, i_n = new_its[:, 0], new_its[:, 1], new_its[:, 2], new_its[:, 3]

    # generating values for graph scale to plot input on
    stop_x = max(abs(x_g.min()), abs(x_n.min()), abs(x_g.max()), abs(x_n.max()))
    stop_y = max(abs(y_g.min()), abs(y_n.min()), abs(y_g.max()), abs(y_n.max()))
    start_x, start_y = -stop_x, -stop_y

    x1 = np.linspace(start_x, stop_x, 50)
    x2 = np.linspace(start_y, stop_y, 50)

    # converting graph values to grids to plug into function for contour lines
    X1, X2 = np.meshgrid(x1, x2)

    # calculating contour line values
    Z = np.empty(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j], _ = f(np.array([X1[i, j], X2[i, j]]), hess=False)

    # initialize plot
    fig, ax = plt.subplots(2, figsize=(10, 14))

    # figure 1
    # contour plot
    contours = ax[0].contour(X1, X2, Z, 50)
    # gradient points
    g_in = ax[0].plot(x_g, y_g, linestyle='dashed', marker='o', label='gradient')
    # newton points
    n_in = ax[0].plot(x_n, y_n, linestyle='dashed', marker='o', label='newton')

    # figure 2
    # gradient points
    g_out = ax[1].plot(i_g, f_g, label='gradient')
    # newton points
    n_out = ax[1].plot(i_n, f_n, label='newton')

    # figure parameters
    ax[0].set_title('Line Search Path over Function Contour Lines')
    ax[0].set_xlim([1.15 * start_x, 1.15 * stop_x])
    ax[0].set_ylim([1.15 * start_y, 1.15 * stop_y])
    ax[0].legend()

    ax[1].set_title('Iteration Function Values')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Function Value')
    ax[1].legend()

    # plt.show()
    return fig