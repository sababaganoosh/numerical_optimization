from examples import *
from src.constrained_min import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # appropriate import to draw 3d polygons

def test_qp(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs,
            x0, obj_tol, param_tol, max_iter, eps, mu):
    # ***Optimization Execution***
    p1_min, p1_its = interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs,
                                 x0, obj_tol, param_tol, max_iter, eps, mu)

    # ***Optimization Visualization***
    # **Viz A**
    print(f'Final Candidate: {[round(var,3) for var in p1_min]}')

    # **Viz B**
    print(f'Final Candidate Objective Value: {round(func(p1_min)[0], 3)}')
    for i, ineq in enumerate(ineq_constraints):
        status = ineq(p1_min)[0] <= 0
        if status:
            print(f'Inequality Constraint {i+1} implemented in solution')
        else:
            print(f'Inequality Constraint {i+1} NOT implemented in solution')

    status = round((eq_constraints_mat @ p1_min)[0], 8) == eq_constraints_rhs[0][0]
    if status:
        print(f'Equality Constraint implemented in solution')
    else:
        print(f'Equality Constraint NOT implemented in solution')

    # Set Fig
    fig = plt.figure(figsize = (10,18))

    # **Viz C**
    # Subplot Config
    ax0 = fig.add_subplot(2, 1, 1, projection='3d')
    ax0.set_title('Algorithm Path and Feasibility Region')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.set_zlabel('Z')
    # *Feasible Region*
    # Create vertices from points
    edge1 = np.array([1, 0, 0])
    edge2 = np.array([0, 1, 0])
    edge3 = np.array([0, 0, 1])
    verts = [edge1, edge2, edge3]

    # Create polygon in 3d space
    shape = Poly3DCollection([verts], alpha=.25, color='y')
    #  Add polygon to plot
    poly = plt.gca().add_collection3d(shape)

    # *Algorithm Path*
    # extracting values
    input_len = len(x0)
    input_var = p1_its[:,:input_len]
    x, y, z = input_var[:,0], input_var[:,1], input_var[:,2]

    # add path to plot
    alg_path = ax0.plot(x, y, z, color='r')

    # **Viz D**
    # Subplot Config
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.set_title('Convergence of Objective')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')

    # extracting values
    input_len = len(x0)
    obj_val, iters = p1_its[:,input_len], p1_its[:,input_len+1]

    g_out = ax1.plot(iters, obj_val, label='gradient')

    plt.show()


def test_lp(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs,
            x0, obj_tol, param_tol, max_iter, eps, mu):
    # ***Optimization Execution***
    p2_min, p2_its = interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs,
                                 x0, obj_tol, param_tol, max_iter, eps, mu)

    # ***Optimization Visualization***
    # **Viz A**
    print(f'Final Candidate: {[round(var,3) for var in p2_min]}')

    # **Viz B**
    print(f'Final Candidate Objective Value: {round(func(p2_min)[0], 3)}')
    for i, ineq in enumerate(ineq_constraints):
        status = ineq(p2_min)[0] <= 0
        if status:
            print(f'Inequality Constraint {i+1} implemented in solution')
        else:
            print(f'Inequality Constraint {i+1} NOT implemented in solution')

    # Set Fig
    fig = plt.figure(figsize = (10,18))

    # **Viz C**
    # Subplot Config
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.set_title('Algorithm Path and Feasibility Region')
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')

    # extracting values
    input_len = len(x0)
    input_var, f, its = p2_its[:,:input_len], -p2_its[:,input_len], p2_its[:,input_len+1]

    # *Feasible Region*
    def fun(x):
        return 1 - x

    # y lower limit
    x1 = np.linspace(fun(0), 2, 50)
    y1 = np.zeros(50)

    # y upper limit
    x2 = np.linspace(fun(1), 2, 50)
    y2 = np.ones(50)

    # x upper limit
    x3 = np.ones(50)*2
    y3 = np.linspace(0, 1, 50)

    # line limit
    y4 = np.linspace(0, 1, 50)
    x4 = fun(y3)

    ax0.plot(x1, y1, color='y')
    ax0.plot(x2, y2, color='y')
    ax0.plot(x3, y3, color='y')
    ax0.plot(x4, y4, color='y')
    plt.fill_between(x4, fun(x4), y2, color='yellow', alpha=0.25)
    plt.fill_between(x1, y1, y2, color='yellow', alpha=0.25)

    # *Algorithm Path*
    X, Y = input_var[:,0], input_var[:,1]
    ax0.plot(X, Y, color='r')

    # **Viz D**
    # Subplot Config
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.set_title('Convergence of Objective')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')

    g_out = ax1.plot(its, f, label='gradient')

    plt.show()


# General Parameters
max_iter = 100
obj_tol = 10**-12
param_tol = 10**-8
eps = param_tol
mu = 10

# QP Test
func = func1
ineq_constraints = [p1_ineq1, p1_ineq2, p1_ineq3]
eq_constraints_mat = np.array([[1, 1, 1]])
eq_constraints_rhs = np.array([[1]])
x0 = np.array([0.1, 0.2, 0.7])

test_qp(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs,
        x0, obj_tol, param_tol, max_iter, eps, mu)


# LP Test
func = func2
ineq_constraints = [p2_ineq1, p2_ineq2, p2_ineq3, p2_ineq4]
eq_constraints_mat = None
eq_constraints_rhs= None
x0 = np.array([0.5, 0.75])

test_lp(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs,
        x0, obj_tol, param_tol, max_iter, eps, mu)