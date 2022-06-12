import numpy as np

def f1(x, hess):
    Q = np.array([[1, 0], [0, 1]])

    f = x.T @ Q @ x
    g = 2 * Q @ x
    if hess:
        h = 2 * Q
        return f, g, h
    else:
        return f, g


def f2(x, hess):
    Q = np.array([[1, 0], [0, 100]])

    f = x.T @ Q @ x
    g = 2 * Q @ x
    if hess:
        h = 2 * Q
        return f, g, h
    else:
        return f, g


def f3(x, hess):
    a = np.array([[np.sqrt(3) / 2, -0.5],
                  [0.5, np.sqrt(3) / 2]])
    Q = a.T @ np.array([[100, 0], [0, 1]]) @ a

    f = x.T @ Q @ x
    g = 2 * Q @ x
    if hess:
        h = 2 * Q
        return f, g, h
    else:
        return f, g


def f4(x, hess):
    x1 = x[0]
    x2 = x[1]

    f = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    g = np.array([-400 * x1 * (x2 - x1 ** 2) - 2 * (1 - x1), 200 * (x2 - x1 ** 2)]).T
    if hess:
        h = np.array([[1200 * x1 ** 2 + 2 * x1, -400 * x1],
                      [-400 * x1, 200]])
        return f, g, h
    else:
        return f, g


def f5(x, hess):
    a = np.array([5, 15]).T

    f = a.T @ x
    g = a
    if hess:
        h = np.array([[0, 0], [0, 0]])
        return f, g, h
    else:
        return f, g


def f6(x, hess):
    x1 = x[0]
    x2 = x[1]

    f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1),
                  3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)]).T
    if hess:
        h = np.array([[np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1),
                       3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)],
                      [3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1),
                       9 * np.exp(x1 + 3 * x2 - 0.1) - 9 * np.exp(x1 - 3 * x2 - 0.1)]])
        return f, g, h
    else:
        return f, g


def func1(var):
    x, y, z = var

    f = x ** 2 + y ** 2 + (z + 1) ** 2
    g = np.array([[2 * x, 2 * y, 2 * z + 2]]).T
    h = np.array([[2, 0, 0],
                  [0, 2, 0],
                  [0, 0, 2]])

    return f, g, h


def p1_ineq1(var):
    x, y, z = var

    f = -x
    g = np.array([[-1, 0, 0]]).T
    h = np.zeros((3, 3))

    return f, g, h


def p1_ineq2(var):
    x, y, z = var

    f = -y
    g = np.array([[0, -1, 0]]).T
    h = np.zeros((3, 3))

    return f, g, h


def p1_ineq3(var):
    x, y, z = var

    f = -z
    g = np.array([[0, 0, -1]]).T
    h = np.zeros((3, 3))

    return f, g, h


def func2(var):
    x, y = var

    f = x + y
    g = np.array([[1, 1]]).T
    h = np.zeros((2, 2))

    return -f, -g, -h


def p2_ineq1(var):
    x, y = var

    f = -x - y + 1
    g = np.array([[-1, -1]]).T
    h = np.zeros((2, 2))

    return f, g, h


def p2_ineq2(var):
    x, y = var

    f = y - 1
    g = np.array([[0, 1]]).T
    h = np.zeros((2, 2))

    return f, g, h


def p2_ineq3(var):
    x, y = var

    f = x - 2
    g = np.array([[1, 0]]).T
    h = np.zeros((2, 2))

    return f, g, h


def p2_ineq4(var):
    x, y = var

    f = -y
    g = np.array([[0, -1]]).T
    h = np.zeros((2, 2))

    return f, g, h