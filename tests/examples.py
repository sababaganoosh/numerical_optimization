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