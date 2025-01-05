# # Riemann SGD

from manifold import Manifold

def rsgdStep(m:Manifold, dfdx, eta, x):
    """
    Riemann SGD
    :param m: manifold
    :param dfdx: Euclidean gradient
    :param eta: current stepsize
    :param x: current point
    :return: next point
    """
    pgrad = m.proj(x, x - eta * dfdx(x))  # TBD
    return m.retr(x, pgrad)
