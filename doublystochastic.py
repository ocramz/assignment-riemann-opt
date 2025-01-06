from torch import Tensor, Size, eye, ones, matmul, kron, exp, div, diag, reciprocal, norm, randn, abs
from torch.linalg import lstsq

def randomNonNegTensor(m):
    return abs(randn([m, m]))

def rowColDiffNorm(x:Tensor):
    (m, n) = x.size()
    (em, en) = ones([m]), ones([n])
    rd = norm(x @ em - em)
    cd = norm(x.mT @ en - en)
    # if verbose:
    #     print(f'row norm: {rowNorm}, col norm: {colNorm}')
    # return rowNorm <= tol and colNorm <= tol
    return rd, cd

def sinkhorn(x0, verbose=False, maxiter: int = 100, tol=1e-3):
    """ fixpoint iteration of Sinkhorn-Knopp from: https://strathprints.strath.ac.uk/19685/1/skapp.pdf
    :param x0: matrix with positive entries
    :param verbose: debug printout
    :param maxiter: max # of iterations
    :param tol: early return tolerance
    :return: doubly stochastic matrix
    """
    m, _ = x0.size()
    onesm = ones([m])
    r = onesm
    for i in range(maxiter):
        c = diag(reciprocal(x0.mT @ r)) @ onesm
        r = diag(reciprocal(x0 @ c)) @ onesm
        xHat = diag(r) @ x0 @ diag(c)
        rn, cn = rowColDiffNorm(xHat)
        if verbose and i % 10 == 0:
            print(f'row diff: {rn}, col diff: {cn}')
        if rn <= tol and cn <= tol:
            break
    if verbose:
        print(f'done in {i} iters at tol {tol}')
        # print(f'result: {xHat}')
    return xHat

class DoublyStochastic:
    """Manifold of doubly-stochastic matrices
    reimplemented from scratch, with references
    interface loosely following that of mctorch but with more comments"""
    def __init__(self, m:int):
        """ n x n doubly stochastic matrices
        :param m: # rows
        """
        self.m = m
        self.idm = eye(self.m)  # Identity(m)
        self.onesm = ones([self.m])  # vector of ones

        self._size = Size((self.m, self.m))

    def size(self):
        return self._size

    def rand(self):
        """
        sample a random point on the manifold
        :return: a random point on the manifold
        """
        x = randomNonNegTensor(self.m)
        return sinkhorn(x)

    def proj(self, x, z):
        """ orthogonal projection on the tangent
        Eqn. B.11 of https://arxiv.org/pdf/1802.02628.pdf
        :param x: point on the manifold at which the tangent is computed
        :param z: point to be projected
        :return: point on the tangent
        """
        # # Eqn B.9
        a = (self.idm - x @ x.mT)
        b = (z - (x @ z.mT)) @ self.onesm
        alpha = a.mT @ b
        # alpha = lstsq(self.idm - x @ x.mT, (z - (x @ z.mT)) @ self.onesm).solution
        # # Eqn B.10
        beta = z.mT @ self.onesm - x.mT @ alpha

        prj = z - (alpha @ self.onesm + self.onesm @ beta) * x

        return prj


    def retr(self, x, v):
        """ Retraction on the manifold
        Eqn 34 of https://arxiv.org/pdf/1802.02628.pdf
        :param x: point on the manifold at which the tangent is computed
        :param v: point to be retracted on the manifold (written "xi" in the paper)
        :return: point on the manifold
        """
        return sinkhorn(x * exp(div(v, x)))

    def egrad2rgrad(self, x, u):
        """
        Euclidean gradient to Riemann gradient
        Lemma 1 of https://arxiv.org/pdf/1802.02628.pdf (p.6)
        :param x: point on the manifold
        :param u: Euclidean gradient vector
        :return: projected gradient
        """
        mu = x * u  # elementwise product
        return self.proj(x, mu)



