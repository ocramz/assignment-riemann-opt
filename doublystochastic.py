from torch import Tensor, eye, ones, matmul, kron, exp, div, diag, reciprocal, norm, randn, abs
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
    def __init__(self, m:int):
        """ n x n doubly stochastic matrices
        :param m: # rows
        """
        self.m = m
        self.idm = eye(self.m)  # Identity(m)
        self.onesm = ones([self.m])  # vector of ones

    def rand(self):
        """
        sample a random point on the manifold
        :return:
        """
        x = randomNonNegTensor(self.m)
        return sinkhorn(x)

    def orthogonalProj(self, x, z):
        """
        Eqn. B.11 of https://arxiv.org/pdf/1802.02628.pdf
        :param x: point on the manifold at which the tangent is computed
        :param z: point to be projected
        :return: point on the tangent
        """
        # # solve A x = b in the least squares sense
        # torch.linalg.lstsq(A, b).solution == A.pinv() @ b

        # # Eqn B.9
        alpha = lstsq(self.idm - x @ x.t, z - (x @ z.t) @ self.onesm).solution
        # # Eqn B.10
        beta = z.t @ self.onesm - x.t @ alpha

        return z - kron(alpha @ self.onesm.t + self.onesm @ beta.t, x)


    def retraction(self, x, v):
        """
        Eqn 34 of https://arxiv.org/pdf/1802.02628.pdf
        :param x: point on the manifold at which the tangent is computed
        :param v: point to be retracted on the manifold (written "xi" in the paper)
        :return: point on the manifold
        """
        return self.sinkhorn(x * exp(div(v, x)))



