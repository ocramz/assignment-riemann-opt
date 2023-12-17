from torch import eye, ones, matmul, kron, exp, div
from torch.linalg import lstsq

class DoublyStochastic:
    def __init__(self, m:int):
        """ n x n doubly stochastic matrices
        :param m: # rows
        """
        self.m = m
        self.idm = eye(self.m)  # Identity(m)
        self.onesm = ones([self.m])  # vector of ones

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
        alpha = lstsq(self.idm - x @ x.t, z - (x @ z.t) @ self.onesm)
        # # Eqn B.10
        beta = z.t @ self.onesm - x.t @ alpha

        return z - kron(alpha @ self.onesm.t + self.onesm @ beta.t, x)

    def sinkhorn(self, x0):
        """
        Sinkhorn-Knopp algorithm
        :param x0: matrix with positive entries
        :return: doubly stochastic matrix
        """
        raise NotImplemented

    def retraction(self, x, v):
        """
        Eqn 34 of "
        :param x: point on the manifold at which the tangent is computed
        :param v: point to be retracted on the manifold (written "xi" in the paper)
        :return: point on the manifold
        """
        return self.sinkhorn(x * exp(div(v, x)))



