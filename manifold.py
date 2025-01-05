class Manifold:
    def __init__(self):
        raise NotImplemented
    def size(self):
        raise NotImplemented
    def rand(self):
        """
        sample a random point on the manifold
        :return: random point on the manifold
        """
        raise NotImplemented
    def proj(self, x, z):
        """
        orthogonal projection on the tangent
        :param x: point on the manifold
        :param z: point to be projected
        :return: point on the tangent
        """
        raise NotImplemented
    def retr(self, x, v):
        """
        retraction on the manifold
        :param x: point on the manifold
        :param v: point to be retracted
        :return: point on the manifold
        """
        raise NotImplemented
    def egrad2rgrad(self, x, u):
        """
        Euclidean gradient to Riemann gradient
        :param x: point on the manifold
        :param u: gradient vector
        :return: projected gradient
        """
        raise NotImplemented