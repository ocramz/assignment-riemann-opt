import torch as t
from mctorch.manifolds import DoublyStochastic
from mctorch.parameter import Parameter
from mctorch.optim import rSGD
import numpy as np
import numpy.random as npr
# from scipy.optimize import linear_sum_assignment

n = 20
nIter = 50

# # cost matrix
cNumpy = np.abs(npr.randn(n,n))
c = t.from_numpy(cNumpy).to(t.float32)
# print(f'c: {c.dim()}')

# 1. Initialize Parameter
x = Parameter(manifold=DoublyStochastic(n,n))
# print(f'types: x: {x.dtype}, c: {c.dtype}')
# print(f'x: {x.shape}')

def cost(xi:t.Tensor):
    """
    :param xi: doubly stochastic matrix. Close to optimality it should act as a permutation mtx
    :return:
    """
    return t.trace(t.einsum('ijk,jl->kl', xi, c))

# 3. Optimize
optimizer = rSGD(params = [x], lr=1e-2)

cs = []  # costs
for epoch in range(nIter):
    fi = cost(x)
    cs.append(fi.data.item())
    # print(f'Cost: {fi}')
    fi.backward()
    optimizer.step()
    optimizer.zero_grad()
print(f'Cost #{epoch}: {fi.data}')