import torch as t
from mctorch.manifolds import DoublyStochastic
from mctorch.parameter import Parameter
from mctorch.optim import rSGD
import numpy as np
import numpy.random as npr
# from scipy.optimize import linear_sum_assignment
from munkres import Munkres

n = 20
nIter = 20

# # cost matrix
costsNumpy = np.abs(npr.randn(n,n) * 100)
# print(f'costs: {costsNumpy}')
costs = t.from_numpy(costsNumpy).to(t.float32)
# print(f'c: {c.dim()}')

# reference assignment with munkres
assignments = Munkres().compute(costsNumpy)
totalCost = 0
for r, c in assignments:
    value = costsNumpy[r, c] # costsNumpy[row][column]
    print(f'row {r}, col {c}: {value}')
    totalCost += value
print(f'Munkres: total cost = {totalCost}')

# 1. Initialize Parameter
x = Parameter(manifold=DoublyStochastic(n,n))
# print(f'types: x: {x.dtype}, c: {c.dtype}')
# print(f'x: {x.shape}')

def cost(xi:t.Tensor):
    """
    :param xi: doubly stochastic matrix. Close to optimality it should act as a permutation mtx
    :return:
    """
    return t.trace(t.einsum('ijk,jl->kl', xi, costs))

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