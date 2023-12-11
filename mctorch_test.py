import torch as t
from mctorch.manifolds import DoublyStochastic
from mctorch.parameter import Parameter
from mctorch.optim import rSGD
import numpy as np
import numpy.random as npr
# from scipy.optimize import linear_sum_assignment
from munkres import Munkres

import matplotlib.pyplot as plt
import networkx as nx
# from networkx.algorithms import bipartite
from adjacency import Adjacency

n = 10
nIter = 200

# # cost matrix
costsNumpy = np.abs(npr.randn(n,n) * 1e2)
# print(f'costs: {costsNumpy}')
costs = t.from_numpy(costsNumpy).to(t.float32)
# print(f'c: {c.dim()}')

B = nx.Graph()
# B.add_nodes_from([1, 2, 3, 4], bipartite=0)
# B.add_nodes_from(["a", "b", "c"], bipartite=1)
# B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])


# # reference assignment with munkres
assignments = Munkres().compute(costsNumpy.copy())  # NB use copy() since Munkres mutates input vector
totalCost = 0
aEdges = []
for r, c in assignments:
    value = costsNumpy[r, c] # costsNumpy[row][column]
    aEdges.append((r, c, value))
    # print(f'row {r}, col {c}: {value}')
    totalCost += value
print(f'Munkres: total cost = {totalCost}')
adj0 = Adjacency()
adj0.fromEdges(aEdges, n, n)
# print(f'adj0 : {type(adj0)}')

# 1. Initialize Parameter
x = Parameter(manifold=DoublyStochastic(n,n))
# print(f'types: x: {x.dtype}, c: {c.dtype}')
# print(f'x: {x.shape}')

def cost(xi:t.Tensor):
    """
    :param xi: doubly stochastic matrix. Close to optimality it should act as a permutation mtx
    :return:
    """
    if xi.dim() == 3:
        return t.trace(t.einsum('ijk,jl->kl', xi, costs))  # Tr(X_i C)
    else:
        return t.trace(t.einsum('jk,jl->kl', xi, costs))  # Tr(X_i C)

# print(f'cost of Munkres assignment: {cost(adj0.tensor)}, {adj0.tensor}')

# 3. Optimize
optimizer = rSGD(params = [x], lr=1e-2)

adj = Adjacency()

fig, ax = plt.subplots()
cs = []  # costs
for epoch in range(nIter):
    fi = cost(x)
    cs.append(fi.data.item())
    # print(f'Cost: {fi}')
    fi.backward()
    optimizer.step()
    optimizer.zero_grad()
    # adjacency
    adj.fromTorch(x.detach().clone().data[0,:,:], kLargest=1)
    nxAdjGraph = adj.g
    # drawing
    plt.clf()
    plt.title(f'Iter {epoch}')
    nx.draw(nxAdjGraph, pos=nx.spring_layout(nxAdjGraph))
    # plt.pause(1)
    plt.show()

print(f'Cost #{epoch}: {fi.data}')
# print(f'Final X: {x.data}')

# fig, ax = plt.subplots()
# ax.plot(list(range(nIter)), cs, linewidth=2.0)
# plt.show()