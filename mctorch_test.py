import torch as t
from mctorch.manifolds import DoublyStochastic
from mctorch.parameter import Parameter
from mctorch.optim import rSGD, rAdagrad
import numpy as np
import numpy.random as npr
# from scipy.optimize import linear_sum_assignment
from munkres import Munkres

import matplotlib.pyplot as plt
import networkx as nx
# from networkx.algorithms import bipartite
from adjacency import BipartiteAdjacency

n = 5
nIter = 300

# # cost matrix
costsNumpy = np.abs(npr.randn(n,n) * 1e2)
# print(f'costs: {costsNumpy}')
costs = t.from_numpy(costsNumpy).to(t.float32)
# print(f'c: {c.dim()}')

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
adj0 = BipartiteAdjacency(n, n, weighted=False)
adj0.fromEdges(aEdges)
# print(f'adj0 : {type(adj0)}')

# 1. Initialize Parameter
x = Parameter(manifold=DoublyStochastic(n,n))
# print(f'types: x: {x.dtype}, c: {c.dtype}')
# print(f'x: {x.shape}')

# # 2. declare cost function
def cost(xi:t.Tensor):
    """ Tr(X_i C)
    :param xi: doubly stochastic matrix. Close to optimality it should act as a permutation mtx
    :return: cost :: R+
    """
    if xi.dim() == 3:
        return t.trace(t.einsum('ijk,jl->kl', xi, costs))  # ignore first dimension since we only use batch size == 1
    else:
        return t.trace(t.einsum('jk,jl->kl', xi, costs))

def distanceToOptAssign(xi:t.Tensor):
    """distance to the known-optimal (Munkres) assignment"""
    xRef = adj0.tensor
    dx = xi - xRef
    return t.linalg.norm(dx, dim=(0, 1))


# print(f'cost of Munkres assignment: {cost(adj0.tensor)}, {adj0.tensor}')

# 3. Optimize
# optimizer = rSGD(params = [x], lr=1e-2)
optimizer = rAdagrad(params = [x])

adj = BipartiteAdjacency(n, n, weighted=True)
# # graph layout
adjPos = nx.bipartite_layout(adj.g, nx.bipartite.sets(adj.g)[0], align='vertical')

def scaleEdgeWidth(w):
    return - np.log(w) / 5

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
cs = []  # costs
ds = []
for epoch in range(nIter):
    fi = cost(x)
    cs.append(fi.data.item())
    di = distanceToOptAssign(x.detach().clone().data[0,:,:])
    ds.append(di)
    # print(f'Cost: {fi}')
    fi.backward()
    optimizer.step()
    optimizer.zero_grad()
    # adjacency
    adj.fromTorch(x.detach().clone().data[0,:,:], kLargest=2)
    nxAdjGraph = adj.g
    # # drawing
    plt.clf()
    plt.title(f'Iter {epoch}')
    # nx.draw(nxAdjGraph, pos=adjPos)
    ws = nx.get_edge_attributes(nxAdjGraph, 'weight')
    wsScaled = [scaleEdgeWidth(w) for w in list(ws.values())]
    # print(f'edge weights: {wsScaled}')

    # # # draw reference (Munkres) solution
    # nx.draw_networkx_edges(nxAdjGraph, pos=adjPos,
    #                        edgelist=nx.get_edge_attributes(adj0.g, 'weight').keys(),
    #                        edge_color='k',
    #                        width=0.5)

    nx.draw_networkx_edges(nxAdjGraph, pos=adjPos,
                           edgelist=ws.keys(),
                           width=wsScaled,
                           edge_color='blue',
                           alpha=0.6
                           )

    plt.pause(0.01)


print(f'Cost #{epoch}: {fi.data}')
print(f'Distance to optimality #{epoch}: {di}')
# print(f'Final X: {x.data}')

fig, ax = plt.subplots()
ax.plot(list(range(nIter)), ds, linewidth=2.0)
plt.show()