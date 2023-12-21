import torch as t
# from mctorch.manifolds import DoublyStochastic
from doublystochastic import DoublyStochastic
from mctorch.parameter import Parameter
from mctorch.optim import rSGD, rAdagrad
import numpy as np
import numpy.random as npr
# from scipy.optimize import linear_sum_assignment
from munkres import Munkres

import logging

import matplotlib.pyplot as plt
import networkx as nx
# from networkx.algorithms import bipartite
from adjacency import BipartiteAdjacency

n = 10
nIter = 500
tol = 1e-3  # early stopping tolerance
learnRate = 2 * 1e-2
wStdev = 1e1  # weight std dev

# # cost matrix
costsNumpy = np.abs(npr.randn(n,n) * wStdev)
# print(f'costs: {costsNumpy}')
costs = t.from_numpy(costsNumpy).to(t.float32)
print(f'cost matrix C: {costs}')
# print(f'c: {c.dim()}')

# Initialize Parameter
x = Parameter(manifold=DoublyStochastic(n))
# print(f'types: x: {x.dtype}, c: {c.dtype}')
# print(f'x: {x.shape}')

# Optimizer
optimizer = rSGD(params = [x], lr=learnRate)
# optimizer = rAdagrad(params = [x])

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



# # 2. declare cost function
def cost(xi:t.Tensor):
    """ Tr(X_i C)
    :param xi: doubly stochastic matrix. Close to optimality it should act as a permutation mtx
    :return: cost :: R+
    """
    return t.trace(xi.mT @ costs)


def distanceToOptAssign(xi:t.Tensor):
    """distance to the known-optimal (Munkres) assignment"""
    xRef = adj0.tensor
    dx = xi - xRef
    return t.linalg.norm(dx, dim=(0, 1))

def rowColMeans(xi:t.Tensor):
    """ensure that row and column sums are close to 1"""
    rs = t.sum(xi, dim=1)
    cs = t.sum(xi, dim=0)
    mi = t.min(xi)
    ma = t.max(xi)
    return t.mean(rs), t.mean(cs), mi, ma

print(f'cost of Munkres assignment: {cost(adj0.tensor)}')



adj = BipartiteAdjacency(n, n, weighted=True)
# # graph layout
adjPos = nx.bipartite_layout(adj.g, nx.bipartite.sets(adj.g)[0], align='vertical')


def scaleEdgeWidth(w):
    return w * 1e2
    # return round(w * 10)

# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

cs = []  # costs
ds = []
for epoch in range(nIter):
    fi = cost(x)
    y = fi.clone().detach().data.item()  # cost at current iteration
    cs.append(y)
    xCurr = x.clone().detach().data  # current iteration

    rmean, cmean, mi, ma = rowColMeans(xCurr)  # row and column mean
    di = distanceToOptAssign(xCurr)  # distance to optimal soln
    ds.append(di)
    if t.any(t.isnan(xCurr)):
        errmsg = f'iter {epoch}: NaN'
        logging.exception(errmsg, exc_info=True)
        # break
        raise FloatingPointError(errmsg)
    # print(f'Cost: {fi}')
    fi.backward()
    optimizer.step()
    optimizer.zero_grad()

    # adjacency matrix from xCurr
    adj.fromTorch(xCurr, kLargest=1)
    nxAdjGraph = adj.g
    ws = nx.get_edge_attributes(nxAdjGraph, 'weight')
    wsScaled = [scaleEdgeWidth(w) for w in list(ws.values())]
    miw, maw = min(wsScaled), max(wsScaled)
    # # drawing
    plt.clf()
    plt.title(f'# {epoch}: cost {y:.2f}, dist to opt {di:.2f}, (row m {rmean:.2f}, col m {cmean:.2f})\n '
              f'X elems ({mi:.2f}, {ma:.2f}), edge weights ({miw:.2f}, {maw:.2f})')
    # nx.draw(nxAdjGraph, pos=adjPos)

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

    if di <= tol:
        break

    plt.pause(0.01)



print(f'Cost #{epoch}: {fi.data}')
print(f'Distance to optimality #{epoch}: {di}')
# print(f'Final X: {x.data}')


try:
    print(f'solution: {xCurr}')
    fig, ax = plt.subplots()
    iters = list(range(epoch + 1))
    ax.plot(iters, ds, linewidth=2.0)
    plt.show()

except ValueError:
    print(f'cannot plot: {len(iters)} != {len(ds)}')


