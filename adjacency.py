import numpy as np
import numpy.random as npr
import torch as t
# from heapq import nlargest
# from bottleneck import argpartition
import networkx as nx
from networkx.algorithms import bipartite

# B = nx.Graph()
# # B.add_nodes_from([1, 2, 3, 4], bipartite=0)
# # B.add_nodes_from(["a", "b", "c"], bipartite=1)
# # B.add_edges_from([(1, "a"), (1, "b"), (2, "b"), (2, "c"), (3, "c"), (4, "a")])

def argSort(xs):
    """
    return a sorted list of elements together with their original indices
    :param xs:
    :return:
    """
    return sorted(list(enumerate(xs)), key=lambda x:x[1], reverse=True)

class BipartiteAdjacency:
    def __init__(self, m:int, n:int, weighted):
        """
        initialize a bipartite graph
        :param n:
        :param weighted:
        """
        self.g = nx.Graph()
        self.tensor = t.Tensor()
        self.weighted = weighted
        self.m = m  # size of left set
        self.n = n  # " right set
        for i in range(m):
            self.add_node(i, 'left')
            for j in range(n):
                self.add_node(j, 'right')
                self.add_edge(i, j, 1)
    def add_node(self, ix, side):
        if side == 'left':
            self.g.add_node(ix, bipartite=0)
        else:
            self.g.add_node(ix + self.m, bipartite=1)
    def add_edge(self, ix, jx, weight):
        if self.weighted:
            self.g.add_edge(ix, jx + self.m, weight=weight)
        else:
            self.g.add_edge(ix, jx + self.m)
    def __str__(self):
        return f'Adjacency'
    def mapEdgeWeight(self, w):
        return
    def fromTorch(self, tt: t.Tensor, kLargest:int = 3):
        self.tensor = tt
        self.g.remove_edges_from(self.g.edges())  # clear all edges
        for i, row in enumerate(tt):
            topK = argSort(row.numpy())[:kLargest]
            for j, c in topK:
                self.add_edge(i, j, c)
    def fromEdges(self, ixs):
        """
        :param ixs: iterable of (i, j, cost)
        :return:
        """
        self.g.remove_edges_from(self.g.edges())
        self.tensor = t.zeros(size=(self.m, self.n))
        for (i, j, c) in ixs:
            if self.weighted:
                self.tensor[i, j] = c
                self.add_edge(i, j, c)
            else:
                self.tensor[i, j] = 1
                self.add_edge(i, j, 1)
