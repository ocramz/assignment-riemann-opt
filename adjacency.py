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
    return sorted(list(enumerate(xs)), key=lambda x:x[1])

class BipartiteAdjacency:
    def __init__(self, m:int, n:int, weighted=False):
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
            self.g.add_node(i, bipartite=0)  # add to left set
            for j in range(m+n, m*n):  # node indices must be unique
                self.g.add_node(j, bipartite=1)
                self.g.add_edge(i, j)
    def __str__(self):
        return f'Adjacency'
    def fromTorch(self, tt: t.Tensor, kLargest:int = 3):
        self.tensor = tt
        for i, row in enumerate(tt):
            topK = argSort(row.numpy())[:kLargest]
            for j, c in topK:
                if self.weighted:
                    self.g.add_edge(i, j, cost=c)
                else:
                    self.g.add_edge(i, j)
    def fromEdges(self, ixs):
        """
        :param ixs: iterable of (i, j, cost)
        :return:
        """
        self.tensor = t.zeros(size=(self.m, self.n))
        for (i, j, c) in ixs:
            if self.weighted:
                self.tensor[i, j] = c
                self.g.add_edge(i, j, cost=c)
            else:
                self.tensor[i, j] = 1
                self.g.add_edge(i, j)
