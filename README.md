## minimum bipartite matching via Riemann optimization

Instead of a scipy one-liner (linear sum assignment), we take the panoramic route and formulate it as an optimization
problem over the manifold of doubly-stochastic matrices, hoping to end up in a corner of the Birkhoff polytope.

If it works I'll write a blog post about it