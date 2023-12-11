## minimum bipartite matching via Riemann optimization

Instead of a scipy one-liner ( [linear sum assignment](#assign) ), we take the panoramic route and formulate it as an optimization
problem over the manifold of doubly-stochastic matrices, hoping to end up in a corner of the [Birkhoff polytope](#birkhoff).

If it works I'll write a blog post about it


### References
<a href="#assign">https://en.wikipedia.org/wiki/Assignment_problem </a>
<a href="#birkhoff">https://en.wikipedia.org/wiki/Doubly_stochastic_matrix </a>