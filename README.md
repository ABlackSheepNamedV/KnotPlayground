# KnotPlayground
A collection of code snippets for looking at and playing with knots.

Currently, this repository only contains KnotDrawing.py, which contains a simple class for relaying the Alexander polynomial (a knot invaraint) from an image of a knot.

At the moment, the code can handle knots but not (knot?) handle links. There appears to be the multivariate Alexander Polynomial, but I'm not quite sure how that works yet.

I'd also like to form some way of relating similar knots together. For example, knots $9_1$, $9_2$, $9_3$, an $9_4$ look similar to one another and feel like they form a sequence of knots that are just off by a single crossing between each successive pair. I wonder if there's a way to construct a graph where each vertex is a knot and edges are drawn between similar pairs of knots...