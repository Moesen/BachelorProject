# Data Analysis with skeletons calculated from graphs
## Introduction
In Data Science, there is a lot of different ways to process data, and here is yet another one. The general idea, is to take some data, and in one way or the other implement a heuristic that calculates the distance between two points in the data. <br>

One example of this, which will be used as the baseline for this project, are the mnist numbers. Here the distance between each datapoint (picture) could just be the difference in pixels, found by multiplying the matrices of the pixelvalues together, but it might be, that this is not good enough, and some thinking is needed <br>

When this the distance between all vertices has been found in some way or the other, the next step will be to create a skeleton from this graph. In this case it will be a graph, where if we have *n* nodes, each node will have *n-1* edges. To create the skeleton the package PyGEL will be used, and more specifically the part that creates skeletons. <br>

Finally this method can be used in semisupervised learning to hopefully reconstruct labels for some of the data.
