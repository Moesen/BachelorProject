import numpy as np
import pymde
import helper_functions as hf
from scipy.spatial import KDTree

if __name__ == "__main__":
    mnist = pymde.datasets.MNIST()
    unembedded = np.array(mnist.data)[:70000]
    dist_matrix = hf.create_dist_matrix(unembedded)
    
    k = 3
    max_dist = 1700
    neighbours = np.zeros((len(unembedded), k), dtype=int)

    # Finding closest points
    for i in range(len(unembedded)):
        cur_row = dist_matrix[i]
        closest_nodes = np.argpartition(cur_row, k+1)[:k+1]
        closest_nodes = np.delete(closest_nodes, np.where(closest_nodes==i))
        idx = 0
        for node in closest_nodes:
            dist = cur_row[node]
            if dist <= max_dist:
                neighbours[i, idx] = node
                idx += 1
    
    np.savetxt("closestneighbours.csv", neighbours, delimiter=",")