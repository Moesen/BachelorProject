import numpy as np
import pymde
import helper_functions_hpc as hf
from scipy.spatial import KDTree
import multiprocessing


def find_closest(start_idx: int, end_idx: int, neighbours, dist_matrix, k, max_dist, Q: multiprocessing.Queue):
    for i in range(start_idx, end_idx):
        cur_row = dist_matrix[i]
        closest_nodes = np.argpartition(cur_row, k+1)[:k+1]
        closest_nodes = np.delete(closest_nodes, np.where(closest_nodes==i))
        idx = 0
        for node in closest_nodes:
            dist = cur_row[node]
            if dist <= max_dist:
                Q.put((i, idx, node))
                idx += 1


if __name__ == "__main__":
    lim = 1000
    mnist = pymde.datasets.MNIST()
    unembedded = np.array(mnist.data)[:lim]
    dist_matrix = hf.create_dist_matrix(unembedded, show_progress=False)
    
    k = 3
    max_dist = 1700
    neighbours = np.zeros((len(unembedded), k), dtype=int)
    num = 4

    Q = multiprocessing.Queue()

    threads = []
    for i in range(num):
        size = lim//num
        start = i * size
        end = min(i * size + size, 1000)
        p = multiprocessing.Process(target=find_closest, args=(start, end, neighbours, dist_matrix, k, max_dist, Q))
        p.start()
        print("Started thread: ", p.name)
        threads.append(p)
    
    for p in threads:
        p.join()
        print("Joined thread: ", p.name)
    # Finding closest points
    
    o_neighbours = np.loadtxt("closestneighbours.csv", delimiter=",")
    print(Q.get())

    # np.savetxt("closestneighbours.csv", neighbours, delimiter=",")