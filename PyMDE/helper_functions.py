from pygel3d import graph, hmesh
import numpy as np
from tqdm import tqdm
import threading

def disconnect_graph(gr: graph.Graph):
    for n in gr.nodes():
        for neighbour in gr.neighbors(n):
            gr.disconnect_nodes(n, neighbour)

# euclidian distance np.linalg.norm(a-b) 
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
# Creates a distance matrix based on euclidian distance
def create_dist_matrix(arr):
    dist_matrix = np.zeros((len(arr), len(arr)), dtype=np.float32)
    for row in tqdm(range(len(arr))):
        for col in range(len(arr)):
            dist = np.linalg.norm(arr[row] - arr[col])
            dist_matrix[row][col] = dist
    return dist_matrix

# Computes the distance in MNnist dimensions between input arr and input pnt
# and returns the closest point
def nearest_neighbours(arr: np.array, pnt: np.array) -> tuple((float, int)):
    min_dist = float("inf")
    idx = 0
    for i, val in enumerate(arr):
        dist = np.linalg.norm(val - pnt)
        if dist < min_dist:
            min_dist = dist
            idx = i
    return (min_dist, idx)
    

def connect_k_closest_points(g: graph.Graph, dist_matrix: np.array, k: int = 2, max_dist: float = 500):
    for idx, n in enumerate(g.nodes()):
        cur_row = dist_matrix[idx]
        closest_nodes = np.argpartition(cur_row, k+1)[:k+1]
        closest_nodes = np.delete(closest_nodes, np.where(closest_nodes==n))
        for neighbour in closest_nodes:
            dist = cur_row[neighbour]
            if dist <= max_dist: 
                g.connect_nodes(n, neighbour)


def heatdata_from_dist_matrix(dist_matrix, labels, np_func):
    heat_data = np.zeros((10, 10), dtype=float)
    for i in range(10): # Because 10 different numbers
        row_dist_idx = np.where(labels==i)[0]
        try:
            row_dist_data = dist_matrix[row_dist_idx]    
        except:
            print(row_dist_idx)
        for j in range(10):
            col_dist_idx = np.where(labels==j)[0]
            col_dist_data = row_dist_data[:,col_dist_idx]
            heat_data[i, j] = np_func(col_dist_data)//1
    return heat_data