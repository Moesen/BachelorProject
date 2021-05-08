import numpy as np
from tqdm import tqdm

# euclidian distance np.linalg.norm(a-b) 
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
# Creates a distance matrix based on euclidian distance
def create_dist_matrix(arr, show_progress=True):
    dist_matrix = np.zeros((len(arr), len(arr)), dtype=np.float32)
    rng = tqdm(range(len(arr))) if show_progress else range(len(arr))
    for row in rng:
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
        if val == 0: break
        dist = np.linalg.norm(val - pnt)
        if dist < min_dist:
            min_dist = dist
            idx = i
    return (min_dist, idx)


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