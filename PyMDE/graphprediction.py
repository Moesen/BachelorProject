import torch
from pygel3d import hmesh, graph, gl_display as gd
import numpy as np
import pymde
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import helper_functions as hf
import sys
import os
import seaborn as sns
import pandas as pd

class GraphAnalysis:
    
    # Loads mnist numbers.
    # Only loads certain labels, if labels specified (Default=ALL)
    # Only loads a certain amount if limit set (Default=1000)
    def __init__(self, limit=1000, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], preserve_neighbours=True, dist_matrix=None):
        mnist = pymde.datasets.MNIST()
        self._mnist = mnist
        self.limit = limit
        mnist_labels = mnist.attributes["digits"][:limit]
        self._labels = mnist_labels[np.in1d(mnist_labels, labels)]
        
        unembedded = np.array(mnist.data)[:limit]
        self._unembedded = unembedded[np.in1d(mnist_labels, labels)]

        if "tensor.pt" in os.listdir():
            embedded = torch.load("tensor.pt").numpy()[:limit] 
        else:
            if preserve_neighbours: mde = pymde.preserve_neighbors(mnist.data, verbose = True, embedding_dim=3)
            else: mde = pymde.preserve_distances(mnist.data, verbose=True, embedding_dim=3)
            
            embedded = mde.embed(verbose=True)
            torch.save(embedded, "tensor.pt")
        self._embedded = embedded[np.in1d(mnist_labels, labels)]

        # Creating graph
        self._g = self._create_graph()
        pos = self._g.positions()
        for n, ppos in zip(self._g.nodes(), self._embedded):
            pos[n] = ppos

        # Creating dist_matrix
        if dist_matrix is not None:
            self._dist_matrix = dist_matrix
        else:
            self._dist_matrix = hf.create_dist_matrix(self._unembedded)

    def _create_graph(self):
            g = self._g = graph.Graph()
            for p in np.copy(self._embedded):
                npp = np.float64(p)
                self._g.add_node(npp)
            return g
    
    # Stats generation
    def show_numbers(self, start_idx, end_idx):
        # Plotting mnist images
        data = self._unembedded[start_idx:end_idx]
        rows, cols = max(1, len(data)//5), 5
        arr = data
        # Plotting
        fig, axes = plt.subplots(rows, cols, figsize=(1.5*cols, 2*rows))
        for i in range(len(data)):
            if rows > 1:
                ax = axes[i//cols, i%cols]
            else:
                ax = axes[i%cols]
            ax.imshow(arr[i].reshape(28, 28), cmap="gray")
            ax.set_title(f"Label: {self._labels[i+start_idx]}")
        plt.tight_layout()
        plt.show()

    def generate_heat_data(self, np_func):
        return hf.heatdata_from_dist_matrix(self._dist_matrix, self._labels, np_func)

    # Graph manipul
    def disconnect_graph(self): hf.disconnect_graph(self._g)
    def connect_graph(self, neighbours=3, max_dist=1700): 
        hf.connect_k_closest_points(self._g, self._dist_matrix, k=neighbours, max_dist=max_dist)
    def skeletonize_graph(self): 
        self._skeleton = graph.LS_skeleton(self._g)
        tree = KDTree(self._embedded)
        self._skeleton.cleanup()

        skel_labels = np.zeros(len(self._skeleton.nodes()))
        skel_non_embed = np.zeros((len(self._skeleton.nodes()), 784))

        pos = self._skeleton.positions()
        for n in self._skeleton.nodes():
            _, idx = tree.query(pos[n])
            skel_labels[n] = self._labels[idx]
            skel_non_embed[n] = self._unembedded[idx]

        self._skel_labels = skel_labels
        self._skel_unembed_pos = skel_non_embed


    # Fields
    def labels(self):       return self._labels
    def embedded(self):     return self._embedded
    def unembedded(self):   return self._unembedded
    def g(self):            return self._g
    def dist_matrix(self):  return self._dist_matrix
    def skeleton(self):    
        if self._skeleton:
            return self._skeleton
    
    def save_graph(self):
        graph.save("graph.graph", self._g)
    def save_skeleton(self):
        graph.save("skeleton.graph", self._skeleton)

    def save_distmatrix(self):
        np.savetxt("distmatrix.csv", self._dist_matrix, delimiter=",")
    @staticmethod
    def load_distmatrix():
        return np.loadtxt("distmatrix.csv", delimiter=",")

    # Accurary testing
    def test_accuracy(self, test_size=1000):
        Y_test = self._mnist.attributes["digits"][self.limit:self.limit + test_size]
        x_test = self._mnist.data[self.limit:self.limit + test_size].numpy()
        guess = list()
        print("Testing accurazy of skeleton")
        for idx, val in tqdm(enumerate(x_test)):
            dist, nearest_idx = hf.nearest_neighbours(self._skel_unembed_pos, val)
            guess.append((self._skel_labels[nearest_idx], Y_test[idx]))
        
        tru, fal = 0, 0
        for gue in guess:
            if gue[0] == gue[1]: tru+=1
            else: fal+= 1
        
        accuracy = (tru)/(tru+fal)
        print(accuracy)

if __name__ == "__main__":
    ganal = GraphAnalysis(limit=20000)
    
    ganal.connect_graph(max_dist=1700, neighbours=1)
    ganal.skeletonize_graph()
    ganal.test_accuracy()
    