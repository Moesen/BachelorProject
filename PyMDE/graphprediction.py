from os import path
from seaborn.palettes import color_palette
import torch
from pygel3d import graph, gl_display as gd
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
from collections import Counter
import pathlib
import json

class GraphAnalysis:
    _cur_path = pathlib.Path(__file__).parent.absolute()
    _results_path = os.path.join(_cur_path, "Results/")
    _dist_matrix_path = os.path.join(_cur_path, "DistMatrix/")


    # Loads mnist numbers.
    # Only loads certain labels, if labels specified (Default=ALL)
    # Only loads a certain amount if limit set (Default=1000)
    def __init__(self,  limit:int = 1000, 
                        labels:np.array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                        preserve_neighbours:bool = True, 
                        dist_matrix:np.array = None,
                        forget_labels:float = None,
                        ):

        print("Loading dataset")
        mnist = pymde.datasets.MNIST()
        self._mnist = mnist
        self._limit = limit
        
        mnist_labels = mnist.attributes["digits"][:limit].numpy()
        if forget_labels is not None:
            np.random.seed(1)
            n = int(mnist_labels.shape[0]*forget_labels)
            random_indices = np.random.choice(mnist_labels.shape[0], n, replace=False)
            mnist_labels[random_indices] = -1

        self._labels = mnist_labels[np.in1d(mnist_labels, labels)]

        # Important otherwise distances will be uint
        unembedded = np.array(mnist.data)[:limit].astype("float")
        self._unembedded = unembedded[np.in1d(mnist_labels, labels)]

        if "tensor.pt" in os.listdir():
            embedded = torch.load("tensor.pt").numpy()[:limit] 
        else:
            if preserve_neighbours: mde = pymde.preserve_neighbors(mnist.data, verbose = True, embedding_dim=3)
            else: mde = pymde.preserve_distances(mnist.data, verbose=True, embedding_dim=3)
            
            embedded = mde.embed(verbose=True)
            torch.save(embedded, "tensor.pt")
        self._embedded = embedded[np.in1d(mnist_labels, labels)]

        print("Creating Graph")

        # Creating graph
        self._g = self._create_graph()
        pos = self._g.positions()
        for n, ppos in zip(self._g.nodes(), self._embedded):
            pos[n] = ppos

        # Creating dist_matrix
        if dist_matrix is not None:
            self._dist_matrix = dist_matrix
            print("Loaded dist matrix")
        else:
            print("Creating dist matrix")
            arr = self._unembedded
            self._dist_matrix = hf.create_dist_matrix(arr)
        try:
            print("Symmetric dist_matrix", np.allclose(self._dist_matrix.transpose(), self._dist_matrix))
        except:
            print("Not enough space to check symmetry")
        self.print_seperator_line()

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

    def show_number(self, arr):
        plt.imshow(arr.reshape(28, 28), cmap="gray")
        plt.show()

    def show_heatmap(self, np_func):
        data = hf.heatdata_from_dist_matrix(self._dist_matrix, self._labels, np_func)
        sns.heatmap(data, cmap="PuBu", annot=True, fmt="g")
        plt.show()

    # Graph manipul
    def disconnect_graph(self): hf.disconnect_graph(self._g)
    def connect_graph(self, neighbours=3, max_dist=1700): 
        self._neighbours = neighbours
        self._max_dist = max_dist
        hf.connect_k_closest_points(self._g, self._dist_matrix, k=neighbours, max_dist=max_dist)
    def skeletonize_graph(self): 
        self._skeleton, self._skel_map = graph.LS_skeleton_and_map(self._g) 
        

    # Current method 
    def relabel_skeleton(self):
        # Empty array for storing the new labels for each index of the skeleton
        self._skel_labels = np.zeros(len(self._skeleton.nodes()))
        self._skel_non_embed = np.zeros((len(self._skeleton.nodes()), 784))

        skel_acum = {x: {} for x in range(0, len(self._skeleton.nodes()))}

        # Gathering closest labels
        print("G nodes and skel_map same length: " + str(len(self._g.nodes()) == len(self._skel_map)))
        for g_node, skel_n in zip(self._g.nodes(), self._skel_map):
            lbl = self._labels[g_node]
            if skel_n >= 0:
                if lbl in skel_acum[skel_n].keys():
                    skel_acum[skel_n][lbl].append(g_node)
                else:
                    skel_acum[skel_n][lbl] = [g_node]


        naughty_boys = []
        # Checking most common labels
        # If count > 1 or == 0 there is either more than 1 common label on that vertex, or none at all
        # If more than one, different labels have been packed together
        # If none it is helper vertex
        # Both needs to be propogated
        
        for skel_n, skel_maps in skel_acum.items():
            # If no nodes helper vertex that needs to be propogated
            if len(skel_maps) == 0: 
                naughty_boys.append(skel_n)
                break
            
            mx = 0
            nds = []
            for label, g_nodes in skel_maps.items():
                if len(g_nodes) > mx:
                    mx = len(g_nodes)
                    nds = [{label: g_nodes}]
                elif len(g_nodes) == mx:
                    nds.append({label: g_nodes})

            # If more than one label, vertex needs to be propogated
            if len(nds) > 1: 
                naughty_boys.append(skel_n)
                break
            
            label = list(nds[0].keys())[0]
            g_ids = list(nds[0].values())[0]
            self._skel_labels[skel_n] = label
            avg = self._avg_position(g_ids)
            self._skel_non_embed[skel_n] = avg

    def _avg_position(self, node_ids):
        positions = [self._unembedded[node_id] for node_id in node_ids]
        return np.average(np.array(positions), axis=0)
        

    # MISC
    def print_seperator_line(self):
        print("".join("=" for _ in range(40)))
        print("".join("=" for _ in range(40)))
    
    def _view_graph(self, g: graph.Graph):
        viewer = gd.Viewer()
        viewer.display(g)

    def view_graph(self): self._view_graph(self._g)
    def view_skele(self): self._view_graph(self._skeleton)

    # Accurary testing
    def test_accuracy(self, test_size=1000, show=True):
        Y_test = self._mnist.attributes["digits"][self._limit:self._limit + test_size]
        x_test = self._mnist.data[self._limit:self._limit + test_size].numpy()
        guess = list()
        print("Testing accurazy of skeleton")
        for idx, val in tqdm(enumerate(x_test)):
            dist, nearest_idx = hf.nearest_neighbours(self._skel_non_embed, val)
            guess.append((self._skel_labels[nearest_idx], Y_test[idx]))

                
        tru, fal = 0, 0
        for gue in guess:
            if gue[0] == gue[1]: tru+=1
            else: fal+= 1
        self._test_accuracy = (tru)/(tru+fal)

        conf_matrix = np.zeros((10, 10))
        for gue in guess:
            predic = int(gue[0])
            truth = int(gue[1])
            conf_matrix[truth, predic] += 1
        
        self._conf_matrix = conf_matrix    
        if show: 
            plt.show()

    def visualize_tests(self,  test_size, show=True, save=True, filename=None):
        if not hasattr(self, "_conf_matrix"):
            raise Exception("Tests not conducted yet.")        

        test_size = np.sum(self._conf_matrix)
        df_cm = pd.DataFrame(self._conf_matrix, range(10), range(10))
        sns.set(font_scale=1.2)
        plt.suptitle(f"Accurazy: {self._test_accuracy}")
        plt.xlabel("Prediction")
        plt.ylabel("Truth")

        color_map = sns.color_palette("PuBu")
        sns.heatmap(df_cm,
                    annot=True, 
                    cmap=color_map, 
                    annot_kws={"size": 14}, 
                    fmt="g").set_title(f"Training size: {self._limit}, Testing size {test_size}")
        if filename is None:    name = "PredictHeatmap" + "_Num" + str(self._limit) + "_TestPoints" + str(int(test_size))
        else:                   name = filename

        files = os.listdir(self._results_path)
        files = list(filter(lambda x: name in x, files))
        cnt = len(files)
        name += "_" + str(cnt)

        if save: 
            plt.savefig(self._results_path + name + ".png", format="png")
            np.savetxt(self._results_path + name + ".csv", self._conf_matrix, delimiter=",")
        if show: plt.show()


    # Fields
    def labels(self):       return self._labels
    def embedded(self):     return self._embedded
    def unembedded(self):   return self._unembedded
    def g(self):            return self._g
    def dist_matrix(self):  return self._dist_matrix
    def skeleton(self):    
        if hasattr(self, "_skeleton"):
            return self._skeleton
        else:
            raise Exception("Skeleton not created yet")
    
    def save_graph(self):
        graph.save("graph.graph", self._g)
    def save_skeleton(self):
        graph.save("skeleton.graph", self._skeleton)
        
    def save_distmatrix(self):
        np.savetxt(f"{self._dist_matrix_path}distmatrix_{self._limit}.csv", self._dist_matrix, delimiter=",")
    @staticmethod
    def load_distmatrix(path):
        return np.loadtxt(path, delimiter=",")


if __name__ == "__main__":
    ganal = GraphAnalysis(limit=1000)

    ganal.connect_graph(max_dist=1700, neighbours=2)
    ganal.skeletonize_graph()
    ganal.relabel_skeleton()
    # ganal.show_heatmap(np_func=np.average)
    test_size = 1000
    ganal.test_accuracy(test_size=test_size, show=False)
    ganal.visualize_tests(show=False, test_size=test_size, save=True, filename="test")
    ganal.save_skeleton()