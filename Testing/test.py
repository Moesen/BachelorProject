import sys
sys.path.append("E:\GIT\Bachelor\PyMDE")

import unittest
import numpy as np
import pymde
import helper_functions as hf



class TestDistMatrix(unittest.TestCase):
    # Setting up distance_matrix
    
    def __init__(self, *args, **kwargs):
        super(TestDistMatrix, self).__init__(*args, **kwargs)
        self.mnist = pymde.datasets.MNIST()
        self.data = np.array(self.mnist.data).astype("float")

    def test_symmetry(self):
        cropped_data = self.data[:100]
        dist_matrix = hf.create_dist_matrix(cropped_data)
        self.assertTrue(np.allclose(dist_matrix.transpose(), dist_matrix))

    

if __name__ == "__main__":
    unittest.main()
