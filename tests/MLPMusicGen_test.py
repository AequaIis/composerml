import unittest
import os
import sys
import numpy as np

from numpy import exp


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)

from Models.MLPMusicGen import MLPMusicGen
from Models.Value import Value


class TestMLPMusicGen(unittest.TestCase):
    def test_initialization(self):
        mlp_music_gen = MLPMusicGen(context_length=10, hidden_sizes=[64, 64], activation_type="tanh")
        self.assertEqual(mlp_music_gen.context_length, 10)
        self.assertEqual(len(mlp_music_gen.layers), 3)  # 2 hidden layers + 1 output layer
        self.assertEqual(mlp_music_gen.input_dim, 1280)  # context_length * 128
        
        
    def test_onehot_encoding(self):
        mlp_music_gen = MLPMusicGen()
        indices = np.array([0, 1, 2])
        onehot = mlp_music_gen._make_onehot(indices, total=3)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        np.testing.assert_array_equal(onehot, expected)
    
