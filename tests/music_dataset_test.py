import unittest
import numpy as np
from pathlib import Path
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)

sys.path.insert(0, str(Path(__file__).parent.parent))

from Music_Generation.music_dataset import MusicDataset

class TestMusicDataset(unittest.TestCase):

    def setUp(self):
        self.ds = MusicDataset.__new__(MusicDataset)
        self.ds.context_length = 3 

    def tearDown(self):
        self.ds = None

    def test_build_sequences(self):
        songs = [[10, 20, 30, 40, 50], [3, 6, 9, 12]]
        X, Y = self.ds._build_sequences(songs, context_length=3)

        expected_X = [[10,20,30],[20,30,40],[30,40,50],[3,6,9],[6,9,12]]
        expected_Y = [40, 50, 0, 12, 0]
        self.assertEqual(X, expected_X)
        self.assertEqual(Y, expected_Y)

    def test_one_hot(self):
        seqs = [[0, 1, 5],[10, 3, 2]]

        oh = self.ds._one_hot(seqs)
        self.assertEqual(oh.shape, (2, 3, 128))

        self.assertTrue(np.array_equal(oh[0][0], np.eye(128)[0]))
        self.assertTrue(np.array_equal(oh[0][1], np.eye(128)[1]))
        self.assertTrue(np.array_equal(oh[0][2], np.eye(128)[5]))
        self.assertTrue(np.array_equal(oh[1][0], np.eye(128)[10]))
