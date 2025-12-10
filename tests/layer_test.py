
import unittest
import os
import sys


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)

from Models.Layer import Layer
from Models.Value import Value




class TestLayer(unittest.TestCase):
    
    def test_initialization(self):
        layer = Layer(3, 4, activation_type='relu')
        self.assertEqual(len(layer.neurons), 4)
        for i, neuron in enumerate(layer.neurons):
            self.assertEqual(neuron.activation_type, 'relu')
            self.assertEqual(len(neuron.w), 3)
        
    @classmethod
    def setUpClass(cls):
        cls.layer = Layer(3, 2, activation_type='relu')
        cls.weights1 = [Value(0.2), Value(-0.5), Value(1.0)]
        cls.bias1 = Value(0.0)
        cls.weights2 = [Value(-1.5), Value(2.0), Value(0.5)]
        cls.bias2 = Value(1.0)
        cls.layer.neurons[0].w = cls.weights1
        cls.layer.neurons[0].b = cls.bias1
        cls.layer.neurons[1].w = cls.weights2
        cls.layer.neurons[1].b = cls.bias2
        
    @classmethod
    def tearDownClass(cls):
        cls.layer = None
        cls.weights1 =None
        cls.bias1 =None
        cls.weights2 = None
        cls.bias2 = None
    
    def test_call(self):
        x = [1, 2, 3]
        
        output = self.layer(x)
        
        expected_output1 = max(0, 0.2*1 + -0.5*2 + 1.0*3 + 0.0)
        expected_output2 = max(0, -1.5*1 + 2.0*2 + 0.5*3 + 1.0)
        
        self.assertEqual(output[0].data, expected_output1)
        self.assertEqual(output[1].data, expected_output2)
    
    def test_parameters(self):
        params = self.layer.parameters()
        self.assertEqual(len(params), 8)  # (3 weights + 1 bias per neuron) * 2 neurons
        for neuron in self.layer.neurons:
            self.assertIn(neuron.b, params)
            for w in neuron.w:
                self.assertIn(w, params)

if __name__ == '__main__':
    unittest.main()