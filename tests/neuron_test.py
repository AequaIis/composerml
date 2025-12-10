import unittest
import os
import sys
from math import tanh

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)


from Models.Neuron import Neuron
from Models.Value import Value

class TestLayer(unittest.TestCase):
    
    def test_initialization(self):
        neuron = Neuron(3)
        self.assertEqual(len(neuron.w), 3)
        self.assertIsInstance(neuron.b, Value)
        self.assertEqual(neuron.activation_type, 'tanh')
        
    @classmethod
    def setUpClass(cls):
        cls.neuron_tanh = Neuron(3, activation_type='tanh')
        cls.neuron_relu = Neuron(3, activation_type='relu')
        cls.weights = [Value(0.5), Value(-1.0), Value(2.0)]
        cls.bias = Value(0.0)   
        cls.neuron_tanh.w = cls.weights
        cls.neuron_tanh.b = cls.bias
        cls.neuron_relu.w = cls.weights
        cls.neuron_relu.b = cls.bias
    
        
    @classmethod
    def tearDownClass(cls):
        cls = None
    
    def test_call(self):
        x= [1,2,3]
        
        output1 = self.neuron_relu(x)
        expected_output = max(0, 0.5*1 + -1.0*2 + 2.0*3 + 0.0)
        self.assertEqual(output1.data, expected_output)  
    
    
        output2 = self.neuron_tanh(x)
        expected_output = tanh(0.5*1 + -1.0*2 + 2.0*3 + 0.0)
        self.assertAlmostEqual(output2.data, expected_output, places=5)
   
   
    def test_parameters(self):
        params = self.neuron_tanh.parameters()
        self.assertEqual(len(params), 4)  # 3 weights + 1 bias
        self.assertIn(self.neuron_tanh.b, params)
        for w in self.neuron_tanh.w:
            self.assertIn(w, params)
    
if __name__ == '__main__':
    unittest.main()
        