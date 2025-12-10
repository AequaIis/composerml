
import unittest
import os
import sys

from numpy import exp


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)

from Models.MLPnetwork import MLPNetwork
from Models.Value import Value


class TestMLPNetwork(unittest.TestCase):
    
    def test_initialization(self):
        mlp = MLPNetwork(input_dim=3, n_neurons=[4, 2], activation_type="relu", classification='softmax')
        self.assertEqual(len(mlp.layers), 2)
        self.assertEqual(mlp.input_dim, 3)
        self.assertEqual(mlp.classification, 'softmax')
        self.assertEqual(len(mlp.layers[0].neurons), 4)
        self.assertEqual(len(mlp.layers[1].neurons), 2)
    
    @classmethod
    def setUpClass(cls):
        cls.mlp = MLPNetwork(input_dim=2, n_neurons=[3, 2], activation_type="relu", classification='sigmoid')
        cls.weights_layer1 = [
            [Value(0.1), Value(0.2)],
            [Value(-0.1), Value(0.4)],
            [Value(0.5), Value(-0.3)]
        ]
        cls.biases_layer1 = [Value(0.0), Value(0.5), Value(-0.5)]
        
        cls.weights_layer2 = [
            [Value(0.3), Value(-0.2), Value(0.1)],
            [Value(-0.4), Value(0.6), Value(0.2)]
        ]
        cls.biases_layer2 = [Value(0.1), Value(-0.1)]
        
        for i, neuron in enumerate(cls.mlp.layers[0].neurons):
            neuron.w = cls.weights_layer1[i]
            neuron.b = cls.biases_layer1[i]
        
        for i, neuron in enumerate(cls.mlp.layers[1].neurons):
            neuron.w = cls.weights_layer2[i]
            neuron.b = cls.biases_layer2[i]


    @classmethod
    def tearDownClass(cls):
        cls.mlp = None
        cls.weights_layer1 = None
        cls.biases_layer1 = None
        
        cls.weights_layer2 = None
        cls.biases_layer2 = None
            
            
    def test_predict(self):
        x = [1.0, 2.0]
        output = self.mlp.predict(x)
        
        # Manually compute expected output
        layer1_out = []
        for i in range(3):
            z = sum(w * xi for w, xi in zip(self.weights_layer1[i], x)) + self.biases_layer1[i]
            a = max(0, z.data)  # ReLU
            layer1_out.append(Value(a))
        
        layer2_out = []
        for i in range(2):
            z = sum(w * a_i for w, a_i in zip(self.weights_layer2[i], layer1_out)) + self.biases_layer2[i]
            z = 1 / (1 + exp(-z.data))  # Sigmoid
            layer2_out.append(Value(z))
        for o, e in zip(output, layer2_out):
            self.assertAlmostEqual(float(o.data), float(e.data), places=5)

    def test_parameters(self):
        params = self.mlp.parameters()
        expected_num_params = (2 * 3 + 3) + (3 * 2 + 2)  # Layer1: weights + biases, Layer2: weights + biases
        self.assertEqual(len(params), expected_num_params)
    
    def test_gradients(self):
        params = self.mlp.parameters()
        grads = self.mlp.gradients()
        self.assertEqual(len(params), len(grads))
        for p, g in zip(params, grads):
            self.assertEqual(p.grad, g)
    
    def test_softmax(self):
        vals = [Value(1.0), Value(2.0), Value(3.0)]
        softmax_vals = MLPNetwork.softmax(vals)
        exp_vals = [exp(v.data) for v in vals]
        total = sum(exp_vals)
        expected = [ev / total for ev in exp_vals]
        for sv, ev in zip(softmax_vals, expected):
            self.assertAlmostEqual(float(sv.data), ev, places=5)
    
    def test_sigmoid(self):
        vals = [Value(0.0), Value(2.0), Value(-2.0)]
        sigmoid_vals = MLPNetwork.sigmoid(vals)
        expected = [1 / (1 + exp(-v.data)) for v in vals]
        for sv, ev in zip(sigmoid_vals, expected):
            self.assertAlmostEqual(float(sv.data), ev, places=5)
    
    def test_zero_grad(self):
        params = self.mlp.parameters()
        for p in params:
            p.grad = 5.0  # Set some non-zero gradient
        self.mlp.zero_grad()
        for p in params:
            self.assertEqual(p.grad, 0.0)

if __name__ == '__main__':
    unittest.main()
