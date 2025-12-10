import unittest
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, root_dir)


from Models.Value import Value
from math import tanh,exp,log

class TestValue(unittest.TestCase):
    
    def test_initialization(self):
        v1 = Value(10, label ='test')
        self.assertEqual(v1.data,10)
        self.assertEqual(v1.grad, 0)
        self.assertEqual(v1._prev, set())
        self.assertEqual(v1._op, '')
        self.assertEqual(v1.label, 'test')
    
    def setUp(self):
        self.v1 = Value(10)
        self.v2 = Value(5)
        self.num1 = 10
        self.num2 = 5
    
    def test_print(self):
        self.assertEqual(repr(self.v1), "Value: 10, Grad: 0")
   
        
    def test_num_conversion(self):
        num1_as_value = Value._as_value(self.num1)
        self.assertIsInstance(num1_as_value, Value)
        self.assertEqual(num1_as_value.data, self.num1)
    
    def test_addition(self):
        result = self.v1 + self.v2
        self.assertEqual(result.data, 15)
        self.assertEqual(result._op, '+')
        self.assertIn(self.v1, result._prev)
        self.assertIn(self.v2, result._prev)
        self.assertEqual(result.grad, 0)
        
        result2 = self.v1 + self.num2
        self.assertEqual(result2.data, 15)
        

    def test_backward_addition(self):
        result = self.v1 + self.v2
        result.grad = 1
        result._backward()
        self.assertEqual(self.v1.grad, 1)
        self.assertEqual(self.v2.grad, 1)
    
    def test_raddition(self):
        result = self.num1 + self.v2
        self.assertEqual(result.data, 15)
        self.assertEqual(result._op, '+')
        self.assertIn(self.v2, result._prev)
    
    def test_negation(self):
        result = -self.v1
        self.assertEqual(result.data, -10)
        self.assertEqual(result._op, 'neg')
        self.assertIn(self.v1, result._prev)
        self.assertEqual(result.grad, 0)
    
    def test_backward_negation(self):
        result = -self.v1
        result.grad = 1
        result._backward()
        self.assertEqual(self.v1.grad, -1)
    
    def test_subtraction(self):
        result = self.v1 - self.v2
        self.assertEqual(result.data, 5)
        self.assertEqual(result._op, '+')  # because subtraction uses addition of negation
        self.assertIn(self.v1, result._prev)
        
        result2 = self.v1 - self.num2
        self.assertEqual(result2.data, 5)       

    def test_rsubtraction(self):
        result = self.num1 - self.v2
        self.assertEqual(result.data, 5)
        self.assertEqual(result._op, '+')  # because subtraction uses addition of negation
  
        
    def test_subtraction_backward(self):
        result = self.v1 - self.v2
        result.grad = 1
        result.backward()
        self.assertEqual(self.v1.grad, 1)
        self.assertEqual(self.v2.grad, -1)
    
    def test_multiplication(self):
        result = self.v1 * self.v2
        self.assertEqual(result.data, 50)
        self.assertEqual(result._op, '*')
        self.assertIn(self.v1, result._prev)
        self.assertIn(self.v2, result._prev)
        self.assertEqual(result.grad, 0)
        
        result2 = self.v1 * self.num2
        self.assertEqual(result2.data, 50)
    
    def test_rmultiplication(self):
        result = self.num1 * self.v2
        self.assertEqual(result.data, 50)
        self.assertEqual(result._op, '*')
        self.assertIn(self.v2, result._prev)
        
    def test_backward_multiplication(self):
        result = self.v1 * self.v2
        result.grad = 1
        result._backward()
        self.assertEqual(self.v1.grad, 5)
        self.assertEqual(self.v2.grad, 10)
    
    def test_power(self):
        result = self.v1 ** 3
        self.assertEqual(result.data, 1000)
        self.assertEqual(result._op, '**')
        self.assertIn(self.v1, result._prev)
        self.assertEqual(result.grad, 0)
    
    def test_backward_power(self):
        result = self.v1 ** 3
        result.grad = 1
        result._backward()
        self.assertEqual(self.v1.grad, 3 * (10 ** 2))
    
    def test_truedivision(self):
        result = self.v1 / self.v2
        self.assertEqual(result.data, 2)
        self.assertEqual(result._op, '*')
        self.assertIn(self.v1, result._prev)
        
        
        result2 = self.v1 / self.num2
        self.assertEqual(result2.data, 2)
    
    def test_rtruedivision(self):
        result = self.num1 / self.v2
        self.assertEqual(result.data, 2)
        self.assertEqual(result._op, '*')

        
    
    def test_backward_truedivision(self):
        result = self.v1 / self.v2
        result.grad = 1
        result.backward()
        self.assertEqual(self.v1.grad, 1 / 5)
        self.assertEqual(self.v2.grad, -10 / (5 ** 2))
    
    def test_tanh(self):
        result = self.v1.tanh()
        self.assertAlmostEqual(result.data, tanh(10))
        self.assertEqual(result._op, 'tanh')
        self.assertIn(self.v1, result._prev)
    
    def test_backward_tanh(self):
        result = self.v1.tanh()
        result.grad = 1
        result._backward()
        self.assertAlmostEqual(self.v1.grad, (1 - result.data**2))
    
    def test_exp(self):
        result = self.v1.exp()
        self.assertAlmostEqual(result.data, exp(10))
        self.assertEqual(result._op, 'exp')
        self.assertIn(self.v1, result._prev)
    
    def test_backward_exp(self):
        result = self.v1.exp()
        result.grad = 1
        result._backward()
        self.assertAlmostEqual(self.v1.grad, result.data)
    
    def test_log(self):
        result = self.v1.log()
        self.assertAlmostEqual(result.data, log(10))
        self.assertEqual(result._op, 'log')
        self.assertIn(self.v1, result._prev)
    
    def test_backward_log(self):
        result = self.v1.log()
        result.grad = 1
        result._backward()
        self.assertAlmostEqual(self.v1.grad, 1 / 10)
    
    def test_relu(self):
        result = self.v1.relu()
        self.assertEqual(result.data, 10)
        self.assertEqual(result._op, 'ReLU')
        self.assertIn(self.v1, result._prev)
    
    def test_backward_relu(self):
        result = self.v1.relu()
        result.grad = 1
        result._backward()
        self.assertEqual(self.v1.grad, 1)
        
        v_neg = Value(-5)
        result_neg = v_neg.relu()
        result_neg.grad = 1
        result_neg._backward()
        self.assertEqual(v_neg.grad, 0)
    
    def test_backward_chain(self):
        # Test a chain of operations: z = ((v1 + v2) * v2).tanh()
        z = (self.v1 + self.v2) * self.v2
        z = z.tanh()
        z.grad = 1
        z._backward()
        
        # Manually compute gradients
        dz_dtanh = 1 - z.data**2
        dz_dsum = dz_dtanh * self.v2.data
        dz_dv2_from_mul = dz_dtanh * (self.v1.data + self.v2.data)
        d1_sum_dv1 = 1
        d1_sum_dv2 = 1
        
        expected_grad_v1 = dz_dsum * d1_sum_dv1
        expected_grad_v2 = dz_dv2_from_mul + dz_dsum * d1_sum_dv2
        
        self.assertAlmostEqual(self.v1.grad, expected_grad_v1)
        self.assertAlmostEqual(self.v2.grad, expected_grad_v2)

if __name__ == '__main__':
    unittest.main()