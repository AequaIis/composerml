import unittest
from unittest.mock import MagicMock
import os
import sys




from composerml.training.trainer import Trainer
from composerml.training.optimizer import SGD
from composerml.training.losses import *
from composerml.models import MLPNetwork

class TestTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # random inputs
        cls.X = [
            [1.0, 2.0],
            [0.5, -1.0],
            [3.0, 0.0],
            [3.0, 4.0],
        ]
        
        # random outputs
        cls.y = [[3.0], [-0.5], [3.0], [5.0]]  
        print("setUpClass: Test data initialized")
        
        
    @classmethod
    def tearDownClass(cls):
        """Runs once after ALL tests finish."""
        print("tearDownClass: TestTrainer completed")

    def setUp(self):
        
        self.model = MLPNetwork(input_dim= 2, n_neurons= [4,2,1])
        self.optimizer = SGD(learning_rate=0.75)
        self.loss_fn = LinearLoss()
            # save originals for teardown
        self._orig_zero_grad = self.model.zero_grad
        self._orig_predict = self.model.predict
        self._orig_step = self.optimizer.step
        
        # Wrap methods to count calls
        self.model.zero_grad = MagicMock(wraps=self.model.zero_grad)
        self.model.predict = MagicMock(wraps=self.model.predict)
        self.optimizer.step = MagicMock(wraps=self.optimizer.step)

        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn
        )
        
    def tearDown(self):
        # restore original methods
        self.model.zero_grad = self._orig_zero_grad
        self.model.predict = self._orig_predict
        self.optimizer.step = self._orig_step

    def test_trainer_uses_provided_optimizer_and_loss(self):
        self.assertIs(self.trainer.optimizer, self.optimizer)
        self.assertIs(self.trainer.loss_fn, self.loss_fn)

    def test_trainer_default_optimizer_and_loss_types(self):
        trainer = Trainer(model=self.model, optimizer=None, loss_fn=None)
        self.assertIsInstance(trainer.optimizer, SGD)
        self.assertIsInstance(trainer.loss_fn, LinearLoss)
        
    def test_fit_calls_zero_grad_and_step_correct_number_of_batches(self):
        batch_size = 2
        epochs = 3
        self.trainer.fit(self.X, self.y, batch_size=batch_size, epochs= epochs)

        n_samples = len(self.X)
        expected_batches_per_epoch = (n_samples + batch_size - 1) // batch_size
        expected_step_calls = expected_batches_per_epoch * epochs

        # optimizer.step must be called once per batch
        self.assertEqual(self.optimizer.step.call_count, expected_step_calls)

        # zero_grad must be called once per batch too
        self.assertEqual(self.model.zero_grad.call_count, expected_step_calls)

    def test_fit_calls_predict_for_each_sample_each_epoch(self):
        batch_size = 2
        epochs = 3
        self.trainer.fit(self.X, self.y, batch_size=batch_size, epochs= epochs)

        n_samples = len(self.X)
        expected_predict_calls = n_samples * epochs

        # predict is called once per sample per epoch
        self.assertEqual(self.model.predict.call_count, expected_predict_calls)



if __name__ == "__main__":
    unittest.main()