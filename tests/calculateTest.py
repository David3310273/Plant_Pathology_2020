from scripts.calculate_auc import calculate
import numpy as np
import torch
from measure import benchmark_fn
import unittest


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def testCalculate(self):
        path1 = "test.csv"
        path2 = "truth.csv"
        result = calculate(path1, path2)
        self.assertAlmostEqual(result, 0.5000)

    def testBenchmark(self):
        output = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        gt = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.float32)
        result = benchmark_fn(torch.from_numpy(output), torch.from_numpy(gt))
        self.assertEqual(result, 1)
