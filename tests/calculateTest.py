from scripts.calculate_auc import calculate
import numpy as np
import torch
from measure import benchmark_fn, getAccuracy
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

    def testAccuracy(self):
        output = torch.Tensor([
            [0.999992847442627,3.99259215555503e-06,2.7557673547562445e-06,4.543750264929258e-07],
            [2.3089718524715863e-05,0.0024196573067456484,3.412924343138002e-05,0.9975231289863586],
            [0.00500213960185647,0.006962891202419996,0.0008244177442975342,0.9872105121612549],
            [0.0017724998760968447,0.0026088396552950144,0.00018246815307065845,0.9954361319541931],
        ])

        gt = torch.Tensor([
            [1.0,0.0,0.0,0.0],
            [0.0,1.0,0.0,0.0],
            [0.0,1.0,0.0,0.0],
            [0.0,0.0,0.0,1.0],
        ])

        self.assertAlmostEqual(getAccuracy(output, gt), 0.5)
