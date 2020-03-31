from scripts.calculate_auc import calculate
import unittest


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def testCalculate(self):
        path1 = "test.csv"
        path2 = "truth.csv"
        result = calculate(path1, path2)
        self.assertAlmostEqual(result, 0.5000)
