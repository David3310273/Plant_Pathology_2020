from model import RecognizeModel
import unittest
import torch


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = RecognizeModel()

    def testInput(self):
        model = self.model
        x = torch.rand((1, 3, 224, 224))
        feat = model(x)
        self.assertEqual(feat.shape, torch.Size((1, 4)))