import unittest
from visualize import *
import os
import configparser
import torch


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.items = {
            "train_1": torch.Tensor([1,2,3,4]),
            "train_2": torch.Tensor([4,5,6,7]),
        }

        self.apppath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.config = configparser.ConfigParser()
        self.config.read(os.path.abspath(os.path.join(self.apppath, "output_config.ini")))

    def testCsvWriter(self):
        items = self.items
        root = self.config["testing"]["csv_output"]
        self.assertTrue(os.path.exists(root))
        write_csv(items, root)

