from dataset import KaggleDataset, KaggleLoader
import unittest
import torch
import configparser
import os


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.apppath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.config = configparser.ConfigParser()
        self.config.read(os.path.abspath(os.path.join(self.apppath, "config.ini")))
        self.path_dict = {
            "img": self.config["training"]["img_root"],
            "label": self.config["training"]["label_root"],
        }

    def testDataset(self):
        dataset = KaggleDataset(self.path_dict, False)
        dataloader = KaggleLoader(dataset, 10)
        for index, data in enumerate(dataloader):
            image, label, name = data
            print(name)
            self.assertEqual(image.shape, torch.Size((10, 3, 224, 224)))
            self.assertEqual(label.shape, torch.Size((10, 4)))