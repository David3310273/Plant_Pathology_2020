from dataset import KaggleDataset, KaggleLoader
from datapicker import DataPicker
import unittest
import torch
import configparser
import os
import math


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.apppath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.config = configparser.ConfigParser()
        self.config.read(os.path.abspath(os.path.join(self.apppath, "output_config.ini")))
        self.path_dict = {
            "img": self.config["training"]["img_root"],
            "label": self.config["training"]["label_root"],
        }

    def testDataset(self):
        picker = DataPicker(self.path_dict["img"], self.path_dict["label"], 0.8)
        train_loader, test_loader = picker.get_loader(batch_size=1)
        visited = set()
        count = 0
        for index, data in enumerate(train_loader):
            image, label, name = data
            self.assertTrue(name[0] not in visited)
            visited.add(name[0])
            self.assertEqual(image.shape, torch.Size((1, 3, 224, 224)))
            self.assertEqual(label.shape, torch.Size((1, 4)))
            count += 1
        self.assertEqual(count, math.ceil(1821*0.8))
        for index, data in enumerate(test_loader):
            image, label, name = data
            self.assertTrue(name[0] not in visited)
            self.assertEqual(image.shape, torch.Size((1, 3, 224, 224)))
            self.assertEqual(label.shape, torch.Size((1, 4)))
            count += 1
        self.assertEqual(count, 1821)
