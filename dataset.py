import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
import pandas
import os
import numpy as np


class KaggleLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size=batch_size, shuffle=True)


class KaggleDataset(Dataset):
    def __init__(self, path_dict, paths=None, need_aug=True):
        super().__init__()

        self.model_size = (224, 224)
        self.path_dict = path_dict
        self.need_aug = need_aug

        assert "img" in self.path_dict
        assert "label" in self.path_dict

        self.labels = pandas.read_csv(self.path_dict["label"])
        self.images = paths

    def __len__(self):
        return len(self.images)

    def aug_transform(self, image):
        composer = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.01, hue=0.01),
            transforms.RandomRotation(degrees=90),
        ])
        pic = composer(image)
        return pic

    def __getitem__(self, index):
        img_path = os.path.join(self.path_dict["img"], self.images[index])
        label = self.labels.loc[self.labels["image_id"] == self.images[index].split(".")[0]].iloc[:, 1:].to_numpy()
        image = Image.open(img_path).resize(self.model_size)
        if self.need_aug:
            image = self.aug_transform(image)
        classes = torch.from_numpy(np.array(label[0], dtype=np.float32))
        return TF.to_tensor(image), classes, self.images[index]


# 加载验证集
class ValidationDataset(Dataset):
    def __init__(self, path_dict, need_aug=True):
        super().__init__()

        self.model_size = (224, 224)
        self.path_dict = path_dict
        self.need_aug = need_aug

        assert "img" in self.path_dict
        self.images = sorted(os.listdir(self.path_dict["img"]))

    def __len__(self):
        return len(self.images)

    def aug_transform(self, image):
        composer = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.01, hue=0.01),
            transforms.RandomRotation(degrees=90),
        ])
        pic = composer(image)
        return pic

    def __getitem__(self, index):
        img_path = os.path.join(self.path_dict["img"], self.images[index])
        image = Image.open(img_path).resize(self.model_size)
        if self.need_aug:
            image = self.aug_transform(image)
        return TF.to_tensor(image), self.images[index]
