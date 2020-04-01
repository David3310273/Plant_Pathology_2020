import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import pandas
import os
import numpy as np


class KaggleLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size=batch_size, shuffle=True)


class KaggleDataset(Dataset):
    def __init__(self, path_dict, need_aug=True):
        super().__init__()

        self.model_size = (224, 224)
        self.path_dict = path_dict

        assert "img" in self.path_dict
        assert "label" in self.path_dict

        self.labels = pandas.read_csv(self.path_dict["label"])
        self.images = [val for val in os.listdir(self.path_dict["img"])]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        print("accessing the image {}".format(self.images[index]))
        img_path = os.path.join(self.path_dict["img"], self.images[index])
        label = self.labels.loc[self.labels["image_id"] == self.images[index].split(".")[0]].iloc[:, 1:].to_numpy()
        image = Image.open(img_path).resize(self.model_size)
        print("the result of {} is {}".format(self.images[index], np.array2string(label[0])))
        classes = torch.from_numpy(np.array(label[0], dtype=np.uint8))
        return TF.to_tensor(image), classes, self.images[index]
