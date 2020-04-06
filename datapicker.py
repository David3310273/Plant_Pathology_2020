import os
import math
import random
import torch

from dataset import KaggleDataset


class DataPicker:
    """
    经典目录结构读取，返回用于交叉验证的train_loader和test_loader，可直接套用，目录样例如下

    - input_root
    - - img1.png
    ...
    - - imgn.png

    - all ground_truth_root
    - - gt1.png
    ...
    - - gtn.png
    """
    def __init__(self, input_root, ground_truth_root, k_fold=1):
        """
        初始化loader
        :param input_root: 原始数据输入绝对路径，为字符串
        :param ground_truth_root: ground truth路径，为{关键字 => 字符串}结构
        :param k_fold: k折交叉验证，实际训练的数据集大小=k_fold*原始数据集大小
        """
        super().__init__()
        self.input_root = input_root
        self.ground_root = ground_truth_root
        self.k_fold = k_fold

        assert type(self.input_root) == str
        assert type(self.ground_root) == str
        assert 0 < k_fold <= 1

    @staticmethod
    def joint_shuffle(lists):
        """
        一致shffule多个数组
        :param lists: 多个list共同组成的数组，shape为n*sizeof_list。
        :return:
        """
        randnum = random.randint(0, 10)

        for i in range(len(lists)):
            random.seed(randnum)
            random.shuffle(lists[i])

    def _make_loader(self, batch_size=1):
        """
        读入数据集的根目录，摘取出训练集和测试集对应的文件路径
        :return:
        """
        input_root = self.input_root
        ground_root = self.ground_root

        path_dict = {
            "img": input_root,
            "label": ground_root,
        }

        input_data_paths = [path for path in os.listdir(input_root)]
        training_set = [input_data_paths]
        training_len = math.ceil(self.k_fold*len(input_data_paths))
        self.joint_shuffle(training_set)

        train_img_paths = training_set[0][:training_len]
        test_img_paths = training_set[0][training_len:]

        train_dataset = KaggleDataset(path_dict, train_img_paths, ground_root)
        test_dataset = KaggleDataset(path_dict, test_img_paths, ground_root)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader

    def get_loader(self, batch_size=1):
        return self._make_loader(batch_size)




