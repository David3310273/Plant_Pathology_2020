import configparser
import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
from dataset import KaggleDataset, KaggleLoader
from visualize import *
import numpy as np
from model import RecognizeModel
import os
from measure import benchmark_fn

"""
为了运行此脚本，你需要提供：

1. 包含training, testing, app字段的两个ini格式配置文件
2. model类型，dataset和dataloader，其中dataset必须返回读取的文件名
3. loss_fn, benchmark_fn

更换相同类型，相同目录结构的数据集需要更换output_config
更换不同类型的数据集同样需要做比较大的改动，包括：

- dataset+picker
- visualization
- 配置文件

"""

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    output_config = configparser.ConfigParser()
    output_config.read("output_config.ini")

    # training配置项
    epoch = int(config["training"]["epoch"])
    gpu = int(config["training"]["device"])
    batch_size = int(config["training"]["batch_size"])
    training_log_dir = output_config["training"]["log"]
    outlier_root = output_config["training"]["outlier_root"]
    iteration = int(config["training"]["iteration"])

    path_dict = {
        "img": output_config["training"]["img_root"],
        "label": output_config["training"]["label_root"],
    }

    # app配置项
    is_debug = config.getboolean("app", "debug")

    # testing配置项
    test_log_dir = output_config["testing"]["log"]
    is_visualize = bool(config["testing"]["vis"])

    device = torch.device("cpu:0")
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device("cuda", int(gpu))

    # TODO: 替换为真正的变量后使用
    model = RecognizeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = tensorboard.SummaryWriter(training_log_dir)
    loader_fn = None
    loss_fn = nn.BCEWithLogitsLoss()
    Dataset = KaggleDataset(path_dict)
    DataLoader = None
    iter_test_vals = []

    model.to(device)

    for iter in range(iteration):
        train_dataset = Dataset
        # train_loader = loader_fn()
        train_loader = KaggleLoader(train_dataset, batch_size)
        # test_dataset = Dataset()
        # test_loader = loader_fn()
        temp_test_vals = []

        for e in range(int(epoch)):
            model.train()
            epoch_benchmark = 0
            index = 0
            for idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data[0].to(device))
                loss = loss_fn(output, data[1].to(device))
                print("training batch {}: the loss is {}".format(idx, loss))
                temp_benchmark = benchmark_fn(output.detach().cpu(), data[1].detach().cpu())
                # 训练阶段只打印异常输出
                if is_debug and temp_benchmark < 0.5:
                    for i in range(batch_size):
                        items = {
                            "{}".format(data[-1][i]): data[0][i]     # data[-1]为读取的文件名
                        }
                        write_image(items, outlier_root)
                print("training batch {}: the benckmark is {}".format(idx, temp_benchmark))
                writer.add_scalar("train/loss/{}/{}".format(iter, e), loss, idx)
                loss.backward()
                optimizer.step()
                epoch_benchmark += temp_benchmark
                index += 1
            avg_benchmark = epoch_benchmark/index
            writer.add_scalar("train/avg_auc/{}".format(iter), avg_benchmark, e)

            # with torch.no_grad():
            #     index = 0
            #     epoch_benchmark = 0
            #     model.eval()
            #     test_writer = tensorboard.SummaryWriter(test_log_dir)
            #     for idx, data in enumerate(test_loader):
            #         output = model(data[0].to(device))
            #         loss = loss_fn(output, data[1].to(device))
            #         temp_benchmark = benchmark_fn()
            #         test_writer.add_scalar("test/loss/{}/{}".format(iter, e), loss, idx)
            #         epoch_benchmark += temp_benchmark
            #         index += 1
            #         # 如有必有，可视化test结果
            #         if is_visualize:
            #             items = {
            #                 "img_{}".format(data[-1]): data[0],  # data[-1]为读取的文件名
            #                 "output_{}".format(data[-1]): output,
            #                 "gt_{}".format(data[-1]): data[1],
            #             }
            #             write_image(items, outlier_root)
            #     avg_benchmark = epoch_benchmark / index
            #     test_writer.add_scalar("test/benchmark/{}".format(iter), avg_benchmark, e)
            #     temp_test_vals.append(avg_benchmark)
        # iter_test_vals.append(np.mean(temp_test_vals))
    print("The final test benchmark is: {}".format(np.mean(iter_test_vals)))


