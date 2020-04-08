import configparser
import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
from dataset import KaggleDataset, KaggleLoader
from datapicker import DataPicker
from visualize import *
from scripts.calculate_auc import calculate
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
    gpu = config["training"]["device"]
    batch_size = int(config["training"]["batch_size"])
    k_fold = float(config["training"]["k"])

    training_log_dir = output_config["training"]["log"]
    outlier_root = output_config["training"]["outlier_root"]
    csv_output = output_config["testing"]["csv_output"]
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

    if not os.path.exists(csv_output):
        os.mkdir(csv_output)

    # TODO: 替换为真正的变量后使用
    model = RecognizeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = tensorboard.SummaryWriter(training_log_dir)
    picker = DataPicker(path_dict["img"], path_dict["label"], k_fold)
    loss_fn = nn.BCEWithLogitsLoss()
    Dataset = KaggleDataset(path_dict)
    DataLoader = None
    iter_test_vals = []

    model.to(device)

    for iter in range(iteration):
        train_loader, test_loader = picker.get_loader(batch_size=10)
        temp_test_vals = []

        for e in range(int(epoch)):
            model.train()
            epoch_benchmark = 0
            index = 0
            for idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                batch = list(data[0].shape)[0]
                output = model(data[0].to(device))
                loss = loss_fn(output, data[1].to(device))
                print("training batch {}: the loss is {}".format(idx, loss))
                temp_benchmark = benchmark_fn(output.detach().cpu(), data[1].detach().cpu())
                # 训练阶段只打印异常输出
                if is_debug and temp_benchmark < 0.5:
                    for i in range(batch):
                        items = {
                            "{}".format(data[-1][i]): data[0][i]     # data[-1]为读取的文件名
                        }
                        write_image(items, outlier_root)
                print("training batch {}: the benckmark is {}".format(idx, temp_benchmark))
                writer.add_scalar("train/loss/{}/{}".format(iter, e), loss, idx)
                writer.add_scalar("train/benchmark/{}/{}".format(iter, e), temp_benchmark, idx)
                loss.backward()
                optimizer.step()
                epoch_benchmark += temp_benchmark
                index += 1
            avg_benchmark = epoch_benchmark/index
            writer.add_scalar("train/avg_auc/{}".format(iter), avg_benchmark, e)

            # 测试代码
            with torch.no_grad():
                index = 0
                epoch_benchmark = 0
                model.eval()
                test_writer = tensorboard.SummaryWriter(test_log_dir)
                items = {}
                gts = {}
                f = nn.Softmax(dim=1)
                for idx, data in enumerate(test_loader):
                    output = model(data[0].to(device))
                    batch = list(data[0].shape)[0]
                    loss = loss_fn(output, data[1].to(device))
                    temp_benchmark = benchmark_fn(output.detach().cpu(), data[1].detach().cpu())
                    print("test batch {}: the loss is {}".format(idx, loss))
                    print("test batch {}: the benchmark is {}".format(idx, temp_benchmark))
                    test_writer.add_scalar("test/loss/{}/{}".format(iter, e), loss, idx)
                    test_writer.add_scalar("test/benchmark/{}/{}".format(iter, e), temp_benchmark, idx)
                    epoch_benchmark += temp_benchmark
                    index += 1
                    visual_output = f(output)
                    for i in range(batch):
                        items["{}".format(data[-1][i].split(".")[0])] = visual_output[i]  # data[-1]为读取的文件名
                        gts["{}".format(data[-1][i].split(".")[0])] = data[1][i]
                # 按照规定标准计算测试性能
                write_csv(items, os.path.join(csv_output, str(iter)), "test_result_{}.csv".format(e))
                write_csv(gts, os.path.join(csv_output, str(iter)), "ground_truths_{}.csv".format(e))
                test_avg_benchmark = calculate(os.path.join(os.path.join(csv_output, str(iter)), "test_result_{}.csv".format(e)), os.path.join(os.path.join(csv_output, str(iter)), "ground_truths_{}.csv".format(e)))
                test_writer.add_scalar("test/avg_benchmark/{}".format(iter), test_avg_benchmark, e)
                temp_test_vals.append(test_avg_benchmark)
        iter_test_vals.append(np.mean(temp_test_vals))
    print("The final test benchmark is: {}".format(np.mean(iter_test_vals)))


