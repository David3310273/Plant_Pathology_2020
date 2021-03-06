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
from measure import benchmark_fn, getAccuracy

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
    pos_weights = torch.FloatTensor([91/516, 1, 91/622, 91/622])

    training_log_dir = output_config["training"]["log"]
    outlier_root = output_config["training"]["outlier_root"]
    csv_output = output_config["testing"]["csv_output"]
    iteration = int(config["training"]["iteration"])

    if not os.path.exists(outlier_root):
        os.mkdir(outlier_root)

    path_dict = {
        "img": output_config["training"]["img_root"],
        "label": output_config["training"]["label_root"],
    }

    # app配置项
    is_debug = config.getboolean("app", "debug")
    model_output = output_config.get("app", "model_output")

    # testing配置项
    test_log_dir = output_config["testing"]["log"]
    is_visualize = bool(config["testing"]["vis"])

    device = torch.device("cpu:0")
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device("cuda", int(gpu))
        pos_weights = pos_weights.to(device)

    if not os.path.exists(csv_output):
        os.mkdir(csv_output)

    # TODO: 替换为真正的变量后使用
    model = RecognizeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = tensorboard.SummaryWriter(training_log_dir)
    picker = DataPicker(path_dict["img"], path_dict["label"], k_fold)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    Dataset = KaggleDataset(path_dict)
    DataLoader = None
    iter_test_vals = []
    iter_test_accuracy = []

    model.to(device)

    if not os.path.exists(model_output):
        os.mkdir(model_output)

    for iter in range(iteration):
        train_loader, test_loader = picker.get_loader(batch_size=10)
        temp_test_vals = []
        temp_test_accuracy = []
        start = 0

        if os.path.exists(os.path.join(model_output, str(iter))):
            path = os.path.join(model_output, str(iter))
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']

        for e in range(start, int(epoch)):
            model.train()
            epoch_benchmark = 0
            epoch_accuracy = 0
            index = 0
            for idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                batch = list(data[0].shape)[0]
                output = model(data[0].to(device))
                loss = loss_fn(output, data[1].to(device))
                print("training batch {}: the loss is {}".format(idx, loss))
                temp_benchmark = benchmark_fn(output.detach().cpu(), data[1].detach().cpu())
                temp_accuracy = getAccuracy(output.detach().cpu(), data[1].detach().cpu())
                # 训练阶段只打印异常输出
                if is_debug and temp_accuracy < 0.5:
                    for i in range(batch):
                        items = {
                            "{}".format(data[-1][i]): data[0][i]     # data[-1]为读取的文件名
                        }
                        write_image(items, os.path.join(outlier_root, str(iter)))
                print("training batch {}: the auc is {}".format(idx, temp_benchmark))
                print("training batch {}: the accuracy is {}".format(idx, temp_accuracy))
                writer.add_scalar("train/loss/{}/{}".format(iter, e), loss, idx)
                writer.add_scalar("train/auc/{}/{}".format(iter, e), temp_benchmark, idx)
                writer.add_scalar("train/accuracy/{}/{}".format(iter, e), temp_accuracy, idx)
                loss.backward()
                optimizer.step()
                epoch_benchmark += temp_benchmark
                epoch_accuracy += temp_accuracy
                index += 1
            avg_benchmark = epoch_benchmark/index
            avg_accuracy = epoch_accuracy/index
            writer.add_scalar("train/avg_auc/{}".format(iter), avg_benchmark, e)
            writer.add_scalar("train/avg_accuracy/{}".format(iter), avg_accuracy, e)

            # 测试代码
            with torch.no_grad():
                index = 0
                epoch_benchmark = 0
                epoch_accuracy = 0
                model.eval()
                test_writer = tensorboard.SummaryWriter(test_log_dir)
                items = {}
                gts = {}
                f = nn.Sigmoid()
                for idx, data in enumerate(test_loader):
                    output = model(data[0].to(device))
                    batch = list(data[0].shape)[0]
                    loss = loss_fn(output, data[1].to(device))
                    temp_benchmark = benchmark_fn(output.detach().cpu(), data[1].detach().cpu())
                    temp_accuracy = getAccuracy(output.detach().cpu(), data[1].detach().cpu())
                    print("test batch {}: the loss is {}".format(idx, loss))
                    print("test batch {}: the auc is {}".format(idx, temp_benchmark))
                    print("test batch {}: the accuracy is {}".format(idx, temp_accuracy))
                    test_writer.add_scalar("test/loss/{}/{}".format(iter, e), loss, idx)
                    test_writer.add_scalar("test/auc/{}/{}".format(iter, e), temp_benchmark, idx)
                    test_writer.add_scalar("test/accuracy/{}/{}".format(iter, e), temp_accuracy, idx)
                    epoch_benchmark += temp_benchmark
                    epoch_accuracy += temp_accuracy
                    index += 1
                    visual_output = f(output)
                    for i in range(batch):
                        items["{}".format(data[-1][i].split(".")[0])] = visual_output[i]  # data[-1]为读取的文件名
                        gts["{}".format(data[-1][i].split(".")[0])] = data[1][i]
                # 按照规定标准计算测试性能
                write_csv(items, os.path.join(csv_output, str(iter)), "test_result_{}.csv".format(e))
                write_csv(gts, os.path.join(csv_output, str(iter)), "ground_truths_{}.csv".format(e))
                test_avg_benchmark = calculate(os.path.join(os.path.join(csv_output, str(iter)), "test_result_{}.csv".format(e)), os.path.join(os.path.join(csv_output, str(iter)), "ground_truths_{}.csv".format(e)))
                test_avg_accuracy = epoch_accuracy/index
                test_writer.add_scalar("test/avg_benchmark/{}".format(iter), test_avg_benchmark, e)
                test_writer.add_scalar("test/avg_accuracy/{}".format(iter), test_avg_accuracy, e)
                temp_test_vals.append(test_avg_benchmark)
                temp_test_accuracy.append(test_avg_accuracy)
            torch.save({
                "epoch": e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(model_output, str(iter)))
        iter_test_vals.append(np.mean(temp_test_vals))
        iter_test_accuracy.append(np.mean(test_avg_accuracy))
    print("The final test benchmark is: {}".format(np.mean(iter_test_vals)))
    print("The final test accuracy is: {}".format(np.mean(iter_test_accuracy)))


