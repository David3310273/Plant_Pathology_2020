import configparser
import torch
from dataset import ValidationDataset, KaggleLoader
from model import RecognizeModel
import os
import torch.nn as nn
from visualize import *

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")

    output_config = configparser.ConfigParser()
    output_config.read("output_config.ini")

    model = RecognizeModel()
    gpu = config["training"]["device"]

    # 最终输出的路径
    csv_output = output_config["validate"]["csv_output"]
    if not os.path.exists(csv_output):
        os.mkdir(csv_output)

    device = torch.device("cpu:0")
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        device = torch.device("cuda", int(gpu))

    # 加载验证集数据
    source_path = output_config["validate"]["img_root"]
    dataset = ValidationDataset({
        "img": source_path,
    })
    dataloader = KaggleLoader(dataset, batch_size=1)

    # 加载模型
    model_output = output_config["app"]["model_output"]
    if os.path.exists(model_output):
        i = max(os.listdir(model_output))
        path = os.path.join(model_output, str(i))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError("Cannot validate dataset because no model found!")

    model.to(device)

    with torch.no_grad():
        model.eval()
        items = {}
        f = nn.Softmax(dim=1)
        for index, data in enumerate(dataloader):
            output = model(data[0].to(device))
            result = f(output)
            items[data[-1][0].split(".")[0]] = result[0]
        write_csv(items, csv_output, "result.csv")
