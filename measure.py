from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch


def getAUC(score, ground_truth):
    result = 0
    try:
        result = roc_auc_score(ground_truth, score)
    except:
        pass
    return result

def getAccuracy(output, ground_truth):
    """
    计算准确率性能
    :param output:
    :param ground_truth:
    :return:
    """
    f = nn.Softmax(dim=1)
    output = f(output)
    output_map = torch.argmax(output, dim=1)
    ground_truth_map = torch.argmax(ground_truth, dim=1)
    ones = torch.ones_like(output_map, dtype=torch.float32)
    zeros = torch.zeros_like(ground_truth_map, dtype=torch.float32)
    result = torch.where(output_map-ground_truth_map == 0, ones, zeros)
    return torch.mean(result).item()



def benchmark_fn(output, ground_truth):
    """
    按列分别计算与gt的AUC，求和，参照kaggle的performance定义
    :param output:
    :param ground_truth:
    :return:
    """
    f = nn.Softmax(dim=1)
    output = f(output)
    assert output.shape == ground_truth.shape
    size = list(output.size())
    benchmark = 0
    transpose_output = torch.t(output)
    transposed_gt = torch.t(ground_truth)
    for i in range(size[1]):
        temp = getAUC(transpose_output[i], transposed_gt[i])
        benchmark += temp
    return benchmark/size[1]
