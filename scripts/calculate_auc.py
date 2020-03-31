import pandas
from performance import getAUC


# 最终计算性能的脚本
def calculate(score_path, gt_path):
    score_csv = pandas.read_csv(score_path, header=None, skiprows=1, index_col=False)
    gt_csv = pandas.read_csv(gt_path, header=None, skiprows=1, index_col=False)
    total = 0
    for i in range(1, 5):
        scores = score_csv[i].to_numpy()
        gts = gt_csv[i].to_numpy()
        total += getAUC(scores, gts)
    return total/4



