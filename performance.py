from sklearn.metrics import roc_auc_score


def getAUC(score, ground_truth):
    return roc_auc_score(ground_truth, score)