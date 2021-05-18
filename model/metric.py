import torch
from sklearn import metrics

""""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import auc 


y_true = [1, 1, 1, 1, 1,    0, 0, 0, 0, 0]
y_pred = [0, 0, 1, 1, 1,    1, 0, 0, 0, 0]

TP，3
FP，1
TN，4
FN，2

Acc，(3+4)/10=0.7
P，3/(3+1)=0.75
R，3/(3+2)=0.6
F1，(2×0.75×0.6)/(0.75+0.6)=0.667

Sensitivity，3/(3+2)=0.6
Specificity，4/(1+4)=0.8
TPR，0.6
FPR，0.2

准确率，
Accuracy = (TP + TN) / (TP + TN + FP + FN)
精确率（差准率），
Precision = TP / (TP + FP)
召回率（查全率），
Recall = TP / (TP + FN)
F1值，
F1 = (2×P×R) /（P+R）

灵敏度（召回率），
Sensitivity = TP / (TP + FN)
特异度（由于我们比较关心正样本，所以需要查看有多少负样本被错误地预测为正样本，所以使用（1-特异度），而不是特异度），
Specificity = TN / (FP + TN)
真正率，
TPR = Sensitivity = TP / (TP + FN)
假正率，
FPR = 1-Specificity = FP / (FP + TN)

y_true = [1, 1, 1, 1, 1,    0, 0, 0, 0, 0]
y_pred = [0, 0, 1, 1, 1,    1, 0, 0, 0, 0]

print(accuracy_score(y_true, y_pred))
print(precision_score(y_true, y_pred))
print(recall_score(y_true, y_pred))
print(f1_score(y_true, y_pred))

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
print(metrics.auc(fpr, tpr))
"""

"""
直接实现binary_accuracy，

# preds = torch.tensor([1., 1., 1., 0., 0.])
# y = torch.tensor([1., 1., 1., 1., 1.])
# preds做sigmoid，再round
rounded_preds = torch.round(torch.sigmoid(preds))
# tensor([1., 1., 1., 0., 0.])
# 判断true/false，转float，
correct = (rounded_preds == y).float()
# tensor([1., 1., 1., 0., 0.])
# 正确/所有，
acc = correct.sum() / len(correct)
# 3/5=0.6
"""


def binary_accuracy(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.round(torch.sigmoid(y_pred))
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
    return metrics.accuracy_score(y_true, y_pred)


def binary_precision(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.round(torch.sigmoid(y_pred))
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
    return metrics.precision_score(y_true, y_pred)


def binary_recall(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.round(torch.sigmoid(y_pred))
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
    return metrics.recall_score(y_true, y_pred)


def binary_f1(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.round(torch.sigmoid(y_pred))
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
    return metrics.f1_score(y_true, y_pred)


def binary_auc(y_pred, y_true):
    with torch.no_grad():
        y_pred = torch.round(torch.sigmoid(y_pred))
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def binary_accuracy_threshold(y_pred, y_true, threshold):
    with torch.no_grad():
        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_ = []
        for i, val in enumerate(y_pred):
            pred_ = 1 if val >= threshold else 0
            y_pred_.append(pred_)
    return metrics.accuracy_score(y_true, y_pred_)


def binary_precision_threshold(y_pred, y_true, threshold):
    with torch.no_grad():
        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_ = []
        for i, val in enumerate(y_pred):
            pred_ = 1 if val >= threshold else 0
            y_pred_.append(pred_)
    return metrics.precision_score(y_true, y_pred_)


def binary_recall_threshold(y_pred, y_true, threshold):
    with torch.no_grad():
        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_ = []
        for i, val in enumerate(y_pred):
            pred_ = 1 if val >= threshold else 0
            y_pred_.append(pred_)
    return metrics.recall_score(y_true, y_pred_)


def binary_f1_threshold(y_pred, y_true, threshold):
    with torch.no_grad():
        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_ = []
        for i, val in enumerate(y_pred):
            pred_ = 1 if val >= threshold else 0
            y_pred_.append(pred_)
    return metrics.f1_score(y_true, y_pred_)


def binary_auc_threshold(y_pred, y_true, threshold):
    """
    计算二分类auc，可调阈值
    """
    with torch.no_grad():
        y_pred = torch.sigmoid(y_pred).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        y_pred_ = []
        for i, val in enumerate(y_pred):
            pred_ = 1 if val >= threshold else 0
            y_pred_.append(pred_)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    return auc
