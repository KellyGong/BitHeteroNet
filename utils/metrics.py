import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, roc_curve


EOS = 1e-10


def fpr95_score(y_true, y_pred_logit):
    # calculate false positive rate (illicit -> licit) at 95% true negative rate(licit->licit)
    pred = 1 - y_pred_logit
    label = 1 - y_true
    fpr, tpr, thresh = roc_curve(label, pred)
    return fpr[np.where(tpr > 0.95)[0][0]]


class Eval_Metrics:
    def __init__(self, y_true, y_pred_logit):
        # y_pred = np.argmax(y_pred_logit, axis=1)
        # y_pred_logit = y_pred_logit[:, 1]

        y_pred = np.where(y_pred_logit > 0.5, 1, 0)

        self.fpr95 = fpr95_score(y_true, y_pred_logit)
        self.f1 = f1_score(y_true, y_pred, average='macro')
        self.auc = roc_auc_score(y_true, y_pred_logit)
        self.ap = average_precision_score(y_true, y_pred_logit)

    def __str__(self):
        return f'fpr95: {self.fpr95:.4f}, F1: {self.f1:.4f}, AUC: {self.auc:.4f}, AP: {self.ap:.4f}'


class Eval_Metrics_Average:
    def __init__(self):
        self.eval_metrics_list = []

    def get_metric(self, metric):
        fpr95s = [x.fpr95 for x in self.eval_metrics_list]
        f1s = [x.f1 for x in self.eval_metrics_list]
        aucs = [x.auc for x in self.eval_metrics_list]
        aps = [x.ap for x in self.eval_metrics_list]
        assert metric in ['fpr95', 'f1', 'auc', 'ap']
        if metric == 'fpr95':
            return self.__get_average(fpr95s)
        if metric == 'f1':
            return self.__get_average(f1s)
        if metric == 'auc':
            return self.__get_average(aucs)
        if metric == 'ap':
            return self.__get_average(aps)

    def __get_average(self, num_list):
        return np.mean(num_list)

    def __get_std(self, num_list):
        return np.std(num_list)

    def __call__(self, eval_metrics):
        self.eval_metrics_list.append(eval_metrics)
    
    def __str__(self):
        fpr95s = [x.fpr95 for x in self.eval_metrics_list]
        f1s = [x.f1 for x in self.eval_metrics_list]
        aucs = [x.auc for x in self.eval_metrics_list]
        aps = [x.ap for x in self.eval_metrics_list]
        
        return f'Average of {len(self.eval_metrics_list)} runs:' \
               f'FPR95s: {self.__get_average(fpr95s):.4f} ± {self.__get_std(fpr95s):.4f}, ' \
               f'F1: {self.__get_average(f1s):.4f} ± {self.__get_std(f1s):.4f}, ' \
               f'AUC: {self.__get_average(aucs):.4f} ± {self.__get_std(aucs):.4f}, ' \
               f'AP: {self.__get_average(aps):.4f} ± {self.__get_std(aps):.4f}'

