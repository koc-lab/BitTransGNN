import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from evaluate import load

def prep_logits(y_pred, y_true, dataset_name):
    """
    Obtains the ground truth labels and detaches the metric computations from the gradient graph.
    """
    if dataset_name == "stsb":
        y_true = y_true.detach().cpu()
        y_pred = y_pred.detach().cpu()
    else:
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()
    return y_pred, y_true

def compute_loss(y_pred, y_true, dataset_name):
    if dataset_name == "stsb":
        return compute_mse_loss(y_pred, y_true)
    else:
        return compute_nll_loss(y_pred, y_true)

def compute_mse_loss(y_pred, y_true):
    return F.mse_loss(input=y_pred, target=y_true)

def compute_nll_loss(y_pred, y_true, reduction="mean"):
    """
    Computes the negative log-likelihood loss between the model output and the ground-truth labels.
    """
    return F.nll_loss(input=y_pred, target=y_true, reduction=reduction)

def distillation_loss(student_out, teacher_out, temperature, dataset_name, reduction="batchmean"):
    """
    Computes the distillation loss based on the student output, teacher output, and the temperature hyperparameter.
    We are applying mean reduction to ensure coherent scale with the original loss function
    Manual division is to avoid torch warning related to mean reduction
    """
    if dataset_name == "stsb":
        return compute_mse_loss(y_pred=student_out, y_true=teacher_out)
    else:
        nb_classes = student_out.size(1)
        loss = F.kl_div(input=student_out, target=teacher_out, reduction=reduction, log_target=True) / nb_classes
    return loss * (temperature ** 2)

class Metrics:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_metric_list(self):
        dataset_metric_dict = {
            "cola": ["matthews_corr", "accuracy"],
            "mrpc": ["f1", "accuracy"],
            "stsb": ["pearson_corr", "spearman_corr"],
            "rte": ["accuracy", None],
            "wnli": ["accuracy", None],
            "20ng": ["accuracy", "w_f1"],
            "mr": ["accuracy", "w_f1"],
            "R8": ["accuracy", "w_f1"],
            "R52": ["accuracy", "w_f1"],
            "ohsumed": ["accuracy", "w_f1"],
        }
        metric_list = dataset_metric_dict[self.dataset_name]
        return metric_list

    def get_metric_func(self, metric_name):
        if metric_name == "accuracy":
            metric_func = accuracy_score
        elif metric_name == "f1" or metric_name == "w_f1":
            metric_func = f1_score
        elif metric_name == "matthews_corr":
            metric_func = load("matthews_correlation")
        elif metric_name == "pearson_corr":
            metric_func = load("pearsonr")
        elif metric_name == "spearman_corr":
            metric_func = load("spearmanr")
        else:
            metric_func = None
        return metric_func

    def compute_metrics(self, y_pred, y_true, compute_all=False):
        if compute_all:
            metric_list = ["accuracy", "f1", "w_f1", "pearson_corr", "matthews_corr"]
        else:
            metric_list = self.get_metric_list()
        metric_scores = {}
        for metric_name in metric_list:
            metric_func = self.get_metric_func(metric_name)
            if metric_name == "w_f1":
                score = 100 * metric_func(y_true, y_pred, average="weighted")
            elif metric_name == "macro_f1":
                score = 100 * metric_func(y_true, y_pred, average="macro")
            elif metric_name == "micro_f1":
                score = 100 * metric_func(y_true, y_pred, average="micro")
            elif metric_name == "pearson_corr":
                score = 100 * metric_func.compute(predictions=y_pred, references=y_true)["pearsonr"]
            elif metric_name == "spearman_corr":
                score = 100 * metric_func.compute(predictions=y_pred, references=y_true)["spearmanr"]
            elif metric_name == "matthews_corr":
                score = 100 * metric_func.compute(predictions=y_pred, references=y_true)["matthews_correlation"]
            elif metric_name is None:
                continue
            else:
                score = 100 * metric_func(y_true, y_pred)
            metric_scores[metric_name] = score
        return metric_scores
