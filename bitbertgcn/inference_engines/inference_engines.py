import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from data.loader.dataloaders import GraphDataObject, TextDataObject
from metrics import Metrics, compute_loss, prep_logits

class BitBERTInference:
    def __init__(self, model, dataset_name, text_data: TextDataObject, device):
        self.model = model
        self.dataset_name = dataset_name
        self.text_data = text_data
        self.device = device
        self.metrics = Metrics(dataset_name)

    def eval_step(self, batch):
        (input_ids, attention_mask, label) = [x.to(self.device) for x in batch]
        y_pred = self.model(input_ids, attention_mask)
        y_true = label
        loss = compute_loss(y_true=y_true, y_pred=y_pred, dataset_name=self.dataset_name)
        return loss, y_pred, y_true
    
    def eval_epoch(self, data_split_name="val"):
        running_loss = 0
        total_samples = 0
        self.model.eval()
        all_preds = []
        all_labels = []
        print(data_split_name)
        with torch.no_grad():
            for i, batch in enumerate(self.text_data.loaders[data_split_name]):
                loss, y_pred, y_true = self.eval_step(batch)
                batch_size = len(y_pred)
                total_samples += batch_size
                running_loss += loss.item() * batch_size
                y_pred, y_true = prep_logits(y_pred, y_true, self.dataset_name)
                all_preds.append(y_pred)
                all_labels.append(y_true)
        final_loss = running_loss / total_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores
    
    def log_results(self, epoch, metric_scores, loss, 
                    report_time=False, time_mean=None,
                    split="train"):
        metrics = {"epochs": epoch+1, f"{split}_loss": loss}
        report_str = f"{split}_loss: {loss:.8f}"
        for metric_name in metric_scores.keys():
            metrics[f"{split}_{metric_name}"] = metric_scores[metric_name]
            report_str += f", {split}_{metric_name}: {metric_scores[metric_name]:.8f}"
        if report_time:
            metrics[f"{split}_time_mean"] = time_mean
        print(report_str)
        return metrics

    def run(self, report_time: bool = False):
        train_time_mean, test_time_mean, val_time_mean = None, None, None
        epoch = 0
        train_t0 = time.time()
        train_loss, train_metric_scores = self.eval_epoch(data_split_name="train")
        train_t1 = time.time()
        if report_time:
            print(f"Inference duration with training set: {train_t1 - train_t0} seconds")
            train_time_mean = train_t1 - train_t0
        test_t0 = time.time()
        test_loss, test_metric_scores = self.eval_epoch(data_split_name="test")
        test_t1 = time.time()
        if report_time:
            print(f"Inference duration with test set: {test_t1 - test_t0} seconds")
            test_time_mean = test_t1 - test_t0
        val_t0 = time.time()
        val_loss, val_metric_scores = self.eval_epoch(data_split_name="val")
        val_t1 = time.time()
        if report_time:
            print(f"Inference duration with validation set: {val_t1 - val_t0} seconds")
            val_time_mean = val_t1 - val_t0
        train_metrics = self.log_results(epoch, train_metric_scores, train_loss, report_time, train_time_mean, split="train")
        val_metrics = self.log_results(epoch, val_metric_scores, val_loss, report_time, val_time_mean, split="val")
        test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")
        
        primary_metric = self.metrics.get_metric_list()[0]
        secondary_metric = self.metrics.get_metric_list()[1] if self.metrics.get_metric_list()[1] else None
        val_track_metric = val_metric_scores[primary_metric]
        test_track_metric = test_metric_scores[primary_metric]
        inference_metrics = {f"best_val_{primary_metric}": val_track_metric, f"best_test_{primary_metric}": test_track_metric}
        if secondary_metric:
            test_secondary_metric = test_metric_scores[secondary_metric]
            inference_metrics[f"best_test_{secondary_metric}"] = test_secondary_metric

        return inference_metrics
    
class BitBERTGCNInference:
    def __init__(self, model, dataset_name, graph_data: GraphDataObject, joint_training, device, batch_size, inductive=False):
        self.model = model
        self.dataset_name = dataset_name
        self.graph_data = graph_data
        self.joint_training = joint_training
        self.device = device
        self.batch_size = batch_size
        self.inductive = inductive
        self.metrics = Metrics(dataset_name)

    def update_cls_feats(self):
        """
        initializes cls_feats by using bert model in inference mode
        generally not mandatory to be used in case bert model and gcn will be jointly trained
        if pretrained bert model will only be used to initialize gcn features and bert model will not be trained,
        then it is necessary to call this function
        """        
        # no gradient needed, since we are only using bert model to get cls features.
        doc_mask = self.graph_data.doc_mask.to(self.device).type(torch.BoolTensor)
        input_ids_full = self.graph_data.input_ids
        attention_mask_full = self.graph_data.attention_mask
        dataloader = DataLoader(
            TensorDataset(input_ids_full.to(self.device)[doc_mask], attention_mask_full.to(self.device)[doc_mask]),
            batch_size=self.batch_size
        )
        with torch.no_grad():
            self.model.eval()
            cls_list = []
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(self.device) for x in batch]
                output = self.model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
                cls_list.append(output.cpu())
            cls_feat = torch.cat(cls_list, axis=0)
        self.graph_data.cls_feats[doc_mask] = cls_feat
        self.graph_data = self.graph_data.convert_device(device="cpu")
    
    def eval_step(self, batch):
        idx = batch[0].to(self.device)
        y_pred = self.model(self.graph_data.convert_device(self.device), idx)
        y_true = self.graph_data.label.to(self.device)[idx]
        y_pred = torch.log(y_pred)
        loss = compute_loss(y_true=y_true, y_pred=y_pred, dataset_name=self.dataset_name)
        return loss, y_pred, y_true

    def eval_epoch(self, data_split_name="val"):
        running_loss = 0
        all_preds = []
        all_labels = []
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.graph_data.idx_loaders[data_split_name]):
                loss, y_pred, y_true = self.eval_step(batch)
                batch_size = len(y_pred)
                total_samples += batch_size
                running_loss += loss.item() * batch_size
                y_pred, y_true = prep_logits(y_pred, y_true)
                all_preds.append(y_pred)
                all_labels.append(y_true)
        final_loss = running_loss / total_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores
    
    def log_results(self, epoch, metric_scores, loss, 
                    report_time=False, time_mean=None,
                    split="train"):
        metrics = {"epochs": epoch+1, f"{split}_loss": loss}
        report_str = f"{split}_loss: {loss:.8f}"
        for metric_name in metric_scores.keys():
            metrics[f"{split}_{metric_name}"] = metric_scores[metric_name]
            report_str += f", {split}_{metric_name}: {metric_scores[metric_name]:.8f}"
        if report_time:
            metrics[f"{split}_time_mean"] = time_mean
        print(report_str)
        return metrics

    def run(self, report_time: bool = False):
        train_time_mean, test_time_mean, val_time_mean = None, None, None
        epoch = 0
        train_t0 = time.time()
        train_loss, train_metric_scores = self.eval_epoch(data_split_name="train")
        train_t1 = time.time()
        if report_time:
            print(f"Inference duration with training set: {train_t1 - train_t0} seconds")
            train_time_mean = train_t1 - train_t0
        test_t0 = time.time()
        test_loss, test_metric_scores = self.eval_epoch(data_split_name="test")
        test_t1 = time.time()
        if report_time:
            print(f"Inference duration with test set: {test_t1 - test_t0} seconds")
            test_time_mean = test_t1 - test_t0
        val_t0 = time.time()
        val_loss, val_metric_scores = self.eval_epoch(data_split_name="val")
        val_t1 = time.time()
        if report_time:
            print(f"Inference duration with validation set: {val_t1 - val_t0} seconds")
            val_time_mean = val_t1 - val_t0
        train_metrics = self.log_results(epoch, train_metric_scores, train_loss, report_time, train_time_mean, split="train")
        val_metrics = self.log_results(epoch, val_metric_scores, val_loss, report_time, val_time_mean, split="val")
        if not(self.inductive):
            test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")
        
        primary_metric = self.metrics.get_metric_list()[0]
        secondary_metric = self.metrics.get_metric_list()[1] if self.metrics.get_metric_list()[1] else None
        val_track_metric = val_metric_scores[primary_metric]
        inference_metrics = {f"best_val_{primary_metric}": val_track_metric}
        if not(self.inductive):
            test_track_metric = test_metric_scores[primary_metric]
            inference_metrics[f"best_test_{primary_metric}"] = test_track_metric
        if secondary_metric:
            if self.inductive:
                val_secondary_metric = val_metric_scores[secondary_metric]
                inference_metrics[f"best_val_{secondary_metric}"] = val_secondary_metric
            else:
                test_secondary_metric = test_metric_scores[secondary_metric]
                inference_metrics[f"best_test_{secondary_metric}"] = test_secondary_metric

        return inference_metrics
