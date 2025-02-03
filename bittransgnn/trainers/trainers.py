import time
from typing import Optional
import copy

import torch
from torch.utils.data import DataLoader, TensorDataset
from data.loader.dataloaders import GraphDataObject, TextDataObject

import numpy as np

from metrics import Metrics, compute_loss, distillation_loss, prep_logits

class BitTransformerTrainer:
    def __init__(self, model, dataset_name, optimizer, scheduler, text_data: TextDataObject, device, 
                 eval_test = True, eval_test_every_n_epochs: int = 1):
        self.model = model
        self.dataset_name = dataset_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.text_data = text_data
        self.device = device
        self.eval_test = eval_test
        self.eval_test_every_n_epochs = eval_test_every_n_epochs
        self.metrics = Metrics(dataset_name)

    def train_step(self, batch):
        (input_ids, attention_mask, label) = [x.to(self.device) for x in batch]
        y_pred, logits = self.model(input_ids, attention_mask)
        y_true = label
        loss = compute_loss(y_pred=y_pred, y_true=y_true, dataset_name=self.dataset_name)
        return loss, y_pred, y_true, logits
    
    def train_epoch(self):
        running_loss = 0
        total_samples = 0
        self.model.train()
        all_cls_logits = []
        all_preds = []
        all_labels = []
        print("train")
        for i, batch in enumerate(self.text_data.loaders["train"]):
            self.optimizer.zero_grad()
            loss, y_pred, y_true, logits = self.train_step(batch)
            loss.backward()
            self.optimizer.step()
            batch_size = len(y_pred)
            total_samples += batch_size
            running_loss += loss.item() * batch_size
            y_pred, y_true = prep_logits(y_pred, y_true, self.dataset_name)
            all_cls_logits.append(logits["cls_logit"].detach().cpu())
            all_preds.append(y_pred)
            all_labels.append(y_true)
        final_loss = running_loss / total_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_cls_logits = np.concatenate(all_cls_logits)
        logits = {"preds": all_preds, "labels": all_labels, "cls_logits": all_cls_logits}
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores, logits
    
    def eval_step(self, batch):
        (input_ids, attention_mask, label) = [x.to(self.device) for x in batch]
        y_pred, logits = self.model(input_ids, attention_mask)
        y_true = label
        loss = compute_loss(y_true=y_true, y_pred=y_pred, dataset_name=self.dataset_name)
        return loss, y_pred, y_true, logits
    
    def eval_epoch(self, data_split_name="val"):
        running_loss = 0
        total_samples = 0
        self.model.eval()
        all_cls_logits = []
        all_preds = []
        all_labels = []
        print(data_split_name)
        with torch.no_grad():
            for i, batch in enumerate(self.text_data.loaders[data_split_name]):
                loss, y_pred, y_true, logits = self.eval_step(batch)
                batch_size = len(y_pred)
                total_samples += batch_size
                running_loss += loss.item() * batch_size
                y_pred, y_true = prep_logits(y_pred, y_true, self.dataset_name)
                all_cls_logits.append(logits["cls_logit"].detach().cpu())
                all_preds.append(y_pred)
                all_labels.append(y_true)
        final_loss = running_loss / total_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_cls_logits = np.concatenate(all_cls_logits)
        logits = {"preds": all_preds, "labels": all_labels, "cls_logits": all_cls_logits}
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores, logits
    
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

    def run(self, nb_epochs: int, patience: int, 
            report_time: bool = False, model_ckpt_dir=None):
        #best_val_metric = 0
        if self.dataset_name == "cola":
            best_val_metric = -100
        else:
            best_val_metric = 0
        train_time_list = []
        test_time_list = []
        val_time_list = []
        best_metrics = {}
        model_checkpoint = {}
        logits = {}
        self.best_model = None
        self.best_logits = None
        train_time_mean, test_time_mean, val_time_mean = None, None, None
        early_stopping = EarlyStopping(patience=patience)
        for epoch in range(nb_epochs):
            if ((epoch+1) % self.eval_test_every_n_epochs == 0 and self.eval_test) or epoch == 0:
                eval_test_epoch = True
            else:
                eval_test_epoch = False
            print(f'Epoch: {(epoch+1):03d}')
            train_t0 = time.time()
            train_loss, train_metric_scores, train_logits = self.train_epoch()
            logits["train_logits"] = copy.deepcopy(train_logits)
            train_t1 = time.time()
            if report_time:
                print(f"Training loop duration for epoch {epoch}: {train_t1 - train_t0} seconds")
                train_time_list.append(train_t1-train_t0)
                train_time_mean = np.mean(train_time_list)
            val_t0 = time.time()
            val_loss, val_metric_scores, val_logits = self.eval_epoch(data_split_name="val")
            logits["val_logits"] = copy.deepcopy(val_logits)
            val_t1 = time.time()
            if report_time:
                print(f"Validation loop duration for epoch {epoch}: {val_t1 - val_t0} seconds")
                val_time_list.append(val_t1-val_t0)
                val_time_mean = np.mean(val_time_list)
            if eval_test_epoch:
                test_t0 = time.time()
                test_loss, test_metric_scores, test_logits = self.eval_epoch(data_split_name="test")
                logits["test_logits"] = copy.deepcopy(test_logits)
                test_t1 = time.time()
                if report_time:
                    print(f"Test loop duration for epoch {epoch}: {test_t1 - test_t0} seconds")
                    test_time_list.append(test_t1-test_t0)
                    test_time_mean = np.mean(test_time_list)
            self.scheduler.step()
            train_metrics = self.log_results(epoch, train_metric_scores, train_loss, report_time, train_time_mean, split="train")
            val_metrics = self.log_results(epoch, val_metric_scores, val_loss, report_time, val_time_mean, split="val")
            if eval_test_epoch:
                test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")

            primary_metric = self.metrics.get_metric_list()[0]
            secondary_metric = self.metrics.get_metric_list()[1] if self.metrics.get_metric_list()[1] else None
            val_track_metric = val_metric_scores[primary_metric]
            
            if val_track_metric > best_val_metric:
                if not(eval_test_epoch):
                    test_loss, test_metric_scores, test_logits = self.eval_epoch(data_split_name="test")
                    logits["test_logits"] = copy.deepcopy(test_logits)
                    test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")
                print("New checkpoint")
                self.best_model = copy.deepcopy(self.model)
                self.best_logits = copy.deepcopy(logits)
                model_checkpoint["epoch"] = epoch
                model_checkpoint["optimizer"] = copy.deepcopy(self.optimizer.state_dict())      

                best_val_metric = val_track_metric
                best_metrics[f"best_val_{primary_metric}"] = best_val_metric
                #if eval_test_epoch:
                best_test_metric = test_metric_scores[primary_metric]
                best_metrics[f"best_test_{primary_metric}"] =  best_test_metric
                #if secondary_metric and eval_test_epoch:
                if secondary_metric:
                    best_secondary_metric = test_metric_scores[secondary_metric]
                    best_metrics[f"best_test_{secondary_metric}"] = best_secondary_metric

            best_val_metric, early_stop = early_stopping(val_metric=best_val_metric, epoch=epoch)

            if early_stop:
                print("Results from converged checkpoint:")
                print(best_metrics)
                model_checkpoint["bert_model"] = copy.deepcopy(self.best_model.bert_model.state_dict())
                model_checkpoint["classifier"] = copy.deepcopy(self.best_model.classifier.state_dict())
                return model_checkpoint, best_metrics, self.best_logits
            if torch.cuda.is_available:
                torch.cuda.empty_cache()
        return model_checkpoint, best_metrics, self.best_logits

class BitTransGNNTrainer:
    def __init__(self, model, dataset_name, 
                 optimizer, scheduler, 
                 graph_data: GraphDataObject, 
                 joint_training, device, batch_size,
                 inductive=False,
                 eval_test=True, eval_test_every_n_epochs: int = 1):
        self.model = model
        self.dataset_name = dataset_name
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.graph_data = graph_data
        self.joint_training = joint_training
        self.device = device
        self.batch_size = batch_size
        self.inductive = inductive
        self.eval_test = eval_test
        self.eval_test_every_n_epochs = eval_test_every_n_epochs
        self.metrics = Metrics(dataset_name)

    def update_cls_feats(self):
        """
        initializes cls_feats by using bert model in inference mode
        generally not mandatory to be used in case bert model and gcn will be jointly trained
        if pretrained bert model will only be used to initialize gcn features and bert model will not be trained,
        then it is necessary to call this function
        """
        # no gradient needed
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
        self.graph_data.convert_device(device="cpu")
        self.graph_data.cls_feats[doc_mask] = cls_feat

    def train_step(self, batch):
        idx = batch[0].to(self.device)
        train_mask = self.graph_data.train_mask.to(self.device)[idx].type(torch.BoolTensor)
        y_pred, _, logits = self.model(self.graph_data.convert_device(self.device), idx)
        y_pred = y_pred[train_mask]
        for key in logits:
            logit = logits[key]
            logit = logit[train_mask]
            logits[key] = logit
        y_true = self.graph_data.train_label.to(self.device)[idx][train_mask]
        if self.dataset_name != "stsb":
            y_pred = torch.log(y_pred)
        loss = compute_loss(y_pred=y_pred, y_true=y_true, dataset_name=self.dataset_name)
        self.graph_data.cls_feats = self.graph_data.cls_feats.detach()
        return loss, y_pred, y_true, logits

    def train_epoch(self):
        running_loss = 0
        all_cls_logits = []
        all_gcn_logits = []
        all_preds = []
        all_labels = []
        total_samples = 0
        self.model.train()
        for i, batch in enumerate(self.graph_data.idx_loaders["train"]):
            self.optimizer.zero_grad()
            loss, y_pred, y_true, logits = self.train_step(batch)
            loss.backward()
            self.optimizer.step()
            batch_size = len(y_pred)
            total_samples += batch_size
            running_loss += loss.item() * batch_size
            y_pred, y_true = prep_logits(y_pred, y_true, self.dataset_name)
            all_cls_logits.append(logits["cls_logit"].detach().cpu())
            all_gcn_logits.append(logits["gcn_logit"].detach().cpu())
            all_preds.append(y_pred)
            all_labels.append(y_true)
        final_loss = running_loss / total_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_cls_logits = np.concatenate(all_cls_logits)
        all_gcn_logits = np.concatenate(all_gcn_logits)
        logits = {"preds": all_preds, "labels": all_labels, "cls_logits": all_cls_logits, "gcn_logits": all_gcn_logits}
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores, logits
    
    def eval_step(self, batch):
        idx = batch[0].to(self.device)
        y_pred, _, logits = self.model(self.graph_data.convert_device(self.device), idx)
        y_true = self.graph_data.label.to(self.device)[idx]
        if self.dataset_name != "stsb":
            y_pred = torch.log(y_pred)
        loss = compute_loss(y_true=y_true, y_pred=y_pred, dataset_name=self.dataset_name)
        return loss, y_pred, y_true, logits

    def eval_epoch(self, data_split_name="val"):
        running_loss = 0
        all_cls_logits = []
        all_gcn_logits = []
        all_preds = []
        all_labels = []
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.graph_data.idx_loaders[data_split_name]):
                loss, y_pred, y_true, logits = self.eval_step(batch)
                batch_size = len(y_pred)
                total_samples += batch_size
                running_loss += loss.item() * batch_size
                y_pred, y_true = prep_logits(y_pred, y_true, self.dataset_name)
                all_cls_logits.append(logits["cls_logit"].detach().cpu())
                all_gcn_logits.append(logits["gcn_logit"].detach().cpu())
                all_preds.append(y_pred)
                all_labels.append(y_true)
        final_loss = running_loss / total_samples
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_cls_logits = np.concatenate(all_cls_logits)
        all_gcn_logits = np.concatenate(all_gcn_logits)
        logits = {"preds": all_preds, "labels": all_labels, "cls_logits": all_cls_logits, "gcn_logits": all_gcn_logits}
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores, logits
    
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
    
    def run(self, nb_epochs: int, patience: int, 
            report_time: bool = False):
        #best_val_metric = 0
        if self.dataset_name == "cola":
            best_val_metric = -100
        else:
            best_val_metric = 0
        train_time_list = []
        test_time_list = []
        val_time_list = []
        best_metrics = {}
        model_checkpoint = {}
        self.best_model = None
        self.best_logits = None
        self.best_cls_feats = None
        logits = {}
        #self.graph_data = self.update_cls_feats()
        self.update_cls_feats()
        train_time_mean, test_time_mean, val_time_mean = None, None, None
        early_stopping = EarlyStopping(patience=patience)
        for epoch in range(nb_epochs):
            if ((epoch+1) % self.eval_test_every_n_epochs == 0 and self.eval_test) or epoch == 0:
                eval_test_epoch = True
            else:
                eval_test_epoch = False
            print(f'Epoch: {(epoch+1):03d}')
            train_t0 = time.time()
            train_loss, train_metric_scores, train_logits = self.train_epoch()
            logits["train_logits"] = copy.deepcopy(train_logits)
            train_t1 = time.time()
            if report_time:
                print(f"Training loop duration for epoch {epoch}: {train_t1 - train_t0} seconds")
                train_time_list.append(train_t1-train_t0)
                train_time_mean = np.mean(train_time_list)
            val_t0 = time.time()
            val_loss, val_metric_scores, val_logits = self.eval_epoch(data_split_name="val")
            logits["val_logits"] = copy.deepcopy(val_logits)
            val_t1 = time.time()
            if report_time:
                print(f"Validation loop duration for epoch {epoch}: {val_t1 - val_t0} seconds")
                val_time_list.append(val_t1-val_t0)
                val_time_mean = np.mean(val_time_list)
            if eval_test_epoch and not(self.inductive):
                test_t0 = time.time()
                test_loss, test_metric_scores, test_logits = self.eval_epoch(data_split_name="test")
                logits["test_logits"] = copy.deepcopy(test_logits)
                test_t1 = time.time()
                if report_time:
                    print(f"Test loop duration for epoch {epoch}: {test_t1 - test_t0} seconds")
                    test_time_list.append(test_t1-test_t0)
                    test_time_mean = np.mean(test_time_list)
            self.scheduler.step()
            train_metrics = self.log_results(epoch, train_metric_scores, train_loss, report_time, train_time_mean, split="train")
            val_metrics = self.log_results(epoch, val_metric_scores, val_loss, report_time, val_time_mean, split="val")
            if not(self.inductive) and eval_test_epoch:            
                test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")

            if self.joint_training:
                self.update_cls_feats()
            
            primary_metric = self.metrics.get_metric_list()[0]
            secondary_metric = self.metrics.get_metric_list()[1] if self.metrics.get_metric_list()[1] else None
            val_track_metric = val_metric_scores[primary_metric]
            
            if val_track_metric > best_val_metric:
                if not(eval_test_epoch) and not(self.inductive):
                    test_loss, test_metric_scores, test_logits = self.eval_epoch(data_split_name="test")
                    logits["test_logits"] = copy.deepcopy(test_logits)
                    test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")
                print("New checkpoint")
                self.best_model = copy.deepcopy(self.model)
                self.best_logits = copy.deepcopy(logits)
                self.best_cls_feats = copy.deepcopy(self.graph_data.cls_feats[self.graph_data.doc_mask.to("cpu").type(torch.BoolTensor)])
                model_checkpoint["epoch"] = epoch
                model_checkpoint["optimizer"] = copy.deepcopy(self.optimizer.state_dict())

                best_val_metric = val_track_metric
                best_metrics[f"best_val_{primary_metric}"] = best_val_metric
                if not(self.inductive):
                    best_test_metric = test_metric_scores[primary_metric]
                    best_metrics[f"best_test_{primary_metric}"] = best_test_metric
                if secondary_metric:
                    if self.inductive:
                        best_secondary_metric = val_metric_scores[secondary_metric]
                        best_metrics[f"best_val_{secondary_metric}"] = best_secondary_metric
                    else:
                        #if eval_test_epoch:
                        best_secondary_metric = test_metric_scores[secondary_metric]
                        best_metrics[f"best_test_{secondary_metric}"] = best_secondary_metric
            best_val_metric, early_stop = early_stopping(val_metric=best_val_metric, epoch=epoch)

            if early_stop:
                print("Results from converged checkpoint:")
                print(best_metrics)
                model_checkpoint["bert_model"] = copy.deepcopy(self.best_model.bert_model.state_dict())
                model_checkpoint["classifier"] = copy.deepcopy(self.best_model.classifier.state_dict())
                model_checkpoint["gcn"] = copy.deepcopy(self.best_model.gcn.state_dict())
                model_checkpoint["cls_feats"] = copy.deepcopy(self.best_cls_feats)
                model_checkpoint["lmbd"] = copy.deepcopy(self.best_model.lmbd)
                return model_checkpoint, best_metrics, self.best_logits
            if torch.cuda.is_available:
                torch.cuda.empty_cache()
        return model_checkpoint, best_metrics, self.best_logits

class BitTransGNNKDTrainer:
    def __init__(self, teacher_model, student_model, dataset_name, student_optimizer, student_scheduler, 
                 graph_data, 
                 alpha_d, temperature, 
                 device, 
                 batch_size,
                 distillation_type, 
                 ext_cls_feats=None, 
                 teacher_optimizer=None, teacher_scheduler=None,
                 inductive=False,
                 eval_test=True, eval_test_every_n_epochs: int = 1):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.dataset_name = dataset_name
        self.student_optimizer = student_optimizer
        self.student_scheduler = student_scheduler
        self.graph_data = graph_data
        self.alpha_d = alpha_d
        self.temperature = temperature
        self.device = device
        self.distillation_type = distillation_type
        self.ext_cls_feats = ext_cls_feats
        if distillation_type == "online":
            assert (teacher_optimizer is not None) and (teacher_scheduler is not None)
            self.teacher_optimizer = teacher_optimizer
            self.teacher_scheduler = teacher_scheduler
        elif distillation_type == "offline":
            self.teacher_optimizer = None
            self.teacher_scheduler = None
        self.batch_size = batch_size
        self.inductive = inductive
        self.eval_test = eval_test
        self.eval_test_every_n_epochs = eval_test_every_n_epochs
        self.metrics = Metrics(dataset_name)

    def update_cls_feats(self):
        """
        initializes cls_feats by using bert model in inference mode
        generally not mandatory to be used in case bert model and gcn will be jointly trained
        if pretrained bert model will only be used to initialize gcn features and bert model will not be trained,
        then it is necessary to call this function
        """
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
        self.graph_data.convert_device(device="cpu")
        self.graph_data.cls_feats[doc_mask] = cls_feat

    def train_step(self, batch):
        idx = batch[0].to(self.device)
        train_mask = self.graph_data.train_mask.to(self.device)[idx].type(torch.BoolTensor)
        y_pred, y_soft_pred, student_logits = self.student_model(self.graph_data.convert_device(self.device), idx, temperature=self.temperature)
        y_pred, y_soft_pred = y_pred[train_mask], y_soft_pred[train_mask]
        for key in student_logits:
            logit = student_logits[key]
            logit = logit[train_mask]
            student_logits[key] = logit
        if self.distillation_type == "offline":
            with torch.no_grad():
                y_teacher, y_teacher_soft, _ = self.teacher_model(self.graph_data.convert_device(self.device), idx, temperature=self.temperature)
                y_teacher, y_teacher_soft = y_teacher[train_mask], y_teacher_soft[train_mask]
        elif self.distillation_type == "online":
            y_teacher, y_teacher_soft, _ = self.teacher_model(self.graph_data.convert_device(self.device), idx, temperature=self.temperature)
            y_teacher, y_teacher_soft = y_teacher[train_mask], y_teacher_soft[train_mask]
        distill_loss = distillation_loss(student_out=y_soft_pred, teacher_out=y_teacher_soft, temperature=self.temperature, dataset_name=self.dataset_name)
        y_true = self.graph_data.train_label.to(self.device)[idx][train_mask]
        true_loss = compute_loss(y_true=y_true, y_pred=y_pred, dataset_name=self.dataset_name)
        self.graph_data.cls_feats = self.graph_data.cls_feats.detach()
        return distill_loss, true_loss, y_pred, y_true, student_logits

    def train_epoch(self):
        running_distill_loss = 0
        running_true_loss = 0
        running_loss = 0
        all_cls_logits = []
        all_preds = []
        all_labels = []
        total_samples = 0
        if self.distillation_type == "offline":
            self.student_model.train()
            self.teacher_model.eval()
        elif self.distillation_type == "online":
            self.student_model.train()
            self.teacher_model.train()
        for i, batch in enumerate(self.graph_data.idx_loaders["train"]):
            self.student_optimizer.zero_grad()
            distill_loss, true_loss, y_pred, y_true, logits = self.train_step(batch)
            loss = self.alpha_d * distill_loss + (1-self.alpha_d) * true_loss
            loss.backward()
            self.student_optimizer.step()
            if self.distillation_type == "online":
                self.teacher_optimizer.step()
            batch_size = len(y_pred)
            running_distill_loss += distill_loss.item() * batch_size
            running_true_loss += true_loss.item() * batch_size
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            y_pred, y_true = prep_logits(y_pred, y_true, self.dataset_name)
            all_cls_logits.append(logits["cls_logit"].detach().cpu())
            all_preds.append(y_pred)
            all_labels.append(y_true)
        final_distill_loss = running_distill_loss / total_samples
        final_true_loss = running_true_loss / total_samples
        final_loss = running_loss / total_samples
        all_cls_logits = np.concatenate(all_cls_logits)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        logits = {"preds": all_preds, "labels": all_labels, "cls_logits": all_cls_logits}
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores, final_distill_loss, final_true_loss, logits
    
    def eval_step(self, batch):
        idx = batch[0].to(self.device)
        y_pred, y_soft_pred, logits = self.student_model(self.graph_data.convert_device(self.device), idx)
        y_teacher, y_teacher_soft, _ = self.teacher_model(self.graph_data.convert_device(self.device), idx)
        distill_loss = distillation_loss(student_out=y_soft_pred, teacher_out=y_teacher_soft, temperature=self.temperature, dataset_name=self.dataset_name)
        y_true = self.graph_data.label.to(self.device)[idx]
        true_loss = compute_loss(y_true=y_true, y_pred=y_pred, dataset_name=self.dataset_name)
        return distill_loss, true_loss, y_pred, y_true, logits

    def eval_epoch(self, data_split_name="val"):
        running_distill_loss = 0
        running_true_loss = 0
        running_loss = 0
        all_cls_logits = []
        all_preds = []
        all_labels = []
        total_samples = 0
        self.student_model.eval()
        self.teacher_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.graph_data.idx_loaders[data_split_name]):
                distill_loss, true_loss, y_pred, y_true, logits = self.eval_step(batch)
                loss = self.alpha_d * distill_loss + (1-self.alpha_d) * true_loss
                batch_size = len(y_pred)
                running_distill_loss += distill_loss.item() * batch_size
                running_true_loss += true_loss.item() * batch_size
                running_loss += loss.item() * batch_size
                total_samples += batch_size
                y_pred, y_true = prep_logits(y_pred, y_true, self.dataset_name)
                all_cls_logits.append(logits["cls_logit"].detach().cpu())
                all_preds.append(y_pred)
                all_labels.append(y_true)
        final_distill_loss = running_distill_loss / total_samples
        final_true_loss = running_true_loss / total_samples
        final_loss = running_loss / total_samples
        all_cls_logits = np.concatenate(all_cls_logits)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        logits = {"preds": all_preds, "labels": all_labels, "cls_logits": all_cls_logits}
        final_metric_scores = self.metrics.compute_metrics(all_preds, all_labels)
        return final_loss, final_metric_scores, final_distill_loss, final_true_loss, logits

    def log_results(self, epoch, metric_scores, loss, distill_loss, true_loss,
                    report_time=False, time_mean=None,
                    split="train"):
        metrics = {"epochs": epoch+1, f"{split}_loss": loss, f"{split}_distill_loss": distill_loss, f"{split}_true_loss": true_loss}
        report_str = f"{split}_loss: {loss:.8f}, {split}_distill_loss: {distill_loss:.8f}, {split}_true_loss: {true_loss:.8f}"
        for metric_name in metric_scores.keys():
            metrics[f"{split}_{metric_name}"] = metric_scores[metric_name]
            report_str += f", {split}_{metric_name}: {metric_scores[metric_name]:.8f}"
        if report_time:
            metrics[f"{split}_time_mean"] = time_mean
        print(report_str)
        return metrics

    def run(self, nb_epochs: int, patience: int, 
            report_time: bool = False):
        #best_val_metric = 0
        if self.dataset_name == "cola":
            best_val_metric = -100
        else:
            best_val_metric = 0
        train_time_list = []
        test_time_list = []
        val_time_list = []
        best_metrics = {}
        model_checkpoint = {}
        logits = {}
        self.best_model = None
        self.best_logits = None
        train_time_mean, test_time_mean, val_time_mean = None, None, None
        early_stopping = EarlyStopping(patience=patience)
        self.update_cls_feats(self.ext_cls_feats)
        for epoch in range(nb_epochs):
            if ((epoch+1) % self.eval_test_every_n_epochs == 0 and self.eval_test) or epoch == 0:
                eval_test_epoch = True
            else:
                eval_test_epoch = False
            print(f'Epoch: {(epoch+1):03d}')
            train_t0 = time.time()
            train_loss, train_metric_scores, train_distill_loss, train_true_loss, train_logits = self.train_epoch()
            logits["train_logits"] = copy.deepcopy(train_logits)
            train_t1 = time.time()
            if report_time:
                print(f"Training loop duration for epoch {epoch}: {train_t1 - train_t0} seconds")
                train_time_list.append(train_t1-train_t0)
                train_time_mean = np.mean(train_time_list)
            val_t0 = time.time()
            val_loss, val_metric_scores, val_distill_loss, val_true_loss, val_logits = self.eval_epoch(data_split_name="val")
            logits["val_logits"] = copy.deepcopy(val_logits)
            val_t1 = time.time()
            if report_time:
                print(f"Validation loop duration for epoch {epoch}: {val_t1 - val_t0} seconds")
                val_time_list.append(val_t1-val_t0)
                val_time_mean = np.mean(val_time_list)
            if eval_test_epoch and not(self.inductive):
                test_t0 = time.time()
                test_loss, test_metric_scores, test_distill_loss, test_true_loss, test_logits = self.eval_epoch(data_split_name="test")
                logits["test_logits"] = copy.deepcopy(test_logits)
                test_t1 = time.time()
                if report_time:
                    print(f"Test loop duration for epoch {epoch}: {test_t1 - test_t0} seconds")
                    test_time_list.append(test_t1-test_t0)
                    test_time_mean = np.mean(test_time_list)
            self.student_scheduler.step()
            train_metrics = self.log_results(epoch, train_metric_scores, train_loss, train_distill_loss, train_true_loss, report_time=False, time_mean=None, split="train")
            val_metrics = self.log_results(epoch, val_metric_scores, val_loss, val_distill_loss, val_true_loss, report_time=False, time_mean=None, split="val")
            if not(self.inductive) and eval_test_epoch:            
                test_metrics = self.log_results(epoch, test_metric_scores, test_loss, test_distill_loss, test_true_loss, report_time=False, time_mean=None, split="test")
            
            if self.distillation_type == "online":
                self.update_cls_feats()

            primary_metric = self.metrics.get_metric_list()[0]
            secondary_metric = self.metrics.get_metric_list()[1] if self.metrics.get_metric_list()[1] else None
            val_track_metric = val_metric_scores[primary_metric]
            
            if val_track_metric > best_val_metric:
                if not(eval_test_epoch) and not(self.inductive):
                    test_loss, test_metric_scores, test_distill_loss, test_true_loss, test_logits = self.eval_epoch(data_split_name="test")
                    logits["test_logits"] = copy.deepcopy(test_logits)
                    test_metrics = self.log_results(epoch, test_metric_scores, test_loss, test_distill_loss, test_true_loss, report_time=False, time_mean=None, split="test")
                print("New checkpoint")
                self.best_model = copy.deepcopy(self.student_model)
                self.best_logits = copy.deepcopy(logits)
                model_checkpoint["epoch"] = epoch
                model_checkpoint["optimizer"] = copy.deepcopy(self.student_optimizer.state_dict())      

                best_val_metric = val_track_metric
                best_metrics[f"best_val_{primary_metric}"] = best_val_metric
                if not(self.inductive):
                    best_test_metric = test_metric_scores[primary_metric]
                    best_metrics[f"best_test_{primary_metric}"] = best_test_metric
                if secondary_metric:
                    if self.inductive:
                        best_secondary_metric = val_metric_scores[secondary_metric]
                        best_metrics[f"best_val_{secondary_metric}"] = best_secondary_metric
                    else:
                        #if eval_test_epoch:
                        best_secondary_metric = test_metric_scores[secondary_metric]
                        best_metrics[f"best_test_{secondary_metric}"] = best_secondary_metric
            best_val_metric, early_stop = early_stopping(val_metric=best_val_metric, epoch=epoch)

            if early_stop:
                print("Results from converged checkpoint:")
                print(best_metrics)
                model_checkpoint["bert_model"] = copy.deepcopy(self.best_model.bert_model.state_dict())
                model_checkpoint["classifier"] = copy.deepcopy(self.best_model.classifier.state_dict())
                return model_checkpoint, best_metrics, self.best_logits
            if torch.cuda.is_available:
                torch.cuda.empty_cache()
        return model_checkpoint, best_metrics, self.best_logits

class EarlyStopping:
    def __init__(self, patience: int, check_finite: bool = True, metric = "acc"):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.check_finite = check_finite
        self.metric = metric
        self.conv_epoch = None
        if metric == "acc":
            self.best_val_metric = 0.0
        elif metric == "loss":
            self.best_val_metric = 1e9

    def __call__(self, val_metric: float, epoch: Optional[int] = None):
        if self.metric == "acc":
            if val_metric > self.best_val_metric:
                self.counter = 0
                self.best_val_metric = val_metric
                if epoch is not None:
                    self.conv_epoch = epoch
            else:
                self.counter += 1
        elif self.metric == "loss":
            if val_metric < self.best_val_metric:
                self.counter = 0
                self.best_val_metric = val_metric
                if epoch is not None:
                    self.conv_epoch = epoch
            else:
                self.counter += 1

        if self.check_finite:
            if not np.isfinite(val_metric):
                self.early_stop = True
                print(f"check_finite flag raised, executing early stopping and terminating training...")
                print(f"Model training finished at epoch {epoch+1}.")
        
        if self.counter >= self.patience:
            self.early_stop = True
            print(f"Val {self.metric} has not improved for {self.patience} epochs, executing early stopping and terminating training...")
            if self.conv_epoch is not None:
                print(f"Model converged at epoch {self.conv_epoch+1}.")
        return self.best_val_metric, self.early_stop
