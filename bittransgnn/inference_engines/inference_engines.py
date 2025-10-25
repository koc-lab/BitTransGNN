import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from data.loader.dataloaders import GraphDataObject, TextDataObject
from metrics import Metrics, compute_loss, prep_logits

class BitTransformerInference:
    def __init__(self, model, dataset_name, text_data: TextDataObject, device):
        self.model = model
        self.dataset_name = dataset_name
        self.text_data = text_data
        self.device = device
        self.metrics = Metrics(dataset_name)

    def eval_step(self, batch):
        (input_ids, attention_mask, token_type_ids, label) = [x.to(self.device) for x in batch]
        y_pred, logits = self.model(input_ids, attention_mask, token_type_ids)
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
        #print("all_labels")
        #print(all_labels)
        #print("all_preds")
        #print(all_preds)
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

    def run(self, report_time: bool = False):
        train_time_mean, test_time_mean, val_time_mean = None, None, None
        epoch = 0
        train_t0 = time.time()
        train_loss, train_metric_scores, train_logits = self.eval_epoch(data_split_name="train")
        train_t1 = time.time()
        if report_time:
            print(f"Inference duration with training set: {train_t1 - train_t0} seconds")
            train_time_mean = train_t1 - train_t0
        test_t0 = time.time()
        test_loss, test_metric_scores, test_logits = self.eval_epoch(data_split_name="test")
        test_t1 = time.time()
        if report_time:
            print(f"Inference duration with test set: {test_t1 - test_t0} seconds")
            test_time_mean = test_t1 - test_t0
        val_t0 = time.time()
        val_loss, val_metric_scores, val_logits = self.eval_epoch(data_split_name="val")
        val_t1 = time.time()
        if report_time:
            print(f"Inference duration with validation set: {val_t1 - val_t0} seconds")
            val_time_mean = val_t1 - val_t0
        train_metrics = self.log_results(epoch, train_metric_scores, train_loss, report_time, train_time_mean, split="train")
        val_metrics = self.log_results(epoch, val_metric_scores, val_loss, report_time, val_time_mean, split="val")
        test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")
        logits = {"train_logits": train_logits, "test_logits": test_logits, "val_logits": val_logits}
        primary_metric = self.metrics.get_metric_list()[0]
        secondary_metric = self.metrics.get_metric_list()[1] if self.metrics.get_metric_list()[1] else None
        val_track_metric = val_metric_scores[primary_metric]
        test_track_metric = test_metric_scores[primary_metric]
        inference_metrics = {f"best_val_{primary_metric}": val_track_metric, f"best_test_{primary_metric}": test_track_metric}
        if secondary_metric:
            test_secondary_metric = test_metric_scores[secondary_metric]
            inference_metrics[f"best_test_{secondary_metric}"] = test_secondary_metric
        inference_metrics["train_time_mean"] = train_time_mean
        inference_metrics["val_time_mean"] = val_time_mean
        inference_metrics["test_time_mean"] = test_time_mean
        return inference_metrics, logits

    def time_text_epoch(self, splits=("train", "test", "val"), warmup=1, repeat=5):
        """
        Times a full end-to-end BERT inference epoch over `split`.
        """
        def _fn():
            self.model.eval()
            with torch.inference_mode():
                for split in splits:
                    _ = self.eval_epoch(data_split_name=split)
        stats, _ = time_callable(_fn, warmup=warmup, repeat=repeat, device=self.device)
        stats["what"] = f"text_full_epoch_{'_'.join(splits)}"
        stats["splits"] = list(splits)
        return stats

    
class BitTransGNNInference:
    def __init__(self, model, dataset_name, graph_data: GraphDataObject, interp_outs, joint_training, device, ext_cls_feats, batch_size, inductive=False, recompute_bert=False):
        self.model = model
        self.dataset_name = dataset_name
        self.graph_data = graph_data
        self.interp_outs = interp_outs
        self.joint_training = joint_training
        self.device = device
        self.ext_cls_feats = ext_cls_feats
        self.batch_size = batch_size
        self.inductive = inductive
        self.recompute_bert = recompute_bert
        self.metrics = Metrics(dataset_name)

    def update_cls_feats(self, ext_cls_feats=None):
        """
        initializes cls_feats by using bert model in inference mode
        generally not mandatory to be used in case bert model and gcn will be jointly trained
        if pretrained bert model will only be used to initialize gcn features and bert model will not be trained,
        then it is necessary to call this function
        """        
        # no gradient needed, since we are only using bert model to get cls features.
        #doc_mask = self.graph_data.doc_mask.to(self.device).type(torch.BoolTensor)
        doc_mask = self.graph_data.doc_mask.to(self.device).bool()
        if self.ext_cls_feats is not None:
            dest = self.graph_data.cls_feats
            mask_for_dest = doc_mask.to(dest.device)
            src = ext_cls_feats.to(dest.device)
            with torch.inference_mode(False):
                self.graph_data.cls_feats = dest.clone().detach()
                self.graph_data.cls_feats.requires_grad_(False)
                self.graph_data.cls_feats[mask_for_dest] = src
        else:
            input_ids_full = self.graph_data.input_ids
            attention_mask_full = self.graph_data.attention_mask
            token_type_ids_full = self.graph_data.token_type_ids
            dataloader = DataLoader(
                TensorDataset(input_ids_full.to(self.device)[doc_mask], attention_mask_full.to(self.device)[doc_mask], token_type_ids_full.to(self.device)[doc_mask]),
                batch_size=self.batch_size
            )
            with torch.no_grad():
                self.model.eval()
                cls_list = []
                for i, batch in enumerate(dataloader):
                    input_ids, attention_mask, token_type_ids = [x.to(self.device) for x in batch]
                    output = self.model.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0][:, 0]
                    cls_list.append(output.cpu())
                cls_feat = torch.cat(cls_list, axis=0)
            dest = self.graph_data.cls_feats
            mask_for_dest = doc_mask.to(dest.device)
            src = cls_feat.to(dest.device)

            with torch.inference_mode(False):
                self.graph_data.cls_feats = dest.clone().detach()
                self.graph_data.cls_feats.requires_grad_(False)
                self.graph_data.cls_feats[mask_for_dest] = src

            self.graph_data = self.graph_data.convert_device(device="cpu")
            
    
    def eval_step(self, batch):
        idx = batch[0].to(self.device)
        y_pred, _, logits = self.model(self.graph_data.convert_device(self.device), idx, recompute_bert=self.recompute_bert)
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

    def run(self, report_time: bool = False):
        self.update_cls_feats(self.ext_cls_feats)
        train_time_mean, test_time_mean, val_time_mean = None, None, None
        epoch = 0
        train_t0 = time.time()
        train_loss, train_metric_scores, train_logits = self.eval_epoch(data_split_name="train")
        train_t1 = time.time()
        if report_time:
            print(f"Inference duration with training set: {train_t1 - train_t0} seconds")
            train_time_mean = train_t1 - train_t0
        test_t0 = time.time()
        test_loss, test_metric_scores, test_logits = self.eval_epoch(data_split_name="test")
        test_t1 = time.time()
        if report_time:
            print(f"Inference duration with test set: {test_t1 - test_t0} seconds")
            test_time_mean = test_t1 - test_t0
        val_t0 = time.time()
        val_loss, val_metric_scores, val_logits = self.eval_epoch(data_split_name="val")
        val_t1 = time.time()
        if report_time:
            print(f"Inference duration with validation set: {val_t1 - val_t0} seconds")
            val_time_mean = val_t1 - val_t0
        train_metrics = self.log_results(epoch, train_metric_scores, train_loss, report_time, train_time_mean, split="train")
        val_metrics = self.log_results(epoch, val_metric_scores, val_loss, report_time, val_time_mean, split="val")
        if not(self.inductive):
            test_metrics = self.log_results(epoch, test_metric_scores, test_loss, report_time, test_time_mean, split="test")
        
        logits = {"train_logits": train_logits, "test_logits": test_logits, "val_logits": val_logits}
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
        inference_metrics["train_time_mean"] = train_time_mean
        inference_metrics["val_time_mean"] = val_time_mean
        inference_metrics["test_time_mean"] = test_time_mean
        return inference_metrics, logits
    
    def time_update_cls_feats(self, warmup=0, repeat=1):
        """
        Times the one-time CLS feature extraction over train+val+test (as your current code does).
        Note: update_cls_feats() ends by moving graph_data to CPU; that's fine since it's a one-time step.
        """
        def _fn():
            # IMPORTANT: do not pass ext_cls_feats here unless you intend to *skip* BERT.
            # We want to time the BERT forward path, so we force using the internal pipeline:
            self.update_cls_feats(ext_cls_feats=None)
        stats, _ = time_callable(_fn, warmup=warmup, repeat=repeat, device=self.device)
        stats["what"] = "bert_cls_feature_extraction_fullset"
        return stats

    def time_gcn_full_graph_epoch(self, splits=("train","val","test"), warmup=1, repeat=5):
        """
        Times a *single* GCN full-graph 'epoch' by sequentially running eval over the requested splits.
        This matches Stage A coverage (train+val+test) so numbers are comparable.
        Includes dataloader H2D and any .cpu() in eval_epoch => 'end-to-end' timing.
        """
        def _one_epoch_fullset():
            self.model.eval()
            with torch.inference_mode():
                # Run the same eval path you already use, per split:
                for sp in splits:
                    _ = self.eval_epoch(data_split_name=sp)
            return None

        stats, _ = time_callable(_one_epoch_fullset, warmup=warmup, repeat=repeat, device=self.device)
        stats["what"] = f"gcn_full_graph_epoch_{'_'.join(splits)}"
        stats["splits"] = list(splits)
        return stats

    def time_gcn_full_graph_epoch_static(self, splits=("train","test","val"), warmup=1, repeat=5):
        idx_all = get_full_doc_indices(self.graph_data).to(self.device)
        full_batch = (idx_all,)  # match your loader batch shape: tuple of one tensor

        def _one_epoch_fullset_static():
            self.model.eval()
            with torch.inference_mode():
                # Run the same eval path you already use, per split:
                _ = self.eval_step(full_batch)
            return None
        stats, _ = time_callable(_one_epoch_fullset_static, warmup=warmup, repeat=repeat, device=self.device)
        stats["what"] = f"gcn_full_graph_epoch_static_{'_'.join(splits)}"
        stats["splits"] = list(splits)
        return stats


def get_full_doc_indices(graph_data):
    if hasattr(graph_data, "doc_mask") and graph_data.doc_mask is not None:
        return torch.nonzero(graph_data.doc_mask, as_tuple=False).squeeze(1).long()
    if hasattr(graph_data, "label"):
        return torch.arange(graph_data.label.shape[0], dtype=torch.long)
    raise RuntimeError("Cannot infer full document indices; provide doc_mask or label.")


import time
import numpy as np
import torch

class Timer:
    def __init__(self, device):
        self.device = device
    def sync(self):
        if torch.cuda.is_available() and "cuda" in str(self.device):
            torch.cuda.synchronize()
    def now(self):
        return time.perf_counter()

def summarize_seconds(times_s):
    a = np.array(times_s, dtype=float)
    return {
        "median_s": float(np.median(a)),
        "p5_s": float(np.percentile(a, 5)),
        "p95_s": float(np.percentile(a, 95)),
        "mean_s": float(a.mean()),
        "std_s": float(a.std(ddof=1) if a.size > 1 else 0.0),
        "n": int(a.size),
    }

def time_callable(fn, *, warmup=1, repeat=5, device="cpu"):
    """
    Times a zero-arg callable `fn` with warm-ups and sync.
    Returns (stats_dict, last_result).
    """
    t = Timer(device)
    # warm-ups (not recorded)
    for _ in range(max(0, warmup)):
        t.sync(); _ = fn(); t.sync()
    # timed
    times = []
    last = None
    for _ in range(max(1, repeat)):
        t.sync()
        t0 = t.now()
        last = fn()
        t.sync()
        t1 = t.now()
        times.append(t1 - t0)
    return summarize_seconds(times), last
