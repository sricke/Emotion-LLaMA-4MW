"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from importlib.readers import NamespaceReader
import logging
import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from sklearn import metrics as met
import numpy as np
import wandb
from tqdm import tqdm
class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.cfg = ""
        self.class2idx = None

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        self.cfg = cfg
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg, fold):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            datasets[name] = builder.build_datasets(fold=fold+1)

            if name == "eval":
                self.class2idx = datasets[name].mw2idx

            """
            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

           datasets[name] = dataset
            """

        return datasets

    def train_step(self, model, samples):
        model.train()
        outputs = model(samples, return_input_embeds=False, return_pred_texts=True)
        # loss = outputs["loss"] + outputs["emos_loss"]
        
        # print(outputs["loss"], outputs["emos_loss"], torch.argmax(outputs['emos_pred'], dim=1), outputs["emotion"])

        return outputs['loss'], outputs['pred_texts'], outputs['gt_texts']
    
    @torch.no_grad()
    def valid_step(self, model, samples):
        model.eval()
        # Call forward() first to get loss, then generate from output embeddings
        output = model(samples, return_input_embeds=True)
        loss = output["loss"]
        
        # Extract condition embeddings from forward pass
        result = model.generate(
            images=None,  # Not needed if using pre-computed embeddings
            video_features=None,
            texts=None,
            num_beams=1, 
            max_new_tokens=20,
            temperature=1.0,
            do_sample=False,
            input_embeds=output["input_embeds"],
            input_attn_mask=output["input_attn_mask"],
        )

        return {
            "loss": result.get("loss", loss),
            "answers": result["answers"],
            "generated_token_ids": result.get("generated_token_ids", None),
        }
        
    def calculate_log_evaluation_metrics(self, metric_logger, predicted, labels):
        labels = [label.lower().strip() for label in labels]
        predicted = [pred.lower().strip() for pred in predicted]
        prec, rec, f1, sup = met.precision_recall_fscore_support(labels, predicted, labels=["non-mind wandering", "mind wandering"])
        #auc = met.roc_auc_score(labels, predicted)
        bal = met.balanced_accuracy_score(labels, predicted)
        metric_logger.update(val_mw_precision=prec[1]) #only for mind wandering
        metric_logger.update(val_mw_f1_score=f1[1]) #only for mind wandering
        metric_logger.update(val_mw_support=sup[1]) #only for mind wandering
        metric_logger.update(val_mw_recall=rec[1]) #only for mind wandering
        #metric_logger.update(val_auc=auc.item())
        metric_logger.update(val_balanced_accuracy=bal)

        
    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, metric_logger, cuda_enabled=True):
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        logging.info(f"Evaluating {len(data_loader)} samples")
        results = []
        
        video_ids = []
        all_predictions = []
        all_labels = []
        #for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        for step in tqdm(range(len(data_loader))):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            logging.info(f"Step {step} of {len(data_loader)}")
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            video_ids.extend(samples['video_id'])
            eval_output = self.valid_step(model=model, samples=samples)
            results.append(eval_output)  # Append dict instead of extend
            if eval_output['loss'] is not None:
                metric_logger.update(val_loss=eval_output['loss'].item())
                
            # Accumulate predictions and labels
            batch_answers = eval_output['answers']
            batch_labels = samples['label']
            
            # Handle both single items and lists
            if not isinstance(batch_answers, list):
                batch_answers = [batch_answers]
            if not isinstance(batch_labels, list):
                batch_labels = [batch_labels]
                
            all_predictions.extend(batch_answers)
            all_labels.extend(batch_labels)
        
        self.calculate_log_evaluation_metrics(metric_logger=metric_logger, predicted=all_predictions, labels=all_labels)
        eval_metrics =  {
            k: "{:.6f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        if is_dist_avail_and_initialized():
            dist.barrier()

        return {
            "predictions": all_predictions,
            "labels": all_labels,
            "video_ids": video_ids,
            "metrics": eval_metrics
        }

    def train_epoch(
        self,
        epoch,
        metric_logger,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        fold,
        scaler=None,
        cuda_enabled=False,
        log_freq=50, 
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            metric_logger=metric_logger,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            fold=fold,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        metric_logger,
        iters_per_inner_epoch,
        fold,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            metric_logger=metric_logger,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        metric_logger,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        fold,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)


        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        video_list = []
        caption_list = []
        pred_texts_list = []
        gt_texts_list = []
        #for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
        for step in tqdm(range(len(data_loader))):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            logging.info(f"Step {step} of {len(data_loader)}")
            samples = next(data_loader)
            video_list.append(samples['video_id'])
            caption_list.append(samples['answer'])

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": step,
                }
            )
            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=step)

            with torch.amp.autocast(device_type=model.device.type):
                loss, pred_texts, gt_texts = self.train_step(model=model, samples=samples)
  
            pred_texts_list.append(pred_texts)
            gt_texts_list.append(gt_texts)
            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (step + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Print the learning rate for attention parameters
            for param_group in optimizer.param_groups:
                if "attention" in param_group.get("params", []):
                    print("Attention LR:", param_group["lr"])


        # save random samples' name
        save_dir = f"/home/ricke/emotion-llama/rep/checkpoints/run_samples/fold_{fold}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_to = os.path.join(
            save_dir,
            "epoch_{}.txt".format(epoch),
        )
        with open(save_to, 'w') as file:
            file.write("'Video'+ ' ' + 'Label' + ' ' + 'Prediction' + ' ' + 'Caption' + '\n'")
            for i in range(len(video_list)): # steps
                names = video_list[i]
                labels = gt_texts_list[i]
                predictions = pred_texts_list[i]
                captions = caption_list[i]
                for j in range(len(names)): # batch size
                    file.write(names[j] + " " + labels[j] + " " + predictions[j] + " " + captions[j] + '\n')
                
        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.6f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
