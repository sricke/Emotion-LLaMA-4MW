"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import torch
import torch.distributed as dist
import webdataset as wds
from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from minigpt4.common.logger import MetricLogger, SmoothedValue, init_mlflow, end_mlflow_run


@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None
        self.best_epoch = None
        self.best_val_loss = None
        self.start_epoch = 0

        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu], find_unused_parameters=True
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            attention = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                print(n)
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()

            logging.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                    "lr": float(self.config.run_cfg.init_lr)
                },
                {"params": p_non_wd, "weight_decay": 0, "lr": float(self.config.run_cfg.init_lr)},
            ]

            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders['train'])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            batch_sizes = {dataset_name: self.config.datasets_cfg.hcmw_caption.batch_size
                           for dataset_name in self.datasets.keys()}
            datasets, batch_sizes = reorg_datasets_by_split(self.datasets, batch_sizes)
            self.datasets = datasets
            # self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            batch_sizes = [batch_sizes[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            print("batch sizes", batch_sizes)

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):

        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        # output_dir = lib_root / self.config.run_cfg.output_dir
        result_dir = output_dir  / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self, fold, experiment_name):
        start_time = time.time()
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)
        self.initialize_logger()
        self.initialize_mlflow(experiment_name=experiment_name+f"_fold_{fold}", tags={"fold": fold})
        output_path = self.log_config()
        self.log_config_to_mlflow(output_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            self.set_mlflow_step(cur_epoch)
            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch, fold)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_results = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch, fold=fold
                    )
                    
                    val_log = val_results['metrics']
  
                    if val_log is not None:
                        if is_main_process():
                            
                            assert (
                                "val_loss" in val_log
                            ), "No val_loss found in validation log."
 
                            val_loss = float(val_log["val_loss"])
                            if val_loss < self.best_val_loss and split_name == "eval":
                                self.best_epoch, self.best_val_loss = cur_epoch, val_loss
                                logging.info(f"New best validation loss {self.best_val_loss} at epoch {self.best_epoch}")
                                self._save_checkpoint(cur_epoch, fold, is_best=True)
                            #val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)
                            self.log_metrics_to_mlflow(step=cur_epoch, global_avg=True)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)
    
            if self.evaluate_only:
                break

            if self.config.run_cfg.distributed:
                dist.barrier()
            self.logger.reset_meters()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        results = self.evaluate(cur_epoch=test_epoch, skip_reload=False, fold=fold)
        for split_name, result in results.items():
            logging.info(f"Best Evaluation results on {split_name} - fold {fold}")
            logging.info(json.dumps(result, indent=4))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))
        
        self.end_mlflow()
        self.cleanup_model()
        
    def cleanup_model(self):
        """Remove model from GPU and free memory."""
        if self._wrapped_model is not None:
            if self.use_distributed:
                # Unwrap DDP model
                self._wrapped_model = self._wrapped_model.module
            self._wrapped_model = self._wrapped_model.cpu()
            del self._wrapped_model
            self._wrapped_model = None
        
        if self._model is not None:
            self._model = self._model.cpu()
            del self._model
            self._model = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    @main_process
    def save_prediction_results(self, video_ids, predictions, labels, split_name, fold):
        output_dir = os.path.join(self.output_dir, f'fold_{fold}')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{split_name}_prediction_{self.best_epoch}_{self.best_val_loss}.txt")
        with open(save_path, "w") as f:
            f.write(f"video_id\tprediction\tlabel\n")
            for i in range(len(predictions)):
                f.write(f"{video_ids[i]}\t{predictions[i]}\t{labels[i]}\n")
                
        self.logger.log_artifact_to_mlflow(save_path)
         
    @main_process
    def save_metrics_results(self, metrics, split_name, fold):
        output_dir = os.path.join(self.output_dir, f'fold_{fold}')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{split_name}_metrics_{self.best_epoch}_{self.best_val_loss}.txt")
        with open(save_path, "w") as f:
            f.write(f"metric\tvalue\n")
            for metric, value in metrics.items():
                f.write(f"{metric}\t{value}\n")
                
        self.logger.log_artifact_to_mlflow(save_path)
    
        
    def evaluate(self, cur_epoch="best", skip_reload=False, fold=None):
        results = dict()
        if len(self.valid_splits) > 0:
            for split_name in self.valid_splits:
                split_results = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload, fold=fold
                )
                results[split_name] = split_results
                self.save_prediction_results(split_results['video_ids'], split_results['predictions'], split_results['labels'], split_name, fold)
                self.save_metrics_results(split_results['metrics'], split_name, fold)
            return results

    def train_epoch(self, epoch, fold):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            fold=fold,
            model=self.model,
            metric_logger=self.logger,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, fold, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            logging.info("Reloading best model")
            model = self._reload_best_model(model, fold)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader, metric_logger=self.logger)
        return results
        """if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )"""

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split

            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together

                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
     
                loader = PrefetchLoader(loader)
    
                #if is_train:
                loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]

                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i]) for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)
        return loaders


    @main_process
    def _save_checkpoint(self, cur_epoch, fold, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """

        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        output_dir = os.path.join(self.output_dir, f'fold_{fold}')
        os.makedirs(output_dir, exist_ok=True)
        save_to = os.path.join(
            output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model, fold):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, f'fold_{fold}', "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        message = self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        print("resume the checkpoint")
        logging.info("Resume checkpoint from {}".format(url_or_filename))
    
    
    def initialize_logger(self):
        self.logger = MetricLogger(delimiter="  ")
        self.logger.add_meter("train_loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("train_accuracy", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("train_f1_score", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("train_precision", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("train_recall", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("train_support", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("train_auc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("train_balanced_accuracy", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        self.logger.add_meter("mw_train_recall", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        
    @main_process
    def initialize_mlflow(self, experiment_name, tags):
        init_mlflow(
            experiment_name=experiment_name,
            tags=tags,
        )
        
    @main_process
    def end_mlflow(self):
        end_mlflow_run()
        
    @main_process
    def set_mlflow_step(self, step):
        self.logger.set_mlflow_step(step)
        
    @main_process
    def log_metrics_to_mlflow(self, step, global_avg=False):
        self.logger.log_metrics_to_mlflow(step=step, log_global_avg=global_avg)

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        output_path = os.path.join(self.output_dir, "log.txt")
        with open(output_path, "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
        return output_path
            
    @main_process
    def log_config_to_mlflow(self, output_path):
        self.logger.log_artifact_to_mlflow(output_path)