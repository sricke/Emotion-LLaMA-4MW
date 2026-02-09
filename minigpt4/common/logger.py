"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from minigpt4.common import dist_utils

import mlflow


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.window_size = window_size
        self.deque = deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not dist_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", use_mlflow=False, mlflow_step=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.use_mlflow = use_mlflow 
        self.mlflow_step = mlflow_step  # Current step for MLflow logging
        self._mlflow_logged_metrics = set()  # Track which metrics have been logged
        
    def set_mlflow_step(self, step):
        """Set the current step for MLflow logging."""
        self.mlflow_step = step
        
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            #assert isinstance(v, (float, int))
            self.meters[k].update(v)
            

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)
    
    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.global_avg))
        return self.delimiter.join(loss_str)
    
    def log_metric_to_mlflow(self, metric_name, step=None):
        if step is not None:
            mlflow.log_metric(metric_name, self.meters[metric_name], step=step)
        else:
            mlflow.log_metric(metric_name, self.meters[metric_name])
    
    def log_metrics_to_mlflow(self, prefix="", step=None, log_global_avg=True):
        """
        Log all current metrics to MLflow.
        
        Args:
            prefix: Optional prefix to add to metric names (e.g., "train_", "val_")
            step: Optional step number (uses self.mlflow_step if not provided)
            log_global_avg: If True, logs global_avg; if False, logs current value
        """
            
        step = step if step is not None else self.mlflow_step
        
        metrics_dict = {}
        for name, meter in self.meters.items():
            metric_name = f"{prefix}{name}" if prefix else name
            if log_global_avg:
                metrics_dict[metric_name] = meter.global_avg
            else:
                metrics_dict[metric_name] = meter.value
      
        if metrics_dict:
            for metric_name, metric_value in metrics_dict.items():
                if step is not None:
                    mlflow.log_metric(metric_name, metric_value, step=step)
                else:
                    mlflow.log_metric(metric_name, metric_value)
                
    def log_artifact_to_mlflow(self, artifact_path):
        try:
            mlflow.log_artifact(artifact_path)
        except Exception as e:
            logging.warning(f"Failed to log artifact to MLflow: {e}")

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # Log to MLflow at print frequency
                if self.use_mlflow and dist_utils.is_main_process():
                    self.set_mlflow_step(i)
                    self.log_metrics_to_mlflow(step=i, log_global_avg=True)
                
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def setup_logger():
    logging.basicConfig(
        level=logging.INFO if dist_utils.is_main_process() else logging.WARN,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def init_mlflow(experiment_name=None, run_name=None, tracking_uri=None, tags=None):
    """
    Initialize MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        run_name: Name for the current run
        tracking_uri: MLflow tracking URI (default: uses MLflow default)
        tags: Optional dict of tags to add to the run
        
    Returns:
        bool: True if MLflow was successfully initialized, False otherwise
    """
        
    if not dist_utils.is_main_process():
        # Only initialize on main process
        return False
        
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        if experiment_name:
            # Get or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logging.info(f"Created MLflow experiment: {experiment_name} (ID: {experiment_id})")
                else:
                    mlflow.set_experiment(experiment_name)
                    logging.info(f"Using existing MLflow experiment: {experiment_name}")
            except Exception as e:
                logging.warning(f"Failed to set MLflow experiment: {e}")
        
        # Start a new run
        if run_name:
            mlflow.start_run(run_name=run_name)
        else:
            mlflow.start_run()
            
        if tags:
            mlflow.set_tags(tags)
            
        logging.info("MLflow tracking initialized successfully")
        return True
    except Exception as e:
        logging.warning(f"Failed to initialize MLflow: {e}")
        return False


def end_mlflow_run():
    """End the current MLflow run."""
    if dist_utils.is_main_process():
        try:
            mlflow.end_run()
            logging.info("MLflow run ended")
        except Exception as e:
            logging.warning(f"Failed to end MLflow run: {e}")
