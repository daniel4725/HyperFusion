
import numpy as np
import time
import os
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from datetime import datetime

def CheckpointCallback4regression(**kwargs):
    now = datetime.now()
    dt_string = now.strftime("%dd%mm%yy-%Hh%Mm%Ss")
    # print("date and time =", dt_string)

    checkpoint_dir = kwargs["checkpoint_dir"]
    experiment_name = kwargs["experiment_name"]

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, experiment_name, f"fold_{kwargs['data_fold']}"),
        # dirpath=os.path.join(checkpoint_dir, experiment_name +"_" + dt_string, f"fold_{kwargs['data_fold']}"),
        filename='best_val',
        # filename=experiment_name + '-epoch={epoch}-val_balanced_acc={val/balanced_acc:.3f}',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        monitor="val/MAE",
        mode="min"
        )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last"
    # checkpoint_callback.CHECKPOINT_NAME_LAST = experiment_name + "-epoch={epoch}-last"
    return checkpoint_callback


def CheckpointCallback(**kwargs):
    now = datetime.now()
    dt_string = now.strftime("%dd%mm%yy-%Hh%Mm%Ss")
    # print("date and time =", dt_string)

    checkpoint_dir = kwargs["checkpoint_dir"]
    experiment_name = kwargs["experiment_name"]

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, experiment_name, f"fold_{kwargs['data_fold']}"),
        # dirpath=os.path.join(checkpoint_dir, experiment_name +"_" + dt_string, f"fold_{kwargs['data_fold']}"),
        filename='best_val',
        # filename=experiment_name + '-epoch={epoch}-val_balanced_acc={val/balanced_acc:.3f}',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        monitor="val/balanced_acc",
        mode="max"
        )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last"
    # checkpoint_callback.CHECKPOINT_NAME_LAST = experiment_name + "-epoch={epoch}-last"
    return checkpoint_callback


class TimeEstimatorCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_of_epochs = kwargs["epochs"]
        self.epochs_left = kwargs["epochs"]
        self.train_epoch_end = False
        self.val_epoch_end = False
        self.time_format = "%Hh %Mm %Ss"
        self.time_per_epoch_queue = []
        self.time_per_epoch_queue_size = 10

    def on_train_start(self, trainer, pl_module):
        self.start_train = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_epoch = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_epoch_end = True
        if self.train_epoch_end:
            self.print_timing()

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_epoch_end = True
        if self.val_epoch_end:
            self.print_timing()

    def print_timing(self):
        self.train_epoch_end = False
        self.val_epoch_end = False

        # TODO mean over window of epochs and not all of them from the beginning of time

        self.epochs_left -= 1
        self.time_per_epoch_queue.append(time.time() - self.start_epoch)
        if len(self.time_per_epoch_queue) > self.time_per_epoch_queue_size:
            self.time_per_epoch_queue.pop(0)
        
        epoch_mean_time = np.mean(self.time_per_epoch_queue)
        time_from_start = time.time() - self.start_train
        estimated_time_remain = epoch_mean_time * self.epochs_left

        # epoch_time = show_time(epoch_time)
        epoch_mean_time = show_time(epoch_mean_time)
        time_from_start = show_time(time_from_start)
        estimated_time_remain = show_time(estimated_time_remain)

        # epoch_time = time.strftime(self.time_format, time.gmtime(epoch_time))
        # time_from_start = time.strftime(self.time_format, time.gmtime(time_from_start))
        # estimated_time_remain = time.strftime(self.time_format, time.gmtime(estimated_time_remain))
        
        print(f"\n\nmean epoch time: {epoch_mean_time}")
        print(f'time from start: {time_from_start}')
        print(f'estimated time remain: {estimated_time_remain}\n')

def show_time(seconds):
    time = int(seconds)
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    if day != 0:
        return "%dD %dH %dM %dS" % (day, hour, minutes, seconds)
    elif day == 0 and hour != 0:
        return "%dH %dM %dS" % (hour, minutes, seconds)
    elif day == 0 and hour == 0 and minutes != 0:
        return "%dM %dS" % (minutes, seconds)
    else:
        return "%dS" % (seconds)
