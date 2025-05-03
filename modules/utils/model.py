"""
This file contains utility functions for training and evaluating models.

Useful functions:
- configure_wandb_logger
- plot_confusion_matrix

Imports: paths
"""

from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import wandb
from wandb.sdk.wandb_run import Run

from modules import paths


def configure_wandb_logger(project: str, name: str) -> WandbLogger:
    """
    Configure the wandb logger.

    WARNING: This function does not work with multiple processes (njobs > 1).
    """

    # Initialize wandb
    wandb_run: Run = wandb.init(project = project,
                                name = name,
                                dir = paths.DATA_DIR,
                                settings = wandb.Settings(quiet = True, console= 'off'),
                                )

    # Set the metrics
    wandb.define_metric('train_*', summary = 'max', step_metric = 'epoch')
    wandb.define_metric('val_*', summary = 'max', step_metric = 'epoch')
    wandb.define_metric('train_loss', summary = 'min', step_metric = 'epoch')
    wandb.define_metric('val_loss', summary = 'min', step_metric = 'epoch')
    wandb.define_metric('*', step_metric = 'epoch')

    # Return the logger
    return WandbLogger(wandb_run = wandb_run)


def plot_confusion_matrix(true_y: ArrayLike, pred_y: ArrayLike, label_encoder: LabelEncoder|None = None) -> None:
    """
    Plot the confusion matrix.
    """

    # Get the confusion matrix
    confusion: NDArray[np.int_] = confusion_matrix(true_y, pred_y)

    # Plot the confusion matrix
    disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay(confusion,
                                  display_labels = label_encoder.classes_ if label_encoder is not None else None
                                  )
    disp.plot(
        cmap = 'Blues',
        xticks_rotation = 45
    )
    plt.title('Confusion Matrix')

    plt.show()