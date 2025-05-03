"""
This file contains utility functions for training and evaluating models.

Useful functions:
- configure_wandb
- plot_confusion_matrix

Imports: paths
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import wandb
from wandb.sdk.wandb_run import Run

from modules import paths


def configure_wandb(project: str, name: str, library: Literal['lightning', 'transformers']) -> Run:
    """
    Configure wandb.

    WARNING: This function does not work with multiple processes (njobs > 1).

    Args:
        project: name of the project.
        name: name of the run.
        library: library that is used.
    """

    # Initialize wandb
    wandb_run: Run = wandb.init(project = project,
                                name = name,
                                dir = paths.DATA_DIR,
                                settings = wandb.Settings(quiet = True, console= 'off'),
                                )

    # Set the metrics
    if library == 'lightning':
        wandb.define_metric('train_*', summary = 'max', step_metric = 'epoch')
        wandb.define_metric('val_*', summary = 'max', step_metric = 'epoch')
        wandb.define_metric('train_loss', summary = 'min', step_metric = 'epoch')
        wandb.define_metric('val_loss', summary = 'min', step_metric = 'epoch')
        wandb.define_metric('*', step_metric = 'epoch')
    else:
        wandb.define_metric('eval/accuracy', summary = 'max')
        wandb.define_metric('eval/precision', summary = 'max')
        wandb.define_metric('eval/recall', summary = 'max')
        wandb.define_metric('eval/f1', summary = 'max')
        wandb.define_metric('train/loss', summary = 'min')
        wandb.define_metric('eval/loss', summary = 'min')

    # Return the logger
    return wandb_run


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
