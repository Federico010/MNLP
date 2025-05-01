"""
This file contains utility functions for training and evaluating models.

Useful functions:
- configure_wandb_logger
- rename_best_checkpoint
- plot_confusion_matrix

Imports: paths
"""

from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import wandb
from wandb.sdk.wandb_run import Run

from modules import paths


def configure_wandb_logger(project: str, name: str, config: dict[str, Any]) -> WandbLogger:
    """
    Configure the wandb logger.

    WARNING: This function does not work with multiple processes (njobs > 1).
    """

    # Initialize wandb
    wandb_run: Run = wandb.init(project = project,
                                name = name,
                                config = config,
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


def rename_best_checkpoint(checkpoint: ModelCheckpoint, trial_number: int) -> Path:
    """
    Rename the best checkpoint file to include the trial number. This change is NOT reflected in the best_model_path attribute of the checkpoint.
    """

    checkpoint_path: Path = Path(checkpoint.best_model_path)
    new_path: Path = checkpoint_path.parent / f'{checkpoint_path.stem}_trial={trial_number}{checkpoint_path.suffix}'
    checkpoint_path.rename(new_path)
    return new_path


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