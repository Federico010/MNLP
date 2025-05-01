"""
Module for the machine learning models creation.

Useful functions:
- transformer_metrics

Useful classes:
- GraphNet
"""

from typing import Any, Literal

import lightning.pytorch as pl
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool, LayerNorm, Sequential
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from transformers.trainer_utils import EvalPrediction


def transformer_metrics(prediction_item: EvalPrediction) -> dict[str, float]:
    """
    Calculate the metrics for the transformer model. To be used in the trainer.
    """

    # Find the predictions and labels
    logits: NDArray[np.float32] = np.array(prediction_item.predictions)
    predictions: NDArray[np.intp] = np.argmax(logits, axis = 1)
    labels: NDArray[np.int_] = np.array(prediction_item.label_ids)

    # Calculate the metrics
    f1: float = float(f1_score(labels, predictions, average = 'macro'))
    accuracy: float = float(accuracy_score(labels, predictions))
    precision: float = float(precision_score(labels, predictions, average = 'macro'))
    recall: float = float(recall_score(labels, predictions, average = 'macro'))

    return {'f1': f1, 'accuracy': accuracy, 'precision': precision, 'recall': recall}


class GraphNet(pl.LightningModule):
    """
    Mixed Graph Convolutional Network for classification.
    """

    def __init__(self, fc_features: int, node_features: int, n_classes: int, depth: int, inner_dim: int, lr: float = 1e-3) -> None:
        """
        Initialize the model.
        
        Args:
            fc_features: Number of features for the fully connected layers.
            node_features: Number of features for each node of the graph.
            n_classes: Number of classes.
            depth: Number oflayers.
            inner_dim: Dimension of the layers.
            lr: Learning rate.
        """

        super().__init__()
        self.save_hyperparameters({
            'fc_features': fc_features,
            'node_features': node_features,
            'n_classes': n_classes,
            'depth': depth,
            'inner_dim': inner_dim,
            'lr': lr
        })

        # Fully connected block
        self.fc_block = nn.Sequential(nn.Linear(fc_features, inner_dim),
                                      nn.LayerNorm(inner_dim),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )
        
        # Transform the node features
        self.node_transform: nn.Linear = nn.Linear(node_features, inner_dim)

        # GIN blocks
        self.gin_blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(depth):
            self.gin_blocks.append(Sequential('x, edge_index', [
                                              (GINConv(nn.Sequential(nn.Linear(inner_dim, inner_dim),
                                                                     nn.ReLU(),
                                                                     nn.Linear(inner_dim, inner_dim)
                                                                     )),
                                               'x, edge_index -> x'),
                                               LayerNorm(inner_dim),
                                               nn.ReLU(),
                                               nn.Dropout()
                                               ]))

        # Merge layers
        self.lin: nn.Linear = nn.Linear(inner_dim, n_classes)

        # Loss
        self.loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy: MulticlassAccuracy = MulticlassAccuracy(n_classes)
        self.f1: MulticlassF1Score = MulticlassF1Score(n_classes)
        self.precision: MulticlassPrecision = MulticlassPrecision(n_classes)
        self.recall: MulticlassRecall = MulticlassRecall(n_classes)


    def forward(self, data_batch: Batch) -> torch.Tensor:

        # Unpack the data
        x_fc: torch.Tensor = data_batch.x_fc  # type: ignore
        x_graph: torch.Tensor = data_batch.x_graph  # type: ignore
        edge_index: torch.Tensor = data_batch.edge_index    # type: ignore
        batch: torch.Tensor = data_batch.batch  # type: ignore

        # Reshape due to geometric dataloader flattening the batch
        x_fc: torch.Tensor = x_fc.view(-1, self.hparams.fc_features)    # type: ignore

        # Fully connected layer
        x_fc = self.fc_block(x_fc)

        # Graph layers
        x_graph = self.node_transform(x_graph)
        for gin_block in self.gin_blocks:
            x_graph = gin_block(x_graph, edge_index) + x_graph

        # Merge layers
        x_graph = global_mean_pool(x_graph, batch = batch)
        x: torch.Tensor = self.lin(x_fc + x_graph)
        
        return x
    

    def _make_step(self, step: Literal["train", "val"], batch: Batch, batch_idx: int) -> torch.Tensor:
        """
        Make a step of the model.
        """

        # Make predictions
        logits: torch.Tensor = self(batch)
        predictions: torch.Tensor = torch.argmax(logits, dim = 1)

        # Evaluate the model
        labels: torch.Tensor = batch.y  # type: ignore

        loss: torch.Tensor = self.loss(logits, labels)
        accuracy: torch.Tensor = self.accuracy(predictions, labels)
        f1: torch.Tensor = self.f1(predictions, labels)
        precision: torch.Tensor = self.precision(predictions, labels)
        recall: torch.Tensor = self.recall(predictions, labels)

        # Log the metrics
        batch_size: int = labels.shape[0]
        self.log(f"{step}_loss", loss, prog_bar = True, batch_size = batch_size)
        self.log(f"{step}_accuracy", accuracy, prog_bar = True, batch_size = batch_size)
        self.log(f"{step}_f1", f1, prog_bar = True, batch_size = batch_size)
        self.log(f"{step}_precision", precision, prog_bar = True, batch_size = batch_size)
        self.log(f"{step}_recall", recall, prog_bar = True, batch_size = batch_size)

        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._make_step("train", batch, batch_idx)

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._make_step("val", batch, batch_idx)


    def configure_optimizers(self) -> dict[str, Adam|dict[str, Any]]:   # type: ignore

        # Optimizer
        optimizer: Adam = Adam(self.parameters(), lr = self.hparams.lr) # type: ignore

        # Scheduler
        scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, patience = 0, factor = 0.5)

        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    

    def on_train_epoch_end(self) -> None:

        # Log the lr
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr", lr)
