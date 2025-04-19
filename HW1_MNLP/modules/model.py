"""
Module for the machine learning models.

Useful classes:
- GCN
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall


class GCN(pl.LightningModule):
    """
    Graph Convolutional Network (GCN) for graph classification in 3 classes.
    """

    def __init__(self, hidden_dim: int, lr: float = 1e-3, dropout: float = 0.5) -> None:
        """
        Initialize the GCN model.
        """

        super().__init__()
        self.save_hyperparameters({
            'hidden_dim': hidden_dim,
            'lr': lr,
            'dropout': dropout
        })

        # Layers
        self.conv1: GCNConv = GCNConv(2, hidden_dim)
        self.norm1: GraphNorm = GraphNorm(hidden_dim)
        self.conv2: GCNConv = GCNConv(hidden_dim, hidden_dim)
        self.norm2: GraphNorm = GraphNorm(hidden_dim)
        self.fc: nn.Linear = nn.Linear(hidden_dim, 3)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Loss
        self.train_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.val_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

        # Metrics
        self.train_accuracy: MulticlassAccuracy = MulticlassAccuracy(3)
        self.val_accuracy: MulticlassAccuracy = MulticlassAccuracy(3)
        self.train_f1: MulticlassF1Score = MulticlassF1Score(3)
        self.val_f1: MulticlassF1Score = MulticlassF1Score(3)
        self.train_precision: MulticlassPrecision = MulticlassPrecision(3)
        self.val_precision: MulticlassPrecision = MulticlassPrecision(3)
        self.train_recall: MulticlassRecall = MulticlassRecall(3)
        self.val_recall: MulticlassRecall = MulticlassRecall(3)


    def forward(self, data_batch: Batch) -> torch.Tensor:

        x: torch.Tensor = data_batch.x  # type: ignore
        edge_index: torch.Tensor = data_batch.edge_index    # type: ignore
        batch: torch.Tensor = data_batch.batch  # type: ignore

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch = batch)
        x = self.dropout(x)

        x = self.fc(x)
        
        return x


    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:

        # Make predictions
        logits: torch.Tensor = self(batch)
        predictions: torch.Tensor = torch.argmax(logits, dim=1)

        # Evaluate the model
        labels: torch.Tensor = batch.y  # type: ignore

        loss: torch.Tensor = self.train_loss(logits, labels)
        accuracy: torch.Tensor = self.train_accuracy(predictions, labels)
        f1: torch.Tensor = self.train_f1(predictions, labels)
        precision: torch.Tensor = self.train_precision(predictions, labels)
        recall: torch.Tensor = self.train_recall(predictions, labels)

        # Log the metrics
        batch_size: int = labels.shape[0]
        self.log("train_loss", loss, prog_bar=True, batch_size = batch_size)
        self.log("train_accuracy", accuracy, prog_bar=True, batch_size = batch_size)
        self.log("train_f1", f1, prog_bar=True, batch_size = batch_size)
        self.log("train_precision", precision, prog_bar=True, batch_size = batch_size)
        self.log("train_recall", recall, prog_bar=True, batch_size = batch_size)

        return loss


    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:

        # Make predictions
        logits: torch.Tensor = self(batch)
        predictions: torch.Tensor = torch.argmax(logits, dim=1)

        # Evaluate the model
        labels: torch.Tensor = batch.y  # type: ignore

        loss: torch.Tensor = self.val_loss(logits, labels)
        accuracy: torch.Tensor = self.val_accuracy(predictions, labels)
        f1: torch.Tensor = self.val_f1(predictions, labels)
        precision: torch.Tensor = self.val_precision(predictions, labels)
        recall: torch.Tensor = self.val_recall(predictions, labels)

        # Log the metrics
        batch_size: int = labels.shape[0]
        self.log("val_loss", loss, prog_bar=True, batch_size = batch_size)
        self.log("val_accuracy", accuracy, prog_bar=True, batch_size = batch_size)
        self.log("val_f1", f1, prog_bar=True, batch_size = batch_size)
        self.log("val_precision", precision, prog_bar=True, batch_size = batch_size)
        self.log("val_recall", recall, prog_bar=True, batch_size = batch_size)

        return loss


    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(),
                                lr = self.hparams.lr  # type: ignore
                                )
