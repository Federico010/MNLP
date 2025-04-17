"""
Module for the machine learning models.

Useful classes:
- GCN
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
#from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall


class GCN(pl.LightningModule):
    """
    Graph Convolutional Network (GCN) for node classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters({'lr': lr})

        # Layers
        self.conv1: GCNConv = GCNConv(input_dim, hidden_dim)
        self.conv2: GCNConv = GCNConv(hidden_dim, hidden_dim)
        self.fc: nn.Linear = nn.Linear(hidden_dim, output_dim)

        # Loss function
        self.loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()


    def forward(self, x, edge_index, batch) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch=batch)
        x = self.fc(x)
        return x


    def training_step(self, batch, batch_idx) -> torch.Tensor:

        logits: torch.Tensor = self(batch.x, batch.edge_index, batch.batch)
        loss: torch.Tensor = self.loss(logits, batch.y)

        self.log("train_loss", loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx) -> torch.Tensor:

        logits: torch.Tensor = self(batch.x, batch.edge_index, batch.batch)
        loss: torch.Tensor = self.loss(logits, batch.y)

        self.log("val_loss", loss, prog_bar=True)

        return loss


    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
