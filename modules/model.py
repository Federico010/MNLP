"""
Module for the machine learning models.

Useful classes:
- GraphNet
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool, Sequential
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall


class GraphNet(pl.LightningModule):
    """
    Mixed Graph Convolutional Network for classification in 3 classes.
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
                                      nn.BatchNorm1d(inner_dim),
                                      nn.ReLU(),
                                      nn.Dropout()
                                      )

        # GIN blocks
        self.gin_blocks: nn.ModuleList = nn.ModuleList()
        for _ in range(depth):
            self.gin_blocks.append(Sequential('x, edge_index', [
                                              (GINConv(nn.Sequential(nn.Linear(node_features, inner_dim),
                                                                     nn.ReLU(),
                                                                     nn.Linear(inner_dim, inner_dim)
                                                                     )),
                                               'x, edge_index -> x'),
                                               nn.ReLU(),
                                               nn.Dropout()
                                               ]))
            
            # Update the dimensions
            fc_features = inner_dim
            node_features = inner_dim

        # Merge layers
        self.lin: nn.Linear = nn.Linear(inner_dim, n_classes)

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
        for gin_block in self.gin_blocks:
            x_graph = gin_block(x_graph, edge_index)

        # Merge layers
        x_graph = global_mean_pool(x_graph, batch = batch)
        x: torch.Tensor = self.lin(x_fc + x_graph)
        
        return x


    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:

        # Make predictions
        logits: torch.Tensor = self(batch)
        predictions: torch.Tensor = torch.argmax(logits, dim = 1)

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
        predictions: torch.Tensor = torch.argmax(logits, dim = 1)

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


    def configure_optimizers(self) -> Adam:

        return  Adam(self.parameters(),
                     lr = self.hparams.lr  # type: ignore
                     )
