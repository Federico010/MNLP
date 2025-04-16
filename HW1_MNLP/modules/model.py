import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall


class LightningGCN(pl.LightningModule):
    def __init__(self, input_feature, hidden_feature, output_feature, lr=1e-3, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()

        # Define layers
        self.conv1 = GCNConv(input_feature, hidden_feature)
        self.conv2 = GCNConv(hidden_feature, hidden_feature)
        self.fc = nn.Linear(hidden_feature, output_feature)

        # Hyperparameters
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = MulticlassAccuracy(num_classes=output_feature)
        self.val_acc = MulticlassAccuracy(num_classes=output_feature)
        self.val_f1 = MulticlassF1Score(num_classes=output_feature)
        self.val_precision = MulticlassPrecision(num_classes=output_feature)
        self.val_recall = MulticlassRecall(num_classes=output_feature)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x  # No softmax here, as CrossEntropyLoss expects raw logits

    def training_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index)
        loss = self.loss_fn(logits, batch.y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, batch.y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, batch.y)

        acc = self.val_acc(preds, batch.y)
        f1 = self.val_f1(preds, batch.y)
        precision = self.val_precision(preds, batch.y)
        recall = self.val_recall(preds, batch.y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1)
        self.log("val_precision", precision)
        self.log("val_recall", recall)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
