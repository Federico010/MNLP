import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torchmetrics import Accuracy, F1Score, Precision, Recall

class LightningGCN(pl.LightningModule):
    def __init__(self, input_feature, hidden_feature, output_feature, lr=1e-3, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = GCNConv(input_feature, hidden_feature)
        self.conv2 = GCNConv(hidden_feature, hidden_feature)
        self.fc = nn.Linear(hidden_feature, output_feature)
        
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

        self.acc = Accuracy(task="multiclass", num_classes=output_feature)
        self.f1 = F1Score(task="multiclass", num_classes=output_feature)
        self.precision = Precision(task="multiclass", num_classes=output_feature)
        self.recall = Recall(task="multiclass", num_classes=output_feature)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x  # No softmax here!

    def training_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index)
        loss = self.loss_fn(logits, batch.y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch.x, batch.edge_index)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, batch.y)

        acc = self.acc(preds, batch.y)
        f1 = self.f1(preds, batch.y)
        precision = self.precision(preds, batch.y)
        recall = self.recall(preds, batch.y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1)
        self.log("val_precision", precision)
        self.log("val_recall", recall)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
