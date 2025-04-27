import numpy as np
from transformers import (AutoModelForSequenceClassification)
from sklearn.metrics import f1_score, accuracy_score


def build_model(model_type: str, classes: int):
    '''Charge and return the model'''
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        num_labels=classes,
        ignore_mismatched_sizes=True
    )
    return model

def metrics(pred):
    logits, labels = pred
    predictions = logits.argmax(-1)
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"f1": f1, "accuracy": acc}

