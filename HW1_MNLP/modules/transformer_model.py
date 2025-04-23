import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoConfig)
import evaluate



def build_model(model_type: str, classes: int):
    '''Charge and return the model'''
    cfg = AutoConfig.from_pretrained(
        model_type,
        num_labels=classes,
        output_hidden_states=True,
        output_attentions=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type,
        config=cfg,
        ignore_mismatched_sizes=True
    )
    return model



def metrics(pred):
    load_acc = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    logits, labels = pred
    best_pred = np.argmax(logits, axis=1)
    acc = load_acc.compute(predictions=best_pred, references=labels)["accuracy"]
    f1  = load_f1.compute(predictions=best_pred, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}


def hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
        "weight_decay": trial.suggest_loguniform("weight_decay", 0.0, 0.1),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
    }