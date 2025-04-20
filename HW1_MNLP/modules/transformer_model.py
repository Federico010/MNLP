import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoConfig)
import evaluate


#def build_model(model_type: str, classes: int):
    #'''Charge and return the model'''
    #model = AutoModelForSequenceClassification.from_pretrained(
        #model_type, 
        #num_labels=classes,
        #ignore_mismatched_sizes=True,
        #output_hidden_states=True,
        #output_attentions=True,
    #)
    
    
    #return model

def build_model(model_type: str, classes: int):
    cfg = AutoConfig.from_pretrained(model_type, num_labels=classes)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_type, config=cfg, ignore_mismatched_sizes=True
    )
    # wrap del forward per stampare le shape
    orig_forward = model.forward
    def debug_forward(*args, input_ids=None, attention_mask=None, labels=None, **kwargs):
        print("\n>>> 🕵️ INSIDE MODEL FORWARD")
        print("    input_ids.shape:     ", None if input_ids is None else input_ids.shape)
        print("    attention_mask.shape:", None if attention_mask is None else attention_mask.shape)
        print("    labels.shape:        ", None if labels is None else labels.shape)
        out = orig_forward(*args,
                           input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels,
                           **kwargs)
        print("    logits.shape:        ", out.logits.shape)
        return out
    model.forward = debug_forward.__get__(model, model.__class__)
    return model


def metrics(pred):
    load_acc = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    logits, labels = pred
    best_pred = np.argmax(logits, axis=1)
    acc = load_acc.compute(predictions=best_pred, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=best_pred, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1}


