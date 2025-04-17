import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    DataCollatorWithPadding, set_seed
)
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0)
model_type = "xml-roberta-base"
batch_size = 16
epochs = 1
learning_rate = 1e-4
weight_decay = 0.01
classes= 3

##TODO take description from wikipedia and...ecc 

df_train = pd.read_csv("updated_train.csv")
df_val = pd.read_csv("updated_val.csv")

all_labels = list(df_train["label"].unique()) #list of labels
lab_to_number = {}
number_to_lab = {}
for i, lab in enumerate(sorted(all_labels)): # create 2 dict lab -> number and number -> lab 
    lab_to_number[lab] = i 
    number_to_lab[i] = lab

df_train["label"] = df_train["label"].map(lab_to_number) # converts the labels[agnostic, exclusive, representative] to numbers
df_val["label"] = df_val["label"].map(lab_to_number)
train_df = df_train[["text", "label"]]  # take only the Wikipedia description and label columns
val_df = df_val[["text", "label"]]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def metrics(pred):
    load_acc = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    logits, labels = pred
    best_pred = np.argmax(logits, axis=1)
    acc = load_acc.compute(predictions=best_pred, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=best_pred, references=labels)["f1"]
    return {"accuracy": acc, "f1": f1}


model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=classes, ignore_mismatched_size =True, output_hidden_states = False, output_attentions = False).to(device)
tokenizer = AutoTokenizer.from_pretrained(model)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

train_dataset = train_dataset.map(tokenize, batched=True) #tokenize dataset
val_dataset = val_dataset.map(tokenize, batched=True)

collator = DataCollatorWithPadding(tokenizer=tokenizer) #applied a zero padding on the elements in batch 

training_arg = TrainingArguments(
    output_dir=".results",
    train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    logging_dir="./logs",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_arg,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=metrics
)

trainer.train() 
trainer.evaluate()