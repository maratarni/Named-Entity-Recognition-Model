import os
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
# -----------------------------
# CONFIG
#folosesc modelul deja antrenat
# -----------------------------
model_checkpoint = "dumitrescustefan/bert-base-romanian-cased-v1"
#label_list = ["O", "B-POLITICIAN", "I-POLITICIAN", "B-ECON", "I-ECON", "B-ORG", "I-ORG"]  # adaptează după setul tău
label_list = [
    "O",
    "B-POLITICIAN",
    "I-POLITICIAN",
    "B-ECON_INDICATOR",
    "I-ECON_INDICATOR",
    "B-ECON_TERM",
    "I-ECON_TERM",
    "B-GOV_BODY",
    "I-GOV_BODY",
    "B-ORG",
    "I-ORG"
]

label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

# -----------------------------
# Load raw BIO data
# -----------------------------
def read_conll(file_path):
    sentences = []
    labels = []
    with open(file_path, encoding="utf-8") as f:
        tokens = []
        tags = []
        for line in f:
            if line.strip() == "":
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.strip().split()
                tokens.append(splits[0])
                tags.append(splits[-1])
    return {"tokens": sentences, "ner_tags": labels}

data_files = {
    "train": "data/train.txt",
    "validation": "./data/val.txt",
    "test": "./data/test.txt",
}
raw_datasets = DatasetDict({
    split: Dataset.from_dict(read_conll(path))
    for split, path in data_files.items()
})

# -----------------------------
# Tokenizare + aliniere etichete
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label_to_id[example["ner_tags"][word_idx]])
        else:
            labels.append(label_to_id[example["ner_tags"][word_idx]])
        previous_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = raw_datasets.map(tokenize_and_align_labels)

# -----------------------------
# Model
# -----------------------------
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)

# -----------------------------
# Training args
# -----------------------------
training_args = TrainingArguments(
    output_dir="./models/ner-ro",
    do_eval=True,
    save_steps=500,  # sau cât vrei tu
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    report_to="tensorboard"
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# -----------------------------
# Metrici
# -----------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# -----------------------------
# Training!
# -----------------------------
trainer.train()
results = trainer.evaluate()

# Raport detaliat
predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
    for pred, label in zip(predictions, labels)
]
true_labels = [
    [id_to_label[l] for (p, l) in zip(pred, label) if l != -100]
    for pred, label in zip(predictions, labels)
]

print("Raport pe clase (test set):\n")
print(classification_report(true_labels, true_predictions))

# Afișare organizată
print("\n" + "="*40)
print("REZULTATE EVALUARE MODEL NER")
print("="*40)
for metric, value in results.items():
    print(f"{metric.capitalize():<15}: {value:.4f}")
print("="*40 + "\n")
#cv
