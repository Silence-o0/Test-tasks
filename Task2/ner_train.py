from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer
from datasets import load_from_disk
from evaluate import load
import numpy as np


# Function for tokenizing dataset
def tokenize_data(dataset, tokenizer):
    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        all_labels = []
        for index, ner_tags in enumerate(examples["ner_tags"]):
            word_id_list = tokenized.word_ids(batch_index=index)
            prev_word_idx = None
            aligned_labels = []
            for word_id in word_id_list:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != prev_word_idx:
                    aligned_labels.append(ner_tags[word_id])
                else:
                    aligned_labels.append(-100)
                prev_word_idx = word_id
            all_labels.append(aligned_labels)

        tokenized["labels"] = all_labels
        return tokenized

    return dataset.map(tokenize_and_align_labels, batched=True)


# Function for computing evaluation metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []
    for label_seq, pred_seq in zip(labels, predictions):
        true_seq = []
        pred_seq_filtered = []
        for label, pred in zip(label_seq, pred_seq):
            if label != -100:
                true_seq.append(id2label[label])
                pred_seq_filtered.append(id2label[pred])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_filtered)
    results = metric.compute(predictions=pred_labels, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }


# Definition unique entity labels
unique_entities = ["O", "B-animal", "I-animal"]
id2label = {i: label for i, label in enumerate(unique_entities)}
label2id = {label: i for i, label in enumerate(unique_entities)}

# Loading preprocessed datasets
loaded_train_ds = load_from_disk("data_ner/train_dataset")
loaded_valid_ds = load_from_disk("data_ner/valid_dataset")
loaded_test_ds = load_from_disk("data_ner/test_dataset")

# Loading tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(unique_entities),
                                                        id2label=id2label, label2id=label2id)

# Definition training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
)

# Tokenizing datasets
train_ds = tokenize_data(loaded_train_ds, tokenizer)
valid_ds = tokenize_data(loaded_valid_ds, tokenizer)
test_ds = tokenize_data(loaded_test_ds, tokenizer)

# Loading evaluation metric
metric = load("seqeval")

# Initializing Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Model training and saving
trainer.train()
trainer.save_model("ner_model")

# Evaluate model on test dataset and print results
test_results = trainer.evaluate(test_ds)
print("Test Dataset Evaluation:")
for key, value in test_results.items():
    print(f"{key}: {value}")
