import math
import json
import glob
import time
import torch
import sklearn
import numpy as np
import pandas as pd
from math import log
import seaborn as sns
from tqdm import tqdm
from datasets import Dataset
import matplotlib.pyplot as plt
from typing import Any, Tuple, Union
from transformers import get_scheduler, AdamW
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]

    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    if predictions.ndim > 1:  # assume logits or softmax
        predictions = np.argmax(predictions, axis=-1)
    return calculate_metric_with_sklearn(predictions, labels)


def load_and_prepare_data():
    splits = {'train': 'train.jsonl', 'validation': 'dev.jsonl', 'test': 'test.jsonl'}

    # df_train = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["train"], lines=True).iloc[:, 0:2]
    # df_validation = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["validation"], lines=True).iloc[:, 0:2]
    # df_test = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["test"], lines=True).iloc[:, 0:2]


    df_train_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["train"], lines=True)
    df_train_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["train"], lines=True)

    df_validation_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["validation"], lines=True)
    df_validation_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["validation"], lines=True)

    df_test_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["test"], lines=True)
    df_test_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["test"], lines=True)

    df_train = pd.concat([df_train_rct.iloc[:,0:2], df_train_chem.iloc[:,0:2]])
    df_validation = pd.concat([df_validation_rct.iloc[:,0:2], df_validation_chem.iloc[:,0:2]])
    df_test = pd.concat([df_test_rct.iloc[:,0:2], df_test_chem.iloc[:,0:2]])

    label_encoder = LabelEncoder()
    all_labels = pd.concat([df_train['label'], df_validation['label'], df_test['label']])
    label_encoder.fit(all_labels)

    df_train.loc[:, 'label'] = label_encoder.transform(df_train['label'])
    df_validation.loc[:, 'label'] = label_encoder.transform(df_validation['label'])
    df_test.loc[:, 'label'] = label_encoder.transform(df_test['label'])

    return df_train, df_validation, df_test

def run_training(tokenizer, model, df_train, df_validation, df_test, output_dir):
    train_dataset = Dataset.from_pandas(df_train).map(lambda x: preprocess_function(x, tokenizer), batched=True)
    val_dataset = Dataset.from_pandas(df_validation).map(lambda x: preprocess_function(x, tokenizer), batched=True)
    test_dataset = Dataset.from_pandas(df_test).map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # Compute total training steps
    per_device_train_batch_size = 16
    num_train_epochs = 5
    train_batch_size = per_device_train_batch_size * max(1, torch.cuda.device_count())
    total_training_steps = math.ceil(len(train_dataset) / train_batch_size) * num_train_epochs


    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    lr_scheduler = get_scheduler(
        name="cosine",  # or "linear"
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),
        num_training_steps=total_training_steps,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        logging_steps=200,
        report_to="none"
    )

    trainer = Trainer(
        model=model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(optimizer, lr_scheduler),
        tokenizer=tokenizer,
    )

    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()

    eval_start_time = time.time()
    results = trainer.evaluate(test_dataset)
    eval_end_time = time.time()

    return results, train_end_time - train_start_time, eval_end_time - eval_start_time
  

def initialize_adapt_model(new_tokens, tokenizer_base, tokenizer_adapt, model_base, model_adapt):
    tokenizer_adapt.add_tokens(new_tokens)
    added = []
    skipped = []
    for tok in new_tokens:
      if tok in tokenizer_adapt.get_vocab():
          added.append(tok)
      else:
          skipped.append(tok)

    print("Added", len(added), "tokens")
    print("Skipped", len(skipped), "tokens")
    # force add
    if len(skipped) > 0:
        tokenizer_adapt.add_tokens(skipped)
    model_adapt.resize_token_embeddings(len(tokenizer_adapt))

    with torch.no_grad():
        for token in new_tokens:
            # Get the subword embeddings from the base model
            subwords = tokenizer_base.tokenize(token)
            subword_ids = tokenizer_base.convert_tokens_to_ids(subwords)
            subword_embeds = model_base.roberta.embeddings.word_embeddings.weight[subword_ids]
            # Compute the mean embedding for the new token
            mean_embed = torch.mean(subword_embeds, dim=0)
            # Assign the mean embedding to the new token in the adapted model
            token_id = tokenizer_adapt.convert_tokens_to_ids(token)
            model_adapt.roberta.embeddings.word_embeddings.weight[token_id] = mean_embed

    return tokenizer_adapt, model_adapt

def extract_eval_f1(state, metric):
    return [(log["epoch"], log[metric]) for log in state["log_history"] if metric in log]

def plot_metric(metric, base_state, adapt_state):

    # Set a consistent color palette
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2", 2)  # 2 distinct colors

    epochs_base, base_score = zip(*extract_eval_f1(base_state, metric))
    epochs_adapt, adapt_score = zip(*extract_eval_f1(adapt_state, metric))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_base, base_score, label="Base", color=colors[0], marker='o')
    plt.plot(epochs_adapt, adapt_score, label="Adapt", color=colors[1], marker='o')
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(metric+" vs. epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/"+metric+"_vs_epoch.jpg", dpi=300)

def visualize_results(results_base, results_adapt, train_time_base, train_time_adapt, eval_time_base, eval_time_adapt):
    base_path = glob.glob('/content/small_with_out_adapt.json')
    adapt_path = glob.glob('/content/small_with_adapt.json')

    # Load both JSON files
    with open(base_path[0]) as f:
        base_state = json.load(f)

    with open(adapt_path[0]) as f:
        adapt_state = json.load(f)

    # Set a consistent color palette
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("Set2", 2)  # 2 distinct colors

    plot_metric("eval_loss", base_state, adapt_state)
    plot_metric("eval_f1", base_state, adapt_state)
    plot_metric("eval_accuracy", base_state, adapt_state)
    plot_metric("eval_precision", base_state, adapt_state)
    plot_metric("eval_recall", base_state, adapt_state)

    plt.figure(figsize=(6, 4))
    plt.bar(["Base", "Adapt"], [results_base['eval_f1'], results_adapt['eval_f1']], color=colors)
    plt.ylabel("F1")
    plt.title("Adapt F1 vs. Base F1")
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/adapt_f1_vs_base_f1.jpg", dpi=300)

    plt.figure(figsize=(6, 4))
    plt.bar(["Base", "Adapt"], [train_time_base, train_time_adapt], color=colors)
    plt.ylabel("Training Time (s)")
    plt.title("Training Time Comparison")
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/training_time_comparison.jpg", dpi=300)

    plt.figure(figsize=(6, 4))
    plt.bar(["Base", "Adapt"], [eval_time_base, eval_time_adapt], color=colors)
    plt.ylabel("Evaluation Time (s)")
    plt.title("Evaluation Time Comparison")
    plt.tight_layout()
    plt.show()
    plt.savefig("plots/evaluation_time_comparison.jpg", dpi=300)



def main():
    start_time = time.time()
    df_train, df_validation, df_test = load_and_prepare_data()
    print('Train:', len(df_train), 'Validation:', len(df_validation), 'Test:', len(df_test))

    print("\nLoading Base models...")
    # Base model training
    # tokenizer_base = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    tokenizer_base = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    model_base = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=len(df_train['label'].unique()))
    print("\tInitiating training for base model...")
    results_base, train_time_base, eval_time_base = run_training(tokenizer_base, model_base, df_train, df_validation, df_test, "/scratch/rahlab/vedant/adapt/small_with_out_adapt")
    print("Base Results:", results_base)
    print("Training Time - Base:", train_time_base)
    print("Evaluation Time - Base:", eval_time_base, "\n")

    # Adapted model training
    # tokenizer_adapt = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    print("\nLoading New models...")
    tokenizer_adapt = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    model_adapt = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=len(df_train['label'].unique()))

    print('\nDomain adaptive Tokenization')
    new_tokens = pd.read_csv('new_tokens/new_tokens.txt', header=None)[0].tolist()
    print('New tokens read:', len(new_tokens))
    print("\nInitiating training for adaptive tokenization model...")
    tokenizer_adapt, model_adapt = initialize_adapt_model(new_tokens, tokenizer_base, tokenizer_adapt, model_base, model_adapt)

    results_adapt, train_time_adapt, eval_time_adapt = run_training(tokenizer_adapt, model_adapt, df_train, df_validation, df_test, "/scratch/rahlab/vedant/adapt/small_with_adapt")
    print("Adapt Results:", results_adapt)
    print("Training Times - Adapt:", train_time_adapt)
    print("Evaluation Time - Adapt:", eval_time_adapt, "\n")

    # Visualize results
    print("\nVisualizing results...")
    visualize_results(results_base, results_adapt, train_time_base, train_time_adapt, eval_time_base, eval_time_adapt)
    print("Results visualized successfully!")
    end_time = time.time()
    print("Total Time Taken:", end_time - start_time, "seconds")
if __name__ == "__main__":
    main()
