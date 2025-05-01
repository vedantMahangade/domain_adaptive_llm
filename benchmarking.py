import os
import csv
import time
import torch
import sklearn
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import Any, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer

os.environ["TQDM_DISABLE"] = "1"

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


def load_and_prepare_data(path):
    splits = {'train': 'train.json', 'validation': 'validation.json', 'test': 'test.json'}

    df_train = pd.read_json(os.path.join(path, splits["train"]), orient="records")
    df_validation = pd.read_json(os.path.join(path, splits["validation"]), orient="records")
    df_test = pd.read_json(os.path.join(path, splits["test"]), orient="records")

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


    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
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
        tokenizer=tokenizer,
    )

    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()

    eval_start_time = time.time()
    results = trainer.evaluate(test_dataset)
    eval_end_time = time.time()

    return results, train_end_time - train_start_time, eval_end_time - eval_start_time
  

def adapt_model_and_tokenizer(new_tokens, tokenizer_base, tokenizer_adapt, model_base, model_adapt):
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

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Apdaptive Tokenization Benchmarking')
    parser.add_argument('--model_name', type=str, help='Model name.', default='FacebookAI/roberta-base')
    parser.add_argument('--adapt', type=str, help='Path to the txt file containing new token. Default=None, means no adaptation', default=None)
    parser.add_argument('--dataset_path', type=str, help='Path to dataset.', default='hf://datasets/AdaptLLM/ChemProt/')
    parser.add_argument('--output_path', type=str, help='Folder for outputs', default='output')
    parser.add_argument('--run_name', type=str, help='Path to dataset.', default='output')
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    start_time = time.time()
    df_train, df_validation, df_test = load_and_prepare_data(args.path)
    print('Train:', len(df_train), 'Validation:', len(df_validation), 'Test:', len(df_test))

    print("\nLoading Base models...")
    tokenizer_base = RobertaTokenizer.from_pretrained(args.model_name)
    model_base = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(df_train['label'].unique()))


    model_save_path = os.path.join(args.output_path, args.run_name)

    if args.adapt is None:
        print("\tInitiating training for base model...")
        results, train_time, eval_time = run_training(tokenizer_base, model_base, df_train, df_validation, df_test, model_save_path)
        print("Base Results:", results)
        print("Training Time - Base:", train_time)
        print("Evaluation Time - Base:", eval_time, "\n")

    else:
        print('\nDomain adaptive Tokenization')
        new_tokens = pd.read_csv(args.adapt, header=None)[0].tolist()
        print('New tokens read:', len(new_tokens))

        print("\nLoading New models...")
        tokenizer_adapt = RobertaTokenizer.from_pretrained(args.model_name)
        model_adapt = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(df_train['label'].unique()))

        tokenizer_adapt, model_adapt = adapt_model_and_tokenizer(new_tokens, tokenizer_base, tokenizer_adapt, model_base, model_adapt)
        print("\nInitiating training for adaptive tokenization model...")
        results, train_time, eval_time = run_training(tokenizer_adapt, model_adapt, df_train, df_validation, df_test,  model_save_path)
        print("Adapt Results:", results)
        print("Training Times - Adapt:", train_time)
        print("Evaluation Time - Adapt:", eval_time, "\n")

    # Append results to CSV
    log_file = os.path.join(args.output_path, "log.csv")
    file_exists = os.path.exists(log_file)
    with open(log_file, mode="a") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Run Name", "Loss", "Accuracy", "F1", "MCC", "Precision", "Recall", "Training Time", "Inference Time"])
        writer.writerow([args.run_name, results["eval_loss"], results["eval_accuracy"], results["eval_f1"],
                         results["eval_matthews_correlation"], results["eval_precision"], results["eval_recall"],
                         train_time, eval_time])

