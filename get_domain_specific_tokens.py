import re
import math
import nltk
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import stopwords
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, RobertaTokenizer


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )



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

# def load_and_prepare_data():
#     splits = {'train': 'train.jsonl', 'validation': 'dev.jsonl', 'test': 'test.jsonl'}

#     # Load RCT data
#     df_train_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["train"], lines=True)
#     df_validation_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["validation"], lines=True)
#     df_test_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["test"], lines=True)

#     # Sample RCT: 1000 per label (train), 500 per label (val/test)
#     def sample_per_label(df, n_per_label):
#         return df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), n_per_label), random_state=42)).reset_index(drop=True)

#     df_train_rct = sample_per_label(df_train_rct, 10000)
#     df_validation_rct = sample_per_label(df_validation_rct, 2000)
#     df_test_rct = sample_per_label(df_test_rct, 2000)

#     # Load ChemProt data (full)
#     df_train_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["train"], lines=True)
#     df_validation_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["validation"], lines=True)
#     df_test_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["test"], lines=True)

#     # Combine RCT + ChemProt
#     df_train = pd.concat([df_train_rct.iloc[:, 0:2], df_train_chem.iloc[:, 0:2]], ignore_index=True)
#     df_validation = pd.concat([df_validation_rct.iloc[:, 0:2], df_validation_chem.iloc[:, 0:2]], ignore_index=True)
#     df_test = pd.concat([df_test_rct.iloc[:, 0:2], df_test_chem.iloc[:, 0:2]], ignore_index=True)

#     # Label encoding
#     label_encoder = LabelEncoder()
#     all_labels = pd.concat([df_train['label'], df_validation['label'], df_test['label']], ignore_index=True)
#     label_encoder.fit(all_labels)

#     df_train['label'] = label_encoder.transform(df_train['label'])
#     df_validation['label'] = label_encoder.transform(df_validation['label'])
#     df_test['label'] = label_encoder.transform(df_test['label'])

#     return df_train, df_validation, df_test


def clean_word_tokens(text, vocab_base):
    # tokens = word_tokenize(text)
    # tokens = text.strip().split()  # whitespace-based
    tokens = text.split()
    clean_tokens = [
        tok for tok in tokens
        if re.match(r"^[A-Za-z0-9\-]+$", tok)
        and len(tok) > 2
        and tok.lower() not in stop_words
        and tok.lstrip('Ġ').lower() not in vocab_base
    ]
    return clean_tokens


def get_unigram_counts(text_list, tokenizer):
    vocab_base = set([x.lstrip('Ġ').lower() for x in list(tokenizer.get_vocab().keys())])
    unigram_counts = defaultdict(int)
    for text in tqdm(text_list):
        for word in clean_word_tokens(text,vocab_base):
            unigram_counts[word] += 1
    return unigram_counts

def get_sequence_counts(unigram_counts, tokenizer, max_len=10, min_count=20):
    seq_counts = defaultdict(int)
    for word, count in tqdm(unigram_counts.items()):
        token_ids = tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)
        for i in range(1, min(len(token_ids), max_len) + 1):
            subseq = tuple(token_ids[:i])
            seq_counts[subseq] += count

    # Optional filter: Keep only sequences seen at least min_count times
    seq_counts = {s: c for s, c in seq_counts.items() if c >= min_count}
    return seq_counts

def normalize_distribution(counts_dict, total_unigrams):
    return {k: v / total_unigrams for k, v in counts_dict.items()}

def compute_phrase_scores(T):
    phrase_scores = {}
    for seq in T:
        if len(seq) <= 1:
            continue
        prefix = seq[:-1]
        phrase_scores[seq] = T.get(seq, 0) / (T.get(prefix, 1e-8))
    return phrase_scores

def compute_pointwise_KL(T_base, T_domain):
    score_dkl = {}
    all_keys = set(T_domain.keys()).union(set(T_base.keys()))
    for seq in all_keys:
        p = T_domain.get(seq, 0)
        q = T_base.get(seq, 1e-8)  # smooth base
        if p > 0:
            score_dkl[seq] = p * math.log(p / q)
    return score_dkl

def select_augmentations(score_dkl, T_domain_raw, T_base_raw, max_len=10, fmin=20, top_n=10000):
    augmentations = []
    sorted_seqs = sorted(score_dkl.items(), key=lambda x: -x[1])
    for seq, score in sorted_seqs:
        if len(augmentations) >= top_n:
            break
        if len(seq) <= max_len and T_domain_raw.get(seq, 0) >= fmin and T_base_raw.get(seq, 0) >= fmin:
            augmentations.append(seq)
    
    return augmentations


def adaptive_tokenization(base_corpus, domain_corpus, tokenizer_base):
    # Requirement, get unigram distribution
    print("\nGetting unigram counts...")
    U_base = get_unigram_counts(base_corpus, tokenizer_base)     # S in the paper
    U_domain = get_unigram_counts(domain_corpus, tokenizer_base) # D in the paper

    # Step 1: Get sequence counts
    print("\nGetting sequence counts...")
    T_base = get_sequence_counts(U_base, tokenizer_base)
    T_domain = get_sequence_counts(U_domain, tokenizer_base)

    # Step 2: Normalize Distribution
    print("\nNormalizing distributions...")
    T_base_norm = normalize_distribution(T_base, sum(U_base.values()))
    T_domain_norm = normalize_distribution(T_domain, sum(U_domain.values()))

    #Step 3: Compute Phrase scores
    print("\nComputing phrase scores...")
    P_base = compute_phrase_scores(T_base_norm)
    P_domain = compute_phrase_scores(T_domain_norm)

    # Step 4: Compute KL divergence
    print("\nComputing KL divergence...")
    score_dkl = compute_pointwise_KL(P_base, P_domain)

    # Step 5: Select sequences for augmentation
    print("\nSelecting augmentations...")
    augmentations_T = select_augmentations(
        score_dkl,
        T_domain_raw=T_domain,  # before normalization
        T_base_raw=T_base,
        max_len=10,
        fmin=20,
        top_n=10000
    )

    # Convert back to token strings for tokenizer extension
    new_tokens = [tokenizer_base.decode(seq)for seq in augmentations_T]

    print("\nNumber of new tokens:", len(new_tokens))
    print("New tokens:", new_tokens)

    return new_tokens






def main():
    df_train, df_validation, df_test = load_and_prepare_data()
    print('\nTrain:', len(df_train), 'Validation:', len(df_validation), 'Test:', len(df_test))

    print('\n')
    tokenizer_base = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    # model_base = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=len(df_train['label'].unique()))

    print('\ndomain adaptive tokenization')
    domain_corpus = df_train['text'].tolist()[:50000]
    # base_corpus = [x['text'] for x in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)]
    # Load the dataset with a streaming approach to avoid loading everything into memory
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    base_corpus = [x['text'] for _, x in zip(range(184209), dataset)]
    new_tokens = adaptive_tokenization(base_corpus, domain_corpus, tokenizer_base)

    # Write each word on a new line
    with open("new_tokens1.txt", "w") as file:
        for token in new_tokens:
            file.write(token + "\n")

if __name__ == "__main__":
    main()