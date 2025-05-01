import os
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# creating dataset for medical domain
splits = {'train': 'train.jsonl', 'validation': 'dev.jsonl', 'test': 'test.jsonl'}

print("Loading RCT+ChemProt datasets from Hugging Face Hub...")

df_train_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["train"], lines=True)
df_train_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["train"], lines=True)

df_validation_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["validation"], lines=True)
df_validation_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["validation"], lines=True)

df_test_rct = pd.read_json("hf://datasets/AdaptLLM/RCT/" + splits["test"], lines=True)
df_test_chem = pd.read_json("hf://datasets/AdaptLLM/ChemProt/" + splits["test"], lines=True)

df_train = pd.concat([df_train_rct.iloc[:,0:2], df_train_chem.iloc[:,0:2]])
# df_train = df_train.rename(columns={'text': 'text', 'label': 'label'})
df_validation = pd.concat([df_validation_rct.iloc[:,0:2], df_validation_chem.iloc[:,0:2]])
# df_validation = df_validation.rename(columns={'text': 'text', 'label': 'label'})
df_test = pd.concat([df_test_rct.iloc[:,0:2], df_test_chem.iloc[:,0:2]])
# df_test = df_test.rename(columns={'text': 'text', 'label': 'label'})



path = '/scratch/rahlab/vedant/adapt/data/med'
df_train.to_json(os.path.join(path, "train.json"), orient='records', index=False)
df_validation.to_json(os.path.join(path, "validation.json"), orient='records', index=False)
df_test.to_json(os.path.join(path, "test.json"), orient='records', index=False)

print("Datasets saved successfully.")

print("Loading Fin phrasebank datasets from Hugging Face Hub...")
# creating dataset for finance domain
df = load_dataset('takala/financial_phrasebank', 'sentences_66agree', split='train').to_pandas()
df = df.rename(columns={'sentence': 'text', 'label': 'label'})

df_train, df_validation = train_test_split(df, test_size=0.3, random_state=42)
df_validation, df_test = train_test_split(df_validation, test_size=0.5, random_state=42)
path = '/scratch/rahlab/vedant/adapt/data/fin'
df_train.to_json(os.path.join(path, "train.json"), orient='records', index=False)
df_validation.to_json(os.path.join(path, "validation.json"), orient='records', index=False)
df_test.to_json(os.path.join(path, "test.json"), orient='records', index=False)

print("Datasets saved successfully.")




