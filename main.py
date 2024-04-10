#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:26:12 2024

@author: danieldabbah
"""
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP
import numpy as np
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import pandas as pd
import pre_process


cols_to_exclude = ['Unnamed: 0', 'OBJECTID', 'FOD_ID', 'FPA_ID',
                   'LOCAL_FIRE_REPORT_ID', 'LOCAL_INCIDENT_ID',
                   'ICS_209_INCIDENT_NUMBER', 'ICS_209_NAME', 'MTBS_ID',
                   'MTBS_FIRE_NAME', 'COMPLEX_NAME', 'OWNER_CODE', 'Shape']

cols_to_use = ['SOURCE_SYSTEM_TYPE', 'DISCOVERY_DATE',
               'DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_DESCR',
               'CONT_DATE', 'CONT_DOY', 'CONT_TIME',   'STATE', ]


cols_to_check_value_counts = ['SOURCE_SYSTEM', 'NWCG_REPORTING_AGENCY',
                              'NWCG_REPORTING_UNIT_ID',
                              'NWCG_REPORTING_UNIT_NAME', 'SOURCE_REPORTING_UNIT',
                              'SOURCE_REPORTING_UNIT_NAME', 'FIRE_CODE',
                              'FIRE_NAME', 'OWNER_DESCR', 'COUNTY', 'FIPS_CODE', 'FIPS_NAME',
                              'LATITUDE', 'LONGITUDE', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'FIRE_YEAR'
                              ]

if __name__ == '__main__':
    k = 100000
    df = pd.read_csv("data/train.csv.gz", usecols=cols_to_use)[:9*k]
    validation = pd.read_csv("data/validation.csv.gz",
                             usecols=cols_to_use)[:3*k]
    test = pd.read_csv("data/test_1.csv.gz", usecols=cols_to_use)[:k]

    b = df[:10]
    df = pre_process.pre_process_time_cols(df)
    a = df[:500]

    df.columns

    df["text"][0]
    validation = pre_process.pre_process_time_cols(validation)
    test = pre_process.pre_process_time_cols(test)
    # TODO: later improve the model by add text features from cols_to_check_values_counts
    k = test[:200]
    # create hugging face data sets
    # start tokenize like they did in the book


# len(cols_to_check_value_counts) + len(cols_to_exclude) + len(cols_to_use)
    len(test['label'].value_counts())
    len(validation['label'].value_counts())
    len(df['label'].value_counts())

    from datasets import Dataset, Features, ClassLabel, Value

    unique_classes = sorted(df['label'].unique())

    unique_classes
    features = Features({
        'label': ClassLabel(names=unique_classes),
        'text': Value('string')
        # Include other columns as needed, e.g., 'text': Value('string')
    })

    features
    train_ds = Dataset.from_pandas(df, features=features)
    val_ds = Dataset.from_pandas(validation, features=features)
    test_ds = Dataset.from_pandas(test, features=features)

    from datasets import DatasetDict

    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })


dataset_dict
text = "Tokenizing text is a core task of NLP."


model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

encoded_text = tokenizer(text)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


dataset_encoded = dataset_dict.map(tokenize, batched=True, batch_size=None)

print(dataset_encoded["train"].column_names)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


device = get_device()
model = AutoModel.from_pretrained(model_ckpt).to(device)


def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k: v.to(device) for k, v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


dataset_encoded.set_format("torch",
                           columns=["input_ids", "attention_mask", "label"])

dataset_hidden = dataset_encoded.map(extract_hidden_states, batched=True)

dataset_hidden["train"].column_names

dataset_dict

X_train = np.array(dataset_hidden["train"]["hidden_state"])
X_valid = np.array(dataset_hidden["validation"]["hidden_state"])
y_train = np.array(dataset_hidden["train"]["label"])
y_valid = np.array(dataset_hidden["validation"]["label"])
X_train.shape, X_valid.shape


# Scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
# Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
df_emb.head()


fig, axes = plt.subplots(3, 5, figsize=(7, 5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]*2+['Greys']
labels = dataset_dict["train"].features["label"].names

for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                   gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])

plt.tight_layout()
plt.show()


# hide_output
# We increase `max_iter` to guarantee convergence

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)


print(lr_clf.score(X_valid, y_valid))
