# -*- coding: utf-8 -*-
"""reddit_cleaning_large.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dkYni_K_l0F2xpyZ8OtibdQzgpEkghIm

## Imports ##
"""

import nltk
from nltk.corpus import words
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

"""## Data Preparation ##"""

nltk.download("words")

data = pd.read_csv("kaggle_RC_2019-05.csv")
data.head()

# remove controversial statements
data = data[data["controversiality"] != 1]
#data = data[["body"]]
print(data.head())

corpus = set(words.words())

def clean_text(string):
  list_ = string.split(" ")
  # convert all to lower case except special case: "I"
  lower_cases = [word.lower() for word in list_ if word != "I"]
  # only include words that appear in the english language
  valid_words = [word for word in lower_cases if word in corpus]
  return valid_words

data['body'] = data['body'].apply(lambda x: clean_text(x))
print(data.head())

all_dataset_words = []
def compile_all_words(word_list):
  all_dataset_words.extend(word_list)
  return word_list

data['body'] = data['body'].apply(compile_all_words)

all_unique_words = set(all_dataset_words)

num_unique_words = len(all_unique_words)
num_unique_words

# create mapping of words to their numbers and vice versa
string_to_int = {s: i for i, s in enumerate(all_unique_words)}
int_to_string = {i: s for s, i in string_to_int.items()}

# build training data and labels

block_size = 3  # trigram model
X, y = [], []

for index, row in data.iterrows():
  word_list = row['body']
  for i in range(len(word_list) - block_size + 1):
    context = word_list[i:i + block_size - 1]  # context (2 words for trigram)
    target = word_list[i + block_size - 1]  # target (3rd word in trigram)

    # Convert context and target to integer representation
    context_indices = [string_to_int[word] for word in context]
    target_index = string_to_int[target]

    X.append(context_indices)
    y.append(target_index)

X = torch.tensor(X)
y = torch.tensor(y)

X.shape, y.shape

# create embedding tensor -> 90 unique words will be embedded each into 10 values
C = torch.randn((num_unique_words, 10))

C.shape

# lookup of embeddings for every context tensor in X
emb = C[X]
emb.shape

"""## Split into Training and Testing ##"""

n_samples = len(X)
reduced_size = n_samples // 3  # Reduce to 1/3
indices = torch.randperm(n_samples)[:reduced_size]
X_reduced = X[indices]
y_reduced = y[indices]


X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y_reduced, test_size=0.2, random_state=42
)

print(f"Training data: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data: X_test={X_test.shape}, y_test={y_test.shape}")