# -*- coding: utf-8 -*-
"""
### Imports ###
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# check if required packages are installed and if not, install them
try:
  import torchmetrics, mlxtend
  print(f"mlxtend version: {mlxtend.__version__}")
  assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19.0 or higher"
except:
  !pip install -q torchmetrics -U mlxtend
  import torchmetrics, mlxtend
  print(f"mlxtend version: {mlxtend.__version__}")

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import spacy
nlp = spacy.load("en_core_web_sm")

import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo
if Path("helper_functions.py").is_file(): # if the file already exists, don't import again
  print("helper_functions.py already exists, skipping download...")
else:
  print("Downloading helper_functions.py")
  request = requests.get(url="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  # note that you need to get the raw version from GitHub
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)


from helper_functions import plot_predictions, plot_decision_boundary

"""### Prepare Functions ###"""

device = "cuda" if torch.cuda.is_available() else "cpu"
device

def print_train_time(start:float, end:float, device:torch.device = None):
  total_time = end - start
  print(f"Train time on {device}: {total_time:.3f} seconds")
  return total_time

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
  device = next(model.parameters()).device

  pred_probs = []
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim = 0).to(device)
      pred_logit = model(sample)
      pred_prob = torch.softmax(pred_logit.squeeze(), dim = 1).argmax(dim = 1)
      # Get the pred_probs off the GPU for further calculations
      pred_probs.append(pred_prob.cpu())

  return torch.stack(pred_probs) # concatenate into a single tensor

def eval_model(model: torch.nn.Module, test_data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for X, y in test_data_loader:
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss += loss_fn(y_pred, y)

      # accumulate the loss and acc value per batch
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim = 1))

    # scale loss and acc to find the average loss/acc per batch
    loss /= len(test_data_loader)
    acc /= len(test_data_loader)

  return {"model_name": model.__class__.__name__, # this only works when model was created with a class
          "model_loss": loss.item(), # .item() will extract a single value
          "model_acc": acc}

def preprocess_text(text: str) -> str:
    text = text.lower()
    doc = nlp(text)
    new_text = [token.text for token in doc if not token.is_stop]

    return " ".join(new_text)

def test_eval_loop(model, epochs, train_dataloader, val_dataloader, loss_function, optimizer):
  for epoch in tqdm(range(epochs)):
    # Training loop
    model.train()
    train_loss, correct_train_predictions, total_train_samples = 0.0, 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
      batch_size = X.size(0)
      y_logits = model(X)
      y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
      loss = loss_function(y_logits, y)

      train_loss += loss.item()
      correct_train_predictions += (y_pred == y).sum().item()
      total_train_samples += batch_size

      optimizer.zero_grad()
      loss.backward()

      # Clip gradients to avoid excessively large updates
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

      optimizer.step()

      # Validation loop
      val_loss, correct_val_predictions, total_val_samples = 0.0, 0, 0
      model.eval()
      with torch.no_grad():
        for batch, (X_val, y_val) in enumerate(val_dataloader):
          val_logits = model(X_val)
          val_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)
          val_loss += loss_function(val_logits, y_val).item()
          correct_val_predictions += (val_pred == y_val).sum().item()
          total_val_samples += X_val.size(0)

    # Print at the end of each epoch
    avg_train_loss = train_loss / total_train_samples
    avg_train_acc = correct_train_predictions * 100 / total_train_samples
    avg_val_loss = val_loss / total_val_samples
    avg_val_acc = correct_val_predictions * 100 / total_val_samples

    if epoch % 10 == 0:
      print(f"Epoch: {epoch}\n---------")
      print(f"\nTrain loss: {avg_train_loss:.4f} Train acc: {avg_train_acc:.2f}% | Val loss: {avg_val_loss:.4f}, Val acc: {avg_val_acc:.2f}%\n")

"""### Build the Neural Network ###"""

class NeuralNet(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Linear(input_size, 400),
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, output_size)
    )

  def forward(self, x):
    return self.layer_stack(x)

"""### For Smaller Data ###"""

# Load and preprocess training data
train_data = pd.read_excel("File1.xlsx")
train_data.rename(columns={"Note": "x_unchanged", "Final": "y"}, inplace=True)
train_data["x"] = train_data["x_unchanged"].apply(preprocess_text)
train_data = train_data[['x', 'y']]
train_data['y'] = train_data['y'].apply(lambda x: int(x - 1))
print(f"Train data shape: {train_data.shape}")

"""### Import and Prepare Data ###"""

# Load and preprocess evaluation data
eval_data = pd.read_excel("File2.xlsx")
eval_data.rename(columns={"Note": "x_unchanged", "Final": "y"}, inplace=True)
eval_data["x"] = eval_data["x_unchanged"].apply(preprocess_text)
eval_data = eval_data[['x', 'y']]
eval_data['y'] = eval_data['y'].apply(lambda x: int(x - 1))
print(f"Eval data shape: {eval_data.shape}")

"""### Using AI Generated Data ###"""

train_data = pd.read_excel("new_train_data.xlsx")
train_data = train_data.dropna()
train_data = train_data[["text", "label"]]
train_data.rename(columns={"label": "y"}, inplace=True)

import re
def remove_special_characters(text):
    # Define the regex pattern to keep only alphabets (both lowercase and uppercase)
    pattern = re.compile(r'[^a-zA-Z\s]')
    # Substitute the matches with an empty string
    return pattern.sub('', text)

train_data['x'] = train_data['text'].apply(remove_special_characters)
#train_data['y'] = train_data['label'].apply(lambda x: int(x + 1))

cat_one = train_data[train_data['y'] == 0]
cat_one = cat_one.sample(n = 300, random_state = 44)
cat_two = train_data[train_data['y'] == 1]
cat_two = cat_two.sample(n = 300, random_state = 44)

cat_three = train_data[train_data['y'] == 2]
cat_three = cat_three.sample(n = 300, random_state = 44)

cat_four = train_data[train_data['y'] == 3]
cat_four = cat_four.sample(n = 300, random_state = 44)

sample_data = pd.concat([cat_one, cat_two, cat_three, cat_four], axis = 'rows')

sample_data = sample_data[['x', 'y']]

train_data.shape

sample_indices = sample_data.index
test_data = train_data[~train_data.index.isin(sample_indices)]
print(test_data.shape)

"""### Vectorize and Create Tensors ###"""

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_data["x"]).toarray()
X_test = vectorizer.transform(test_data["x"]).toarray()

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_data["y"])
y_test = test_data["y"].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=44)

print(f"Shape of X: {X_train.shape}")
print(f"First row of X:\n{X_train[0].shape}")
print(f"Shape of y: {y_train.shape}")
print(f"First element of y: {y_train[0].shape}")

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of first row of X_train: {X_train[0].shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"First element of y_train: {y_train[0].shape}")

"""### Declare Global Variables for Training and Testing ###"""

input_size = X_train.shape[1]
print(input_size)
output_size = 4
#print(label_encoder.classes_)
batch_size = 8

# Create DataLoader objects
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

"""### Train Model ###"""

torch.manual_seed(21)
torch.cuda.manual_seed(21)

model = NeuralNet(input_size, output_size)
model.to(device)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

torch.manual_seed(21)
torch.cuda.manual_seed(21)

epochs = 30
start_time = timer()

test_eval_loop(model, epochs, train_dataloader, val_dataloader, loss_fn, optimizer)

end_time = timer()

total_time = print_train_time(start_time, end_time, device)
print(total_time)

"""### Plot Confusion Matrix ###"""

from tqdm.auto import tqdm
start_time = timer()
y_preds = []
model.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc = "Making predictions..."):
    X, y = X.to(device), y.to(device)
    y_logit = model(X)
    y_pred = torch.softmax(y_logit, dim = 1).argmax(dim = 1)
    y_preds.append(y_pred.cpu())

end_time = timer()
print(print_train_time(start_time, end_time, device))
  # Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)
print(y_pred_tensor[:10])
print(len(y_pred_tensor))

# Check the number of unique classes in y_pred_tensor
num_classes_pred = len(y_pred_tensor.unique()) # should be 4
print("Number of unique classes in predictions:", num_classes_pred)

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Adjust num_classes based on the predictions
confmat = ConfusionMatrix(task = 'multiclass', num_classes = num_classes_pred)
confmat_tensor = confmat(preds = y_pred_tensor,
                         target = torch.tensor(test_data.y.values))

confmat_tensor

fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(),
    class_names = ['Name1', 'Name2', 'Name3, 'Name4'],
    figsize = (10, 7)
)

from sklearn.metrics import classification_report

target_names = ['Name1', 'Name2', 'Name3, 'Name4']
print(classification_report(test_data.y.values, y_pred_tensor.numpy(), target_names = target_names))
