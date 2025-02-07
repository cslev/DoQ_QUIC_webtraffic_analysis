#!/usr/bin/python3
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

## inspired by VIT then include some libraries
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import pandas as pd
import numpy as np
import random
from numpy import load


from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Prepare arguements
parser = argparse.ArgumentParser(description="Hyperparameters for tranformer model")
parser.add_argument("--num_classes", dest="num_classes", default=499, type=int,
                    help="number of websites to be classified")
parser.add_argument("--feature_size", dest="feature_size", default=200, type=int,
                    help="numbers of packets to keep as features")
parser.add_argument("--embedding_size", dest="embedding_size", default=64, type=int,
                    help="size of feature embedding")
parser.add_argument("--word_size", dest="word_size", default=5, type=int,
                    help="size of word embedding for each packet")
parser.add_argument("--num_heads", dest="num_heads", default=32, type=int,
                    help="numbers of attention heads in a transformer encoder")
parser.add_argument("--num_encoders", dest="num_encoders", default=2, type=int,
                    help="numbers of transformer encoders")
parser.add_argument("--dense_layer_1", dest="dense_layer_1", default=1024, type=int,
                    help="first dense layer size")
parser.add_argument("--dense_layer_2", dest="dense_layer_2", default=512, type=int,
                    help="numbers of packets to keep as features")
parser.add_argument("--npz_file_path", dest="npz_file_path", default="../../dataset/dataset_500_200.npz", type=str,
                    help="the data location")
parser.add_argument('--open_world', dest="open_world", action='store_true', default=False,
                    help='indicator of running close-world or open-world experiment')
args = parser.parse_args()


# Model Hyperparameter
open_world = args.open_world
parm_num_classes = args.num_classes + 1 + int(open_world)
parm_num_ow_classes = 4500

parm_feature_size = args.feature_size
parm_emb_size = args.embedding_size
parm_word_size = args.word_size
parm_num_heads = args.num_heads
parm_num_encoders = args.num_encoders
parm_dense_layer_1 = args.dense_layer_1
parm_dense_layer_2 = args.dense_layer_2
N_EPOCHS = 120
LR = 0.01

print("-----Hyperparameters-----")
print(f"parm_num_classes: {parm_num_classes}")
print(f"parm_feature_size: {parm_feature_size}")
print(f"parm_emb_size: {parm_emb_size}")
print(f"parm_word_size: {parm_word_size}")
print(f"parm_num_heads: {parm_num_heads}")
print(f"parm_num_encoders: {parm_num_encoders}")
print(f"parm_dense_layer_1: {parm_dense_layer_1}")
print(f"parm_dense_layer_2: {parm_dense_layer_2}")
print(f"N_EPOCHS: {N_EPOCHS}")
print(f"LR: {LR}")
print("----------")

# Load dataset
npz_file_path = args.npz_file_path
dict_data = load(npz_file_path)
x_train, y_train, x_test, y_test = dict_data["x_train"], dict_data["y_train"], dict_data["x_test"], dict_data["y_test"]

print(f"dataset: {npz_file_path}")
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = torch.tensor(x_train).to(torch.float)
y_train = torch.tensor(y_train).to(torch.float)
x_test = torch.tensor(x_test).to(torch.float)
y_test = torch.tensor(y_test).to(torch.float)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Load transformer
# first conv2d to improve the performance
class NetPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, feature_size: int = parm_feature_size, emb_size: int = parm_emb_size, word_size: int = parm_word_size, ):
        self.feature_size = feature_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a linear one to map the input to emb_size
            nn.Linear(word_size, emb_size)
        )
        # sla_token: which aggrgate sequence level information
        self.sla_token = nn.Parameter(torch.randn(1,1,emb_size))

        # positional embedding : we let the model learn it, positional embedding is just a tensor of shape N_patchs
        self.positions = nn.Parameter(torch.randn(feature_size + 1, emb_size) )

    def forward(self, x: Tensor) -> Tensor:
        # print(x.shape)
        b, _, _, = x.shape

        x = self.projection(x)
        # repeat sla_token to all the samples number b
        sla_tokens = repeat(self.sla_token, '() n e -> b n e', b = b)

        # then prepend the sla token to the input
        x = torch.cat([sla_tokens, x], dim = 1)
        # add positional embedding
        x += self.positions

        return x

## self-attention mechanism

class NetMultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = parm_emb_size, num_heads: int = parm_num_heads, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        ## calculate the Query key and value
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        # the result query key and values has a shape of BATCH, HEADS, SEQUENCE_LEN, EMBEDDING_SIZE
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # sum up over the last axis   batch, num_heads, query_len, key_len
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)


        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


#add residual value
class NetResidualAdd(nn.Module):
    def __init__(self, fn):
      super().__init__()
      self.fn = fn

    def forward(self, x, **kwargs):
      res = x
      x = self.fn(x, **kwargs)
      x += res
      return x


class NetFeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p : float = 0.):
      super().__init__(
          nn.Linear(emb_size, expansion * emb_size),
          nn.GELU(),
          nn.Dropout(),
          nn.Linear(expansion*emb_size, emb_size)
      )

class NetTransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 8,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            NetResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                NetMultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            NetResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                NetFeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

## adding multiple encoders if needed
class NetTransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = parm_num_encoders, **kwargs):
        super().__init__(*[NetTransformerEncoderBlock(**kwargs) for _ in range(depth)])

## classification head for classification
class NetClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = parm_emb_size, n_classes: int = parm_num_classes, ):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            # add one layer for feature extraction
            nn.Linear(emb_size, parm_dense_layer_1),
            nn.Linear(parm_dense_layer_1, parm_dense_layer_2),
            nn.Linear(parm_dense_layer_2, n_classes)
            )
##
class NetFormer(nn.Sequential):
    def __init__(self,
                in_channels: int = 1,
                feature_size: int = parm_feature_size,
                emb_size: int = parm_emb_size,
                depth: int = parm_num_encoders,
                n_classes: int = parm_num_classes,
                **kwargs):
        super().__init__(
            NetPatchEmbedding(in_channels, feature_size, emb_size,),
            NetTransformerEncoder(depth, emb_size=emb_size, **kwargs),
            NetClassificationHead(emb_size, n_classes)
        )
summary(NetFormer(), (parm_feature_size, parm_word_size), device='cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
model = NetFormer().to(device)

# Train model
acc_train_set = []
acc_test_set = []

loss_train_set = []
loss_test_set = []

## train the model

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in trange(N_EPOCHS, desc="Training"):
  start_time = time.time()
  train_loss = 0.0
  correct, total = 0, 0
  TP, FP, TN, FN, WP = 0, 0, 0, 0, 0
  true_postive, wrong_positive, false_positive = 0, 0, 0
  for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} in training", leave=True,  mininterval=50, miniters=50):
    x, y = batch
    y = y.type(torch.LongTensor)
    x, y = x.to(device), y.to(device)
    y_hat = model(x)
    loss = criterion(y_hat, y)
    train_loss += loss.detach().cpu().item() / len(train_dataloader)
    correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
    total += len(x)

    if open_world:
        predicted_labels = torch.argmax(y_hat, dim=1).detach().cpu()
        correct_labels = y.detach().cpu()
        unmonitored_class = parm_num_classes - 1
        TP += ((predicted_labels == correct_labels) & (predicted_labels != unmonitored_class)).sum(axis=0)
        FP += ((predicted_labels != unmonitored_class) & (correct_labels == unmonitored_class)).sum(axis=0)
        TN += ((predicted_labels == correct_labels) & (correct_labels == unmonitored_class)).sum(axis=0)
        FN += ((predicted_labels == unmonitored_class) & (correct_labels != unmonitored_class)).sum(axis=0)
        WP += ((predicted_labels != correct_labels) & (predicted_labels != unmonitored_class) & (correct_labels != unmonitored_class)).sum(axis=0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  if epoch == 80:
    scheduler = StepLR(optimizer, step_size=5, gamma=0.75)
  scheduler.step()
  print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"Epoch training time is: {elapsed_time}s")

  print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
  print(f"Epoch {epoch + 1}/{N_EPOCHS} Training accuracy: {correct / total * 100:.2f}%")
  if open_world:
    print(TP, FP, TN, FN, WP)
    print(f"Epoch {epoch + 1}/{N_EPOCHS} Training TPR: {TP / (TP + WP + FN) * 100:.2f}%")
    print(f"Epoch {epoch + 1}/{N_EPOCHS} Training WPR: {WP / (TP + WP + FN) * 100:.2f}%")
    print(f"Epoch {epoch + 1}/{N_EPOCHS} Training FPR: {FP / (FP + TN) * 100:.2f}%")
  acc_train_set.append(correct / total)
  loss_train_set.append(train_loss)
  # Test loop
  with torch.no_grad():
    correct, total = 0, 0
    test_loss = 0.0
    TP, FP, TN, FN, WP = 0, 0, 0, 0, 0
    for batch in tqdm(test_dataloader, desc="Testing", leave=True, mininterval=5, miniters=10):
        x, y = batch
        y = y.type(torch.LongTensor)
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        test_loss += loss.detach().cpu().item() / len(test_dataloader)

        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total += len(x)
        if open_world:
            predicted_labels = torch.argmax(y_hat, dim=1).detach().cpu()
            correct_labels = y.detach().cpu()
            TP += ((predicted_labels == correct_labels) & (predicted_labels != unmonitored_class)).sum(axis=0)
            FP += ((predicted_labels != unmonitored_class) & (correct_labels == unmonitored_class)).sum(axis=0)
            TN += ((predicted_labels == correct_labels) & (correct_labels == unmonitored_class)).sum(axis=0)
            FN += ((predicted_labels == unmonitored_class) & (correct_labels != unmonitored_class)).sum(axis=0)
            WP += ((predicted_labels != correct_labels) & (predicted_labels != unmonitored_class) & (correct_labels != unmonitored_class)).sum(axis=0)

    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")
    if open_world:
      print(TP, FP, TN, FN, WP)
      print(f"Test TPR: {TP / (TP + FN) * 100:.2f}%")
      print(f"Test WPR: {WP / (TP + WP + FN) * 100:.2f}%")
      print(f"Test FPR: {FP / (FP + TN) * 100:.2f}%")
    acc_test_set.append(correct / total)
    loss_test_set.append(test_loss)
