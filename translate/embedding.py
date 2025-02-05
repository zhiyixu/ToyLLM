from torch import nn
import torch
import math
from dataset import de_vocab, de_process, train_dataset


class Embedding(nn.Embedding):
    ...


