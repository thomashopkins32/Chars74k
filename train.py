import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Chars74k
from models import BaselineCNN

# PARAMETERS
BATCH_SIZE = 32
LR = 0.01
WD = 1e-4
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data = Chars74k(test=False)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
model = BaselineCNN().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)

for i, x in tqdm(enumerate(train_loader)):
    print(x)
    break
