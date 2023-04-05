import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import Chars74k

# PARAMETERS
BATCH_SIZE = 32
LR = 0.01
WD = 1e-4

train_data = Chars74k(test=False)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
optimizer = optim.SGD(lr=LR, wd=WD)



