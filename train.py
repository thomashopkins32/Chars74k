import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Chars74k
from models import BaselineCNN
from utils import accuracy

# PARAMETERS
BATCH_SIZE = 64
LR = 0.01
WD = 1e-4
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_STEP = 10
RNG = 2
EPOCHS = 100

torch.manual_seed(RNG)

writer = SummaryWriter()
train_data = Chars74k(test=False)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
model = BaselineCNN().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
global_step = 0

for e in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE)
        y = y.long().to(DEVICE)

        logits = model(x)
        loss = loss_func(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % VALID_STEP == 0:
            global_step += i
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Accuracy/train", accuracy(logits, y), global_step)
writer.close()