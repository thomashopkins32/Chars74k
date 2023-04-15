import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from dataset import Chars74k
from models import BaselineCNN
from utils import accuracy

# PARAMETERS
BATCH_SIZE = 128
LR = 3e-4
WD = 0.0
MOMENTUM = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_STEP = 10
RNG = 2
EPOCHS = 100

torch.manual_seed(RNG)

writer = SummaryWriter()
dataset = Chars74k(test=False)
generator = torch.Generator().manual_seed(RNG)
train_data, valid_data = random_split(dataset, [0.9, 0.1], generator=generator)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False)
model = BaselineCNN().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
global_step = 0

'''
images, labels = next(iter(train_loader))
x = images.to(DEVICE)
y = labels.long().to(DEVICE)
logits = model(x)
loss = loss_func(logits, y)
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
'''

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
            global_step += 1
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Accuracy/train", accuracy(logits, y), global_step)
            for i, (x, y) in enumerate(valid_loader):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                loss = loss_func(logits, y)
                writer.add_scalar("Loss/valid", loss.item(), global_step)
                writer.add_scalar("Accuracy/valid", accuracy(logits, y), global_step)
writer.close()