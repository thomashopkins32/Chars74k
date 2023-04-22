import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from dataset import *
from models import *
from utils import *

# PARAMETERS
BATCH_SIZE = 64
LR = 0.1
WD = 1e-8
MOMENTUM = 0.9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_STEP = 10
RNG = 32
EPOCHS = 300

SHOW_MODEL = True
SHOW_SAMPLES = False

torch.manual_seed(RNG)

writer = SummaryWriter()
dataset = Chars74k(test=False)
generator = torch.Generator().manual_seed(RNG)
train_data, valid_data = random_split(dataset, [0.9, 0.1], generator=generator)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False)
model = CNNv2().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 100, 200], 0.1, verbose=False)
global_step = 0

if SHOW_MODEL:
    images, labels = next(iter(train_loader))
    x = images.to(DEVICE)
    writer.add_graph(model, x)
if SHOW_SAMPLES:
    images, labels = next(iter(train_loader))
    x = images.to(DEVICE)
    y = labels.long().to(DEVICE)
    logits = model(x)
    loss = loss_func(logits, y)
    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)

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
    scheduler.step()
writer.close()