import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm

from dataset import *
from models import *
from utils import *

# PARAMETERS
BATCH_SIZE = 128
LR = 0.001
WD = 0.1
MOMENTUM = 0.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_STEP = 10
RNG = 16
EPOCHS = 500

SHOW_MODEL = False
SHOW_SAMPLES = False

torch.manual_seed(RNG)

writer = SummaryWriter(max_queue=1000, flush_secs=300)
dataset = Chars74k(test=False)
generator = torch.Generator().manual_seed(RNG)
train_data, valid_data = random_split(dataset, [0.9, 0.1], generator=generator)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False, pin_memory=True)
model = CNNv3().to(DEVICE)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
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
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("Accuracy/train", accuracy(logits, y), global_step)

        if i % VALID_STEP == 0:
            global_step += 1
    model.eval()
    dataset.eval()
    for i, (x, y) in enumerate(valid_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        loss = loss_func(logits, y)
        acc = accuracy(logits, y)
        writer.add_scalar("Loss/valid", loss.item(), global_step)
        writer.add_scalar("Accuracy/valid", acc, global_step)
    model.train()
    dataset.train()
    # scheduler.step(acc)
writer.close()