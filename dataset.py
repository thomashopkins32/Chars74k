import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision as tv


class Chars74k(Dataset):
    def __init__(self, test=True):
        self.test = test
        self.labels = pd.read_csv('trainLabels.csv', index_col='ID')
        self.train_transform = tv.transforms.Compose(
            [
                tv.transforms.RandomAffine(degrees=(8, 8), translate=(0.3, 0.3), scale=(1, 1.3)),
                tv.transforms.RandomInvert(p=0.5),
                tv.transforms.ToTensor(),
            ]
        )
        self.test_transform = tv.transforms.ToTensor()
        self.idx_to_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}
        if test:
            self.path = 'testResized/'
            self.eval()
        else:
            self.path = 'trainResized/'
            self.train()
        self.images = []
        for i in range(1, len(self.labels) + 1):
            self.images.append(Image.open(os.path.join(self.path, f'{i}.Bmp')).convert('RGB'))

    def __getitem__(self, i):
        image = self.images[i]
        if self.test:
            tensor = self.test_transform(image)
        else:
            tensor = self.train_transform(image) 
        y = self.char_to_idx[self.labels.loc[i + 1].item()]
        return tensor, y

    def __len__(self):
        return len(self.labels)

    def eval(self):
        self.test = True
    
    def train(self):
        self.test = False