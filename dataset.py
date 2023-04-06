import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision as tv


class Chars74k(Dataset):
    def __init__(self, test=True):
        self.test = test
        if test:
            self.path = 'testResized/'
        else:
            self.path = 'trainResized/'
        self.labels = pd.read_csv('trainLabels.csv', index_col='ID')
        self.transform = tv.transforms.ToTensor()
        self.idx_to_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}

    def __getitem__(self, i):
        if self.test:
            i += 6285 # offset for file name
            y = None
        else:
            i += 1
            y = self.char_to_idx[self.labels.loc[i].item()]
        image = Image.open(os.path.join(self.path, f'{i}.Bmp')).convert('RGB')
        tensor = self.transform(image) 
        assert tensor.shape == (3, 20, 20)
        return tensor, y

    def __len__(self):
        return len(self.labels)