import os

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

    def __getitem__(self, i):
        if self.test:
            i += 6285 # offset for file name
            y = None
        else:
            i += 1
            y = self.labels.loc[i]
        image = Image.open(os.path.join(self.path, f'{i}.Bmp'))
        tensor = self.transform(image) 
        assert tensor.shape == (3, 20, 20)
        return tensor