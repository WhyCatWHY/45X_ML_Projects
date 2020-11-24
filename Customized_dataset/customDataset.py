import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

# random comment

class PrepareDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass