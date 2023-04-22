import torch

torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
rootdir=r"/Users/liguang/课业/4-2-0-0毕业论文/1_参考工作/C. Spampinato/eeg_visual_classification-main/"
subject=0
model_type="lstm"
split_num=0
time_low=20
time_high=460
batch_size=16
eeg_dataset=rootdir+r"data/eeg_5_95_std.pth"
splits_path=rootdir+r"data/block_splits_by_image_all.pth"


# Dataset class
class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if subject != 0:
            self.data = [
                loaded['dataset'][i] for i in range(len(loaded['dataset']))
                if loaded['dataset'][i]['subject'] == subject
            ]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[time_low:time_high, :]

        if model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, time_high - time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label


# Splitter class
# BL: Splitter is used to split a dataset into training sets, val sets and test sets.
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [
            i for i in self.split_idx
            if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600
        ]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label


# Load dataset
dataset = EEGDataset(eeg_dataset)
# Create loaders
loaders = {
    split: DataLoader(Splitter(dataset,
                               split_path=splits_path,
                               split_num=split_num,
                               split_name=split),
                      batch_size=batch_size,
                      drop_last=True,
                      shuffle=True)
    for split in ["train", "val", "test"]
}
