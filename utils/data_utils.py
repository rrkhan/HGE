"""Data utils functions for pre-processing and data loading."""

import csv
import logging
from random import random
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# helper functions
def exists(val):
    return val is not None

def coin_flip():
    return random() > 0.5


string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
# augmentation
def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

class SupervisedDataset(Dataset):
    """Dataset for Genomic Datasets."""

    def __init__(self, 
                 file_path: str,
                 length: int,
                 rc_aug: bool = False
                ):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(file_path, "r") as f:
            data = list(csv.reader(f))[1:]
            # data is in the format of [text, label, ...]
            logging.warning("Perform single sequence classification...")
            seqs = [d[0] for d in data]
            labels = [int(d[1]) for d in data]

        self.input_ids = seqs
        self.all_labels = labels
        self.max_length = length
        self.rc_aug = rc_aug # toggle reverse-complement data augmentation

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):

        x = self.input_ids[idx]
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)  

        # convert to tensor
        seq = self._one_hot_encode_dna(x)

        # need to wrap in list
        target = torch.LongTensor([y])
        return seq, target

    def _one_hot_encode_dna(self, dna_sequence):
        # Define the mapping of characters to integers
        base_to_int = {'A': 0, 'C': 1, 'T': 2, 'G': 3, 'N': 4}

        # Convert the DNA sequence to a list of integers
        integer_encoded = [base_to_int[base] for base in dna_sequence]

        # Create a zero-initialized one-hot encoded array
        one_hot_encoded = np.zeros((5, self.max_length), dtype=np.float32)

        # Set the corresponding positions in the one-hot encoded array to 1
        one_hot_encoded[integer_encoded, np.arange(len(dna_sequence))] = 1

        return torch.from_numpy(one_hot_encoded)


def create_paths(args):
    paths = []
    if args.benchmark == "TEB":
        for split in ["train_", "valid_", "test_"]:
            paths.append(''.join([args.data_path, split, args.dataset_name, '.csv']))
    elif args.benchmark == "GUE":
        for split in ["/train.csv", "/dev.csv", "/test.csv"]:
            paths.append(''.join([args.data_path, args.dataset_name, split]))
    else:
        raise ValueError("Dataset format not supported.")
    print(paths)
    return paths[0], paths[1], paths[2]    


def select_dataset(args):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """
    train_path, val_path, test_path = create_paths(args)    
    
    ds_train = SupervisedDataset(
        file_path=train_path,
        length=args.length
    )

    ds_val = SupervisedDataset(
        file_path=val_path,
        length=args.length
    )

    ds_test = SupervisedDataset(
        file_path=test_path,
        length=args.length
    )

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False) 
         
    return train_loader, test_loader, val_loader