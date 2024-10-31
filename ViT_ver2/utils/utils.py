import torch
import argparse
from torch.utils.data import DataLoader, random_split

#return what device is available
def check_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

#convert dict to namspace object
def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

#split dataset to dataloader
def split_data(dataset, split_ratio, batch_size=32):
    '''
        split_ratio (list or float):
            - If float (e.g., 0.8): Ratio for the train set, remaining goes to test.
            - If list (e.g., [0.7, 0.15, 0.15]): Ratios for train, validation, and test.
        return:
            tuple: Contains DataLoaders for train, test, and optionally validation sets.
    '''
    if isinstance(split_ratio, float):
        # Convert to list if ratio is provided as a float (train/test split)
        split_ratio = [split_ratio, 1-split_ratio]
    
    # Compute lengths for each subset
    dataset_size = len(dataset)
    lengths = [int(r*dataset_size) for r in split_ratio]

    # Adjust the last length to ensure the total matches the dataset size
    lengths[-1] = dataset_size - sum(lengths[:-1])

    # Split the dataset into subsets
    subsets = random_split(dataset, lengths)

    # Create DataLoaders
    loaders = [
        DataLoader(subset, batch_size=batch_size,
                   shuffle=(i==0))
        for i, subset in enumerate(subsets)
    ]
    return tuple(loaders)