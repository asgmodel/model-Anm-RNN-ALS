# @title Batch Dataset

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split

def padded_sequences_batch(batch):
    sequences = []
    for sequence in batch:  # Iterate over each item in the batch
        sequences.append(sequence)

    # Pad the sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    padded_sequences = padded_sequences.float()

    return padded_sequences

def create_data_loaders(dataset, batch_size=32, train_ratio=0.7, val_ratio=0.15):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=padded_sequences_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=padded_sequences_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=padded_sequences_batch)
    
    return train_loader, val_loader, test_loader
