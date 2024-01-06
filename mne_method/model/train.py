from model import Transformer
import torch
from torch import nn
import numpy as np
from mne_method.EEG_patch import EEG_patch
from mne_method.model import Transformer
from mne_method.model import EEGDataPatcher
from mne_method.model import EEGDataLoader
from mne_method.preprocess_eeg import EEGDataProcessor
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

config = {
    "num_channels": 63,
    "epoch_length": 128,
    "embedding_dim": 63,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
    
}

model = Transformer(config)

# Define your optimizer, loss function, and data loaders
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        # Apply masking
        x_masked, mask, _ = masker.mask(batch)

        # Forward pass
        logits = model(x_masked)

        # Compute loss and backpropagate
        loss = loss_fn(logits, batch) # Adjust depending on your exact task
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")