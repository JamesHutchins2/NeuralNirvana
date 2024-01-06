# imports
import preprocess_eeg
import numpy as np
import torch
from torch import nn
import mne
import matplotlib.pyplot as plt
import os
from model import model

config = {
    "num_channels": 63,
    "epoch_length": 128,
    "embedding_dim": 64,
    "num_heads": 7,
    "num_layers": 6,
    "dropout_rate": 0.1,
}
model_save_path = 'transformer_v2_'
final_path = 'transformer_v2_final.pt'

load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
transformer = model.Transformer(**config)
transformer.to(device)

if load_model:
    if os.path.exists(model_save_path):
        transformer = torch.load(model_save_path, map_location=device)
        print("Loaded saved model.")
    else:
        print("No saved model found. Starting training from scratch.")

    # Ensure the model is in the correct mode (train or eval) and on the right device
    transformer.to(device)
    transformer.train() 
    
def loss_fn(outputs, targets, mask):
    
    #print(f"Outputs shape: {outputs.shape}")
    #print(f"Targets shape: {targets.shape}")
    #print(f"Mask shape: {mask.shape}")
    #mask = mask.transpose(1, 0)
    
    targets = targets.transpose(1, 0)
    
    loss = (outputs - targets) ** 2
    loss = loss.mean(dim=-1)
    #print(f"Loss shape: {loss.shape}")
    mask = mask.squeeze(-1)
    loss = (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (loss * mask).sum()
    return loss

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)


for i in range(1, 48):
    train_loader, test_loader, n_channels, n_times = preprocess_eeg.manage_loader(i)
    
    for batch in train_loader:
        eeg_data = batch[0].to(device)  # Move EEG data to the device

        batch_dims = eeg_data.shape
        #print(f"Batch shape [0]: {batch_dims}")

        # Iterate over each sample in the batch
        for sample in eeg_data:
            sample = sample.unsqueeze(0)  # Add a batch dimension if your model expects it
            #print(f"Sample shape: {sample.shape}")

            # Run the transformer on the sample
            pred, masks, ids_restore, patched_data = transformer(sample)
            
            # Ensure pred, masks, ids_restore are on the same device as the loss function expects
            #pred = pred.to(device)
            #masks = masks.to(device)
            #ids_restore = ids_restore.to(device)

            # Print out shapes
            
            #now we need to patch the sample
            #print(f"Prediction shape: {pred.shape}")
            #print(f"Mask shape: {masks.shape}")
            #print(f"patched data shape: {patched_data.shape}")
            pred = pred.transpose(1, 0)
            loss = loss_fn(pred, patched_data, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #print(pred.shape)
            #rint(patched_data.shape)

            #print("Sample processed")
        #print epoch number and loss
        print(f"Epoch {i} Loss: {loss.item()}")
    
    #plot the last sample, and prediction
    """print("Plotting last sample")
    
    plt.figure(figsize=(12, 4))
    plt.plot(patched_data.cpu().detach().numpy().squeeze(-1), label="Target")
    plt.plot(pred.cpu().detach().numpy(), label="Prediction")
    plt.legend()
    plt.show()
        """
    #every 10 epochs save the model
    if i % 10 == 0:
        print("Saving model")
        torch.save(transformer, model_save_path + i + '.pt')
    
    if i % 17 == 0:
        print("Plotting last sample")
    
        plt.figure(figsize=(12, 4))
        plt.plot(patched_data.cpu().detach().numpy().squeeze(-1), label="Target")
        plt.plot(pred.cpu().detach().numpy(), label="Prediction")
        plt.legend()
        plt.show()
        
    #save the model
torch.save(transformer, model_save_path)
                
            
            #run the transfotmer on the sample
            
            
            
            