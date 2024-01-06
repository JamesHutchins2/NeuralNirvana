# imports
import preprocess_eeg
import numpy as np
import torch
from torch import nn
import mne
import matplotlib.pyplot as plt
import os

#now we will test the training of the transformer
from model import model
config = {
    "num_channels": 63,
    "epoch_length": 128,
    "embedding_dim": 63,
    "num_heads": 7,
    "num_layers": 6,
    "dropout": 0.1,
    
}

config = {
    "num_channels": 63,
    "epoch_length": 128,
    "embedding_dim": 64,
    "num_heads": 8,
    "num_layers": 6,
    "dropout_rate": 0.1,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
transformer = model.Transformer(config).to(device)


model_save_path = 'transformer.pt'

load_model = False

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
    mask = mask.transpose(1, 0)
    
    loss = (outputs - targets) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum() if mask.sum() != 0 else (loss * mask).sum()
    return loss

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
#loss_fn = nn.MSELoss().to(device)
from EEG_patch import EEGDataPatcher
for i in range(1,10):
    train_loader, test_loader, n_channels, n_times = preprocess_eeg.manage_loader(i)
    patcher = EEGDataPatcher(train_loader, conv=True)
    masked_datas, masks, ids_restore, patched_data = patcher.patch_data()
    
    #move all of the data to the device
    masked_datas = [masked_data.to(device) for masked_data in masked_datas]
    masks = [mask.to(device) for mask in masks]
    ids_restore = [id_restore.to(device) for id_restore in ids_restore]
    patched_data = [patched_data.to(device) for patched_data in patched_data]
    
    print("epoch number: " + str(i) + "moved to device")
    
    for j in range(len(masked_datas)):
        # Move original_data to the same device as the model and inputs
        original_data = patched_data[j]
        original_data.squeeze(0)
        
        #print(f"original_data shape: {original_data.shape}")
        
        masked_datas[j] = masked_datas[j].float()
        
        
        batch = masked_datas[j]
        batch = batch.squeeze(-1)
        
        #print(f"batch shape: {batch.shape}")
        
        logits = transformer(batch)
        #print(f"Logits shape: {logits.shape}")
        # Make sure the target tensor is on the same device as logits
        target = original_data
        target = target.squeeze(-1)
        #print(f"Target shape: {target.shape}")
        loss = loss_fn(logits, target, masks[j])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        """if j % 100 == 0:
            #print the loss
            print(f"Epoch {i}, Loss: {loss.item()}")
            
            #move target and logits to cpu
            target = target.cpu()
            logits = logits.cpu()
            
            #plot the target and the logits
            plt.plot(target[0].detach().numpy())
            plt.plot(logits[0].detach().numpy())
            plt.show()"""
            
        

    print(f"Epoch {i}, Loss: {loss.item()}")

    # Move target and logits to CPU and detach them from the computation graph
    target_cpu = target.cpu().detach()
    logits_cpu = logits.cpu().detach()

        # Convert tensors to numpy arrays for plotting
    target_numpy = target_cpu.numpy().transpose(1, 0)
    logits_numpy = logits_cpu.numpy().transpose(1, 0)

        # Plot all 63 channels
    plt.figure(figsize=(15, 10))  # Adjust figure size as needed
    for channel in range(63):
        plt.plot(target_numpy[channel], label=f'Target Channel {channel + 1}')
        plt.plot(logits_numpy[channel], label=f'Logits Channel {channel + 1}', linestyle='--')

    plt.title("Target vs. Logits Across All Channels")
    plt.xlabel("Time Steps")
    plt.ylabel("Channel Values")
    plt.legend()  # You can comment this out if the legend makes the plot too crowded
    plt.show()
    
#save the model
torch.save(transformer, 'transformer.pt')
    
    