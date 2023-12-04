# here we will implement a method to take the epoched data, and embed it into a series of patches so that we can embedd with cls tokens


#_______Imports____________________

import preprocess_eeg
import mne
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from preprocess_eeg import EEGDataProcessor

    
class patcher:
    
    def __init__(self, participant_number, patch_length = 3):
        
        self.participant_number = participant_number
        self.patch_length = patch_length
        self.train_loader = None
        self.test_loader = None
        self.n_channels = None
        self.n_times = None
        
        
    
    def load_data(self):
        train_loader, test_loader, n_channels, n_times = preprocess_eeg.manage_loader(self.participant_number)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_channels = n_channels
        self.n_times = n_times
        
        #call make patches
        self.make_patches(self.train_loader)
        
    def make_patches(self, data_set):
        
        batch = []
        
        for i, batch in enumerate(data_set):
            print(f"Batch: {i+1}")

            if isinstance(batch, list) and len(batch) == 1:
                # Extract the single item from the batch
                single_item = batch[0]
                
                # Check the type of the single item
                if isinstance(single_item, torch.Tensor):
                    print(f"Single item is a tensor with shape: {single_item.shape}")
                elif isinstance(single_item, list):
                    print(f"Single item is a list with length: {len(single_item)}")
                    # Further inspect elements of this list if needed
                else:
                    print(f"Single item is of type {type(single_item)}")
                    
        
        #now we will loop though each index of batch
        patches = []
        for b in batch:
            for i in range(0, 128, 4):
                for j in range(0, 63, 63):
                    patches.append(b[i:i+4, j:j+63])
                    
                    
        # convert them to Numpy arrays
        
        if isinstance(patches, torch.Tensor):
            patches = patches.numpy()
        else:
            patches = patches
            
            
        
        num_patches_x = len(range(0, 128, 4))
        num_patches_y = len(range(0, 63, 63))  # This might need adjustment
        patches = np.empty((num_patches_x * num_patches_y, 4, 63))
        
        patch_idx = 0
        height, width = patches.shape[0], patches.shape[1]
        for patch in patches:
            for i in range(0, height, 4):
                for j in range(0, width, 63):
                    # Check if there is enough data to form a (4, 63) patch
                    if i + 4 <= height and j + 63 <= width:
                        patch = patch[i:i+4, j:j+63]
                        patch_idx += 1
                        
        return patches
            
            
def main():
    #call the class
    patch_class = patcher(1)
    patch_class.load_data()

    
if __name__ == "__main__":
    main()
    
    