import preprocess_eeg
import numpy as np
import torch
class EEGDataPatcher:
    def __init__(self, data_loader, cls_value=1, cls_on = True):
        self.data_loader = data_loader
        self.cls_on = cls_on
        if cls_on:
            self.cls_token = torch.full((4,), cls_value)  

    def patch_data(self):
        patched_data = []
        for batch in self.data_loader:
            # batch[0] contains the EEG data
            eeg_data = batch[0]  # Assuming the EEG data is the first element in the batch
            # Iterate over each sample in the batch
            for sample in eeg_data:
                if self.cls_on:
                    sample_patches = sample.view(-1, 63, 4)
                    # Prepend CLS token to each patch
                    sample_patches_with_cls = [torch.cat([self.cls_token.unsqueeze(0), patch], dim=0) for patch in sample_patches]
                    patched_data.append(torch.stack(sample_patches_with_cls))
                else:
                    sample_patches = sample.view(-1, 63, 4)
                    patched_data.append(sample_patches)
                    
        #print out the shape of the patched data
        print(f"Shape of patched data: {np.array(patched_data).shape}")
           
            
        return patched_data
    
    
train_loader, test_loader, n_channels, n_times = preprocess_eeg.manage_loader(1)
    
patcher = EEGDataPatcher(train_loader)
patched_train_data = patcher.patch_data()