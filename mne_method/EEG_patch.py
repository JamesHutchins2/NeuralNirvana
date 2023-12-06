import preprocess_eeg
import numpy as np
import torch
from torch import nn
class EEGDataPatcher:
    def __init__(self, data_loader, cls_value=1, cls_on = False, conv = True):
        self.data_loader = data_loader
        self.cls_on = cls_on
        self.ids_restore = None
        self.masks = None
        self.un_masked_data = None
        self.conv = conv
        if cls_on & conv:
            self.cls_token = torch.full((1,), cls_value)  
        elif cls_on & ~conv:
            self.cls_token = torch.full((4,), cls_value)

    def patch_data(self):
        patched_data = []
        for batch in self.data_loader:
            batch_dims = batch[0].shape
            print(f"Batch shape [0]: {batch_dims}")
            # batch[0] contains the EEG data
            eeg_data = batch[0]  # Assuming the EEG data is the first element in the batch
            # Iterate over each sample in the batch
            for sample in eeg_data:
                sample_shape = sample.shape
                print(f"Sample shape: {sample_shape}")
                
                if self.conv:
                
                    sample_patches = sample.view(-1, 63, 4)
                    
                
                    conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
                    # Reshaping the data to match the Conv1D input requirements: (batch_size, channels, length)
                    # Here, channels = 4 (time steps) and length = 63 (channels)
                    sample_patches_reshaped = sample_patches.permute(0, 2, 1)

                    # Applying convolution
                    output = conv(sample_patches_reshaped)
                    output_reshaped = output.permute(0, 2, 1)
                        
                    sample_patches = output_reshaped
                        
                    patched_data.append(sample_patches)
                        
                        
                    sample_patch_shape = sample_patches.shape
                    print(f"Sample patch shape post conv: {np.array(sample_patch_shape)}")
                    
        #print out the shape of the patched data
        
        #print the shape of the entire patched data
        print(f"Shape of patched data (whole): {patched_data[0].shape}")
        
        
        #now let us mask the data
        masked_datas = []
        masks = []
        ids_restore = []
        
        for batch in patched_data:
            #print the shape of the batch
            print(f"Shape of batch: {batch.shape}")
            
            masked_data, mask, id_restore = masker.mask(batch)
            masked_datas.append(masked_data)
            masks.append(mask)
            ids_restore.append(id_restore)
            
            
        
        #print out the shapes
        for i, tensor in enumerate(masked_datas):
            print(f"Shape of masked data at index {i}: {tensor.detach().numpy().shape}")
        print(f"Shape of mask: {mask.detach().numpy().shape}")
        print(f"Shape of id_restore: {id_restore.detach().numpy().shape}")
        
        
        
           
            
        return patched_data
    
    def positional_encoder(self):
        
        pos_enc = torch.zeros([128, 64, 4])

    
            
    
class masker:
    
    
    def mask(x, mask_ratio= 0.8, conv = False):
        
        if conv == False:
            N, L, D = x.shape
        else:
            D, L = x.shape
            N = 1
        
        #determine the ammount of data to keep
        
        keep = int(L * (1-mask_ratio))
        
        noise = torch.rand(N,L,device = x.device)
        ids_shuffle = torch.argsort(noise, dim = 1)
        ids_restore = torch.argsort(ids_shuffle, dim = 1)
        
        ids_keep = ids_shuffle[:, :keep]
        x_masked = torch.gather(x, dim =1, index = ids_keep.unsqueeze(-1).repeat(1,1,D))
        
        
        mask = torch.ones([N,L], device = x.device)
        mask[:,:keep] = 0
        
        mask = torch.gather(mask, dim =1, index = ids_restore)
        
        return x_masked, mask, ids_restore
        
        
    def un_mask():
        pass
    

train_loader, test_loader, n_channels, n_times = preprocess_eeg.manage_loader(1)
    
patcher = EEGDataPatcher(train_loader, conv=True)
patched_train_data = patcher.patch_data()



"""if self.cls_on:
                    #masked_data, masks, ids_restore = self.mask_data(sample)
                    #sample_patches = masked_data
                    #self.masks = masks
                    #self.ids_restore = ids_restore
                    sample_patches = sample.view(-1, 63, 4)
                    sample_patch_shape = sample_patches.shape
                    print(f"Sample patch shape: {sample_patch_shape}")
                    #let's preform a row wise convelution along the final axis (patch length axis)
                    if self.conv:
                        conv = nn.Conv1d(in_channels=4, out_channels=1, kernel_size=1)
                        # Reshaping the data to match the Conv1D input requirements: (batch_size, channels, length)
                        # Here, channels = 4 (time steps) and length = 63 (channels)
                        sample_patches_reshaped = sample_patches.permute(0, 2, 1)

                        # Applying convolution
                        output = conv(sample_patches_reshaped)
                        output_reshaped = output.permute(0, 2, 1)
                        
                        sample_patches = output_reshaped
                        
                        
                        

                        # Expand cls_token to match the batch size of sample_patches
                        cls_token_expanded = self.cls_token.unsqueeze(0).unsqueeze(-1).expand(128, -1, -1)

                        # Concatenate the expanded cls_token to sample_patches along the channel dimension
                        sample_patches_with_cls = torch.cat([sample_patches, cls_token_expanded], dim=1)
#


                        patched_data.append(sample_patches_with_cls)
                        sample_patch_shape_with_cls = sample_patches_with_cls.shape
                        print(f"Sample patch shape post conv (and cls): {sample_patch_shape_with_cls}")
                    else:

                        # Prepend CLS token to each patch
                        sample_patches_with_cls = [torch.cat([self.cls_token.unsqueeze(0), patch], dim=0) for patch in sample_patches]
                        patched_data.append(torch.stack(sample_patches_with_cls))"""