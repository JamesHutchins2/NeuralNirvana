import numpy as np
import torch


class masker:
    
    
    def mask(x, mask_ratio= 0.8):
        
        N, L, D = x.shape
        
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