import torch
from torch.utils.data import Dataset


#------------------------------------------------------------------------
# This class helps us use train_diffusion_with_checkpoints
# without using 'for batch, _ in data_loader:'.

class ImageOnlyDataset:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image
    

#------------------------------------------------------------------------
# A class to make a NumPy array into a TensorFlow dataset.
# Works for any dimensions. The name is a legacy artifact.
class DataD(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data) # same as self.data.shape[0]

  def __getitem__(self, idx):
    return self.data[idx] # same as self.data[idx,]