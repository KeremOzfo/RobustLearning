import torch

def gaussian_noise(batch,mean=0,std=1):
    new_batch = batch.add(torch.randn(batch.size()).to(batch) * std + mean)
    return new_batch