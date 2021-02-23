import torch

def similarity_loss(x,bs):
    loss=torch.zeros(1,bs)
    for i in range(bs):
        loss[0][i]=torch.sum(x[i][:]*x[i+bs][:])/(torch.norm(x[i][:])*torch.norm(x[i+bs][:])) # Cosine similarity
    loss=loss.mean()    
    return loss


def gaussian_noise(batch,mean=0,std=1):
    new_batch = batch.add(torch.randn(batch.size()).to(batch) * std + mean)
    return new_batch
