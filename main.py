import nn_classes as networks
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from dataset import *
from dataset_utlis import *

def look_norms(net):
    vec = []
    for layer in net.modules():  # Prune only convolutional and linear layers
        if isinstance(layer, networks.kerem_layer):
            for p in layer.parameters():
                print(p.item())
                vec.append(p.item())
    return None
def epoch(loader, model, opt=None, device=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0., 0.
    if opt is None:
        model.eval()
    else:
        model.train()

    for img, label in loader:
        img, label = img.to(device), label.to(device)
        img_aug = gaussian_noise(img) # augmented image batch
        img_cum=torch.cat((img,img_aug),0) # combine the original and augmented images
        label_cum=torch.cat((label,label),0) # combine labels
        predict1, predict2 = model(img_cum)
        loss = nn.CrossEntropyLoss()(predict1, label_cum) + 0.001*similarity_loss(predict2,128): # Two loss term; first) cross-entropy second) cosine similarity
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_err += (predict1.max(dim=1)[1] != label).sum().item()
        total_loss += loss.item() * img.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

device =torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")

cifar_train, cifar_test = get_cifar10()
train_loader = DataLoader(cifar_train, batch_size = 128, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size = 100, shuffle=True)

model = networks.get_resnet18().to(device)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
accs = []

for ep in range(300):
    if ep == 150:
        for param_group in opt.param_groups:
                param_group['lr'] = 0.01
    elif ep == 200:
        for param_group in opt.param_groups:
                param_group['lr'] = 0.001
    train_err, train_loss = epoch(train_loader, model, opt, device=device)
    test_err, test_loss = epoch(test_loader, model, device=device)
    acc = (1-test_err)*100
    accs.append(acc)
    print(acc)
    look_norms(model)
