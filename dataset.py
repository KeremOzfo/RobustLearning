from torchvision import datasets, transforms
from torch.utils.data import DataLoader
def get_cifar10():
    norm_mean = 0
    norm_var = 1
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((norm_mean, norm_mean, norm_mean), (norm_var, norm_var, norm_var)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm_mean, norm_mean, norm_mean), (norm_var, norm_var, norm_var)),
    ])
    cifar_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    cifar_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    return cifar_train, cifar_test