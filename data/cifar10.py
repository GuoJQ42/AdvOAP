import torch

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465], 
    'std': [0.2023, 0.1994, 0.2010],
}

class GetData(Dataset):
    # 初始化为整个class提供全局变量，为后续方法提供一些量
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.fromarray(self.features[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return self.features.shape[0]


def load_MNIST(path):
    used_data = datasets.MNIST(root = path, download = True)
    features = used_data.data
    
    features = features.view([features.shape[0], features.shape[1], features.shape[2], 1])
    features = torch.cat([features, features, features], dim=3).numpy()

    features = np.pad(features, pad_width=((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=0)

    # breakpoint()
    labels = used_data.targets
    np.random.seed(42)
    shuffled_indices = np.random.permutation(features.shape[0])
    train_data_size = features.shape[0]//10*7
    train_data_features, train_data_labels = features[shuffled_indices[:train_data_size]], labels[shuffled_indices[:train_data_size]]
    test_data_features, test_data_labels = features[shuffled_indices[train_data_size:]], labels[shuffled_indices[train_data_size:]]
    # breakpoint()
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    # train_loader = DataLoader(GetData(train_data_features, train_data_labels), batch_size = BATCH_SIZE, num_workers = 4, shuffle = True)
    # train_loader = DataLoader(GetDataAndClassifier(train_data_features, train_data_labels), batch_size = BATCH_SIZE, num_workers = 4, shuffle = True)
    # test_loader = DataLoader(GetData(test_data_features, test_data_labels), batch_size = BATCH_SIZE, num_workers = 4, shuffle = False)
    # num_channel = 1

    train_data = GetData(train_data_features, train_data_labels, train_transform)
    test_data = GetData(test_data_features, test_data_labels, test_transform)
    return train_data, test_data


def load_FashionMNIST(path):
    used_data = datasets.FashionMNIST(root = path, download = True)
    features = used_data.data
    
    features = features.view([features.shape[0], features.shape[1], features.shape[2], 1])
    features = torch.cat([features, features, features], dim=3).numpy()

    features = np.pad(features, pad_width=((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=0)

    # breakpoint()
    labels = used_data.targets
    np.random.seed(42)
    shuffled_indices = np.random.permutation(features.shape[0])
    train_data_size = features.shape[0]//10*7
    train_data_features, train_data_labels = features[shuffled_indices[:train_data_size]], labels[shuffled_indices[:train_data_size]]
    test_data_features, test_data_labels = features[shuffled_indices[train_data_size:]], labels[shuffled_indices[train_data_size:]]
    # breakpoint()
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    # train_loader = DataLoader(GetData(train_data_features, train_data_labels), batch_size = BATCH_SIZE, num_workers = 4, shuffle = True)
    # train_loader = DataLoader(GetDataAndClassifier(train_data_features, train_data_labels), batch_size = BATCH_SIZE, num_workers = 4, shuffle = True)
    # test_loader = DataLoader(GetData(test_data_features, test_data_labels), batch_size = BATCH_SIZE, num_workers = 4, shuffle = False)
    # num_channel = 1

    train_data = GetData(train_data_features, train_data_labels, train_transform)
    test_data = GetData(test_data_features, test_data_labels, test_transform)
    return train_data, test_data


def load_MNIST_per_class():
    used_data = datasets.MNIST(root = "./data/", download = True)
    features = used_data.data.float()/255
    features = features.view([features.shape[0], 1, features.shape[1], features.shape[2]])
    labels = used_data.targets
    mnist_per_class = []
    for i in range(10):
        class_idx = np.where(labels == i)[0]
        mnist_per_class.append(features[class_idx])

    avg_norm = 0
    i = 5
    j = 6
    for a in range((mnist_per_class[i].shape[0])):
        for b in range((mnist_per_class[j].shape[0])):
            tmp_res = torch.norm(mnist_per_class[i][a] - mnist_per_class[j][b])
            print(tmp_res)
            avg_norm += tmp_res
    avg_norm = avg_norm*2/((mnist_per_class.shape[0])*((mnist_per_class.shape[0])-1))
    print(avg_norm)

    for i in range(10):
        for j in range(i, 10):
            avg_norm = 0
            for a in range((mnist_per_class[i].shape[0])):
                for b in range((mnist_per_class[j].shape[0])):
                    tmp_res = torch.norm(mnist_per_class[i][a] - mnist_per_class[j][b])
                    print(tmp_res)
                    avg_norm += tmp_res
            avg_norm = avg_norm*2/((mnist_per_class.shape[0])*((mnist_per_class.shape[0])-1))
            print(avg_norm)



    for i in range(10):
        avg_norm = 0
        for a in range((mnist_per_class[i].shape[0])):
            for b in range(a+1, (mnist_per_class[i].shape[0])):
                tmp_res = torch.norm(mnist_per_class[i][a] - mnist_per_class[i][b])
                print(tmp_res)
                avg_norm += tmp_res
        avg_norm = avg_norm*2/((mnist_per_class.shape[0])*((mnist_per_class.shape[0])-1))
        print(avg_norm)

    breakpoint()

    return mnist_per_class


def load_cifar10_semi_supervised_learning(data_dir, times, use_augmentation='base'):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset_with_label = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    
    # breakpoint()

    train_feature = train_dataset_with_label.data
    train_label = np.array(train_dataset_with_label.targets)

    

    # # Label may be imbalanced
    # train_idx = np.array([i for i in range(train_feature.shape[0])])
    # np.random.shuffle(train_idx)
    # train_idx_sup = train_idx[:train_idx.shape[0]//80]
    # train_idx_unsup = train_idx[train_idx.shape[0]//80: train_idx.shape[0]//80*13]


    # Label is imbalanced
    np.random.seed(times)
    choose_idx_sup = None
    choose_idx_unsup = None
    for i in range(10):
        class_idx = np.where(train_label == i)[0]
        np.random.shuffle(class_idx)
        class_idx_sup = class_idx[:class_idx.shape[0]//80]
        class_idx_unsup = class_idx[class_idx.shape[0]//80: class_idx.shape[0]//80*13]
        if choose_idx_sup is None:
            choose_idx_sup = class_idx_sup
            choose_idx_unsup = class_idx_unsup
        else:
            choose_idx_sup = np.concatenate([choose_idx_sup, class_idx_sup])
            choose_idx_unsup = np.concatenate([choose_idx_unsup, class_idx_unsup])
    train_idx_sup = choose_idx_sup
    train_idx_unsup = choose_idx_unsup
    np.random.shuffle(train_idx_sup)
    np.random.shuffle(train_idx_unsup)

    train_feature_sup = train_feature[train_idx_sup]
    train_label_sup = train_label[train_idx_sup]
    train_feature_unsup = train_feature[train_idx_unsup]
    train_label_unsup = train_label[train_idx_unsup]


    train_dataset_sup = GetData(train_feature_sup, train_label_sup, train_transform)
    train_dataset_unsup = GetData(train_feature_unsup, train_label_unsup, train_transform)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)    

    # breakpoint()

    return train_dataset_sup, train_dataset_unsup, test_dataset



def load_cifar10(data_dir, use_augmentation='base'):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)    
    return train_dataset, test_dataset



def load_cifar10_two_class_regression(class1, class2, data_dir='./data/data', use_augmentation='base'):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset_with_label = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    

    train_feature = train_dataset_with_label.data
    train_label = np.array(train_dataset_with_label.targets)


    class1_idx = np.where(train_label == class1)[0]
    class2_idx = np.where(train_label == class2)[0]

    choose_idx_sup = np.concatenate([class1_idx, class2_idx])

    train_idx_sup = choose_idx_sup

    train_label[class1_idx] = 1
    train_label[class2_idx] = 0

    np.random.shuffle(train_idx_sup)

    train_feature_sup = train_feature[train_idx_sup]
    train_label_sup = train_label[train_idx_sup]
    train_label_sup = train_label_sup.astype(np.float32)

    train_label_sup = 2*train_label_sup - 1

    train_dataset_sup = GetData(train_feature_sup, train_label_sup, train_transform)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)   

    test_feature = test_dataset.data
    test_label = np.array(test_dataset.targets)


    class1_idx = np.where(test_label == class1)[0]
    class2_idx = np.where(test_label == class2)[0]

    choose_idx_sup = np.concatenate([class1_idx, class2_idx])

    test_idx_sup = choose_idx_sup

    test_label[class1_idx] = 1
    test_label[class2_idx] = 0

    np.random.shuffle(test_idx_sup)

    test_feature_sup = test_feature[test_idx_sup]
    test_label_sup = test_label[test_idx_sup]
    test_label_sup = test_label_sup.astype(np.float32)

    test_label_sup = 2*test_label_sup - 1

    # breakpoint()

    test_dataset = GetData(test_feature_sup, test_label_sup, test_transform) 

    return train_dataset_sup, test_dataset
    



def load_cifar10_two_class_regression(class1, class2, data_dir='./data/data', use_augmentation='base'):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset_with_label = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    

    train_feature = train_dataset_with_label.data
    train_label = np.array(train_dataset_with_label.targets)


    class1_idx = np.where(train_label == class1)[0]
    class2_idx = np.where(train_label == class2)[0]

    choose_idx_sup = np.concatenate([class1_idx, class2_idx])

    train_idx_sup = choose_idx_sup

    train_label[class1_idx] = 1
    train_label[class2_idx] = 0

    np.random.shuffle(train_idx_sup)

    train_feature_sup = train_feature[train_idx_sup]
    train_label_sup = train_label[train_idx_sup]
    train_label_sup = train_label_sup.astype(np.float32)

    train_label_sup = 2*train_label_sup - 1

    train_dataset_sup = GetData(train_feature_sup, train_label_sup, train_transform)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)   

    test_feature = test_dataset.data
    test_label = np.array(test_dataset.targets)


    class1_idx = np.where(test_label == class1)[0]
    class2_idx = np.where(test_label == class2)[0]

    choose_idx_sup = np.concatenate([class1_idx, class2_idx])

    test_idx_sup = choose_idx_sup

    test_label[class1_idx] = 1
    test_label[class2_idx] = 0

    np.random.shuffle(test_idx_sup)

    test_feature_sup = test_feature[test_idx_sup]
    test_label_sup = test_label[test_idx_sup]

    test_label_sup = test_label_sup.astype(np.float32)

    test_label_sup = 2*test_label_sup - 1

    # breakpoint()

    test_dataset = GetData(test_feature_sup, test_label_sup, test_transform) 

    return train_dataset_sup, test_dataset


def load_cifar10_two_class_classification(class1, class2, data_dir='./data/data', use_augmentation='base'):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset. 
    """
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    train_dataset_with_label = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    

    train_feature = train_dataset_with_label.data
    train_label = np.array(train_dataset_with_label.targets)


    class1_idx = np.where(train_label == class1)[0]
    class2_idx = np.where(train_label == class2)[0]

    choose_idx_sup = np.concatenate([class1_idx, class2_idx])

    train_idx_sup = choose_idx_sup

    train_label[class1_idx] = 1
    train_label[class2_idx] = 0

    np.random.shuffle(train_idx_sup)

    train_feature_sup = train_feature[train_idx_sup]

    train_label_sup = train_label[train_idx_sup]

    train_dataset_sup = GetData(train_feature_sup, train_label_sup, train_transform)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)   

    test_feature = test_dataset.data
    test_label = np.array(test_dataset.targets)


    class1_idx = np.where(test_label == class1)[0]
    class2_idx = np.where(test_label == class2)[0]

    choose_idx_sup = np.concatenate([class1_idx, class2_idx])

    test_idx_sup = choose_idx_sup

    test_label[class1_idx] = 1
    test_label[class2_idx] = 0

    np.random.shuffle(test_idx_sup)

    test_feature_sup = test_feature[test_idx_sup]
    test_label_sup = test_label[test_idx_sup]


    test_dataset = GetData(test_feature_sup, test_label_sup, test_transform) 

    return train_dataset_sup, test_dataset









