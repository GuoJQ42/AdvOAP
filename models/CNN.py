import torch
import torch.nn.functional as F


class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 24,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(24,24,5)
        self.fc1 = torch.nn.Linear(24*5*5, 120)
        self.fc2 = torch.nn.Linear(120,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24*5*5) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_NET_regression(torch.nn.Module):
    def __init__(self):
        super(CNN_NET_regression,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 24,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(24,24,5)
        self.fc1 = torch.nn.Linear(24*5*5, 100)
        self.fc2 = torch.nn.Linear(100,1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24*5*5) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class CNN_NET_classification(torch.nn.Module):
    def __init__(self):
        super(CNN_NET_classification,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,
                                     out_channels = 24,
                                     kernel_size = 5,
                                     stride = 1,
                                     padding = 0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2,
                                       stride = 2)
        self.conv2 = torch.nn.Conv2d(24,24,5)
        self.fc1 = torch.nn.Linear(24*5*5, 100)
        self.fc2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24*5*5) #卷积结束后将多层图片平铺batchsize行16*5*5列，每行为一个sample，16*5*5个特征
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
