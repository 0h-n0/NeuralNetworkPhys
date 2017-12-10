import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms

class BinaryLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super(BinaryLinear, self).__init__()                
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(in_size, out_size)        
        self.init_weights()
        self.set_hook()
        
    def init_weights(self):
        def _init_weights(m):
            if type(m) == nn.Linear:
                binary_w = np.random.choice([-1, 1],
                                          size=(self.in_size,
                                                self.out_size))
                binary_b = np.random.choice([-1, 1],
                                          size=(self.in_size))
                binary_w = binary_w.astype(np.float32)
                binary_b = binary_b.astype(np.float32)
                m.weight.data = torch.FloatTensor(binary_w)
                m.bias.data = torch.FloatTensor(binary_b)
            self.apply(_init_weights)

    def set_hook(self):
        def binarize(m, inputs):
            w = m.fc.weight.data.numpy()
            w = np.where(w > 0, 1, -1)
            b = m.fc.bias.data.numpy()
            b = np.where(b > 0, 1, -1)
            m.fc.weight.data = \
                torch.FloatTensor(w.astype(np.float32))
            m.fc.bias.data = \
                torch.FloatTensor(b.astype(np.float32))
        self.register_forward_pre_hook(binarize)

    def forward(self, x):
        return self.fc(x)
        
class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.fc1 = BinaryLinear(784, 200)
        self.fc2 = BinaryLinear(200, 50)
        self.fc3 = BinaryLinear(50, 10)                

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)        
        return F.softmax(x)

lr = 0.01
momentum = 0.9
batch_size = 10
epochs = 10
cuda = None
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

model = BinaryNet()
optimizer = optim.SGD(model.parameters(),
                      lr=lr, momentum=momentum)
loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

        
for epoch in range(1, epochs + 1):
    train(epoch)
