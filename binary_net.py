import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class BinaryLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super(BinaryLinear, self).__init__()                
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(in_size, out_size)        
        self.init_weights()
        self.set_forward_hook()
        
    def init_weights(self):
        def _init_weights(m):
            if type(m) == nn.Linear:
                binary_w = np.random.choice([-1, 1],
                                          size=(self.in_size,
                                                self.out_size))
                binary_b = np.random.choice([-1, 1],
                                          size=(self.in_size))
                binary_w = binary_w.astype(np.int16)
                binary_b = binary_b.astype(np.int16)
                m.weight.data = torch.ShortTensor(binary_w)
                m.bias.data = torch.ShortTensor(binary_b)                
        self.apply(_init_weights)

    def set_forward_hook(self):
        def binarize(mod, inputs, outputs):
            if type(mod) == nn.Linear:
                print(mod.weight.data)
        self.register_forward_hook(binarize)
        
    def forward(self, x):
        return self.fc(x)
        
class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)                

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)        
        return F.log_softmax(x)
    
bl = BinaryLinear(2, 2)
x = np.ones(2)
x = Variable(torch.ShortTensor(x.astype(np.int16)))
print(bl(x))
