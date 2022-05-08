import numpy as np
import torch
import torch.nn as nn

# simple bottle neck structure
# will serve a benchmark
class Bottleneck(nn.Module):
    
    def __init__(self, Veclen, Necklen):
        super(Bottleneck, self).__init__()
        
        self.fc1 = nn.Linear(Veclen, Necklen)
        
        self.bn1 = nn.BatchNorm1d(Necklen)
        
        self.fc2 = nn.Linear(Necklen, Necklen)
        
        self.bn2 = nn.BatchNorm1d(Necklen)
        
        self.fc3 = nn.Linear(Necklen, Veclen)
        
        self.bn3 = nn.BatchNorm1d(Veclen)
        
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.fc4 = nn.Linear(2,1)
        
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        residual = torch.unsqueeze(x, 1)

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = torch.unsqueeze(out, 1)
        
        out = torch.cat((out,residual),1).permute(0,2,1)
        
        out = torch.squeeze(self.fc4(out),2)
        
        out = self.sigmoid(out)

        return out

class MarkovStructure(nn.Module):
    
    def __init__(self, Veclen):
        super(MarkovStructure, self).__init__()
        
        self.weights = nn.Parameter(torch.rand(1,Veclen, Veclen))

        self.bias = nn.Parameter(torch.rand(1,Veclen, Veclen))

        #vself.In = nn.InstanceNorm1d(Veclen, affine=True)

        self.fc = nn.Linear(Veclen, 1)
        
        self.sigmoid = nn.Sigmoid()
        
        
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        x = self.weights * x + self.bias

        # x = self.In(x)

        x = self.fc(x)
        
        out = torch.squeeze(x,2)
        
        out = self.sigmoid(out)

        return out
