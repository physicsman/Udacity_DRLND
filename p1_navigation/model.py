# Referenced from: https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork_image_(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed, in_channels=3):
        """Initialize parameters and build model.
        
        Experimenting with 3D convolution over successive frames.
        
        Params
        ======
            action_size (int): Dimension of each action
            seed (int): Random seed # Not Used
            
        """
        super(QNetwork_image, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cuda_seed = torch.cuda.manual_seed(seed)
        self.alpha = 0.05
        s = 2
        s = 2**s

        filters = [16*s, 32*s, 64*s, 128*s] 
        kernel_size =  [3, 3, 2, 2]
        stride = [3, 3, 2, 2] 

        k = filters[0] #(in_channels*(filters[0]//in_channels))
        
        self.conv1a = nn.Conv3d(3, k, kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        #self.conv1a_bn = nn.BatchNorm2d(k)
        self.conv1b = nn.Conv3d(k, k, kernel_size=(1,2,2),stride=(1,2,2))
        self.conv1b_bn = nn.BatchNorm3d(k)
        
        self.conv2a = nn.Conv3d(k, filters[1], kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        #self.conv2a_bn = nn.BatchNorm2d(filters[1])
        self.conv2b = nn.Conv3d(filters[1], filters[1], kernel_size=(1,2,2),stride=(1,2,2))
        self.conv2b_bn = nn.BatchNorm3d(filters[1])
        
        self.conv3a = nn.Conv3d(filters[1], filters[2], kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        #self.conv3a_bn = nn.BatchNorm2d(filters[2])
        self.conv3b = nn.Conv3d(filters[2], filters[2],kernel_size=(1,2,2),stride=(1,2,2))
        self.conv3b_bn = nn.BatchNorm3d(filters[2])
        
        self.conv4a = nn.Conv3d(filters[2], filters[3], kernel_size=(3,3,3),stride=1,padding=(1,1,1))
        #self.conv4a_bn = nn.BatchNorm2d(filters[3])
        self.conv4b = nn.Conv3d(filters[3], filters[3], kernel_size=(1,2,2),stride=(1,2,2))
        self.conv4b_bn = nn.BatchNorm3d(filters[3])

        self.conv5a = nn.Conv3d(filters[3], action_size*s, kernel_size=(1,2,2),stride=(1,2,2))
        #self.conv5a_bn = nn.BatchNorm2d(action_size*s)        
        # Global Average Pooling 
        self.gapool = nn.AdaptiveAvgPool3d(1)

        #self.linear = nn.Linear(action_size, action_size)
        self.linear1 = nn.Linear(action_size*s, action_size*s)
        self.linear2 = nn.Linear(action_size*s, action_size)

    def forward(self, x, a=None, n=None):
        ## TODO: Define the feedforward behavior of this model
        x = F.selu(self.conv1a(x))
        x = F.selu(self.conv1b_bn(self.conv1b(x)))
        if n == 1: 
            return x
        x = F.selu(self.conv2a(x))
        x = F.selu(self.conv2b_bn(self.conv2b(x)))
        if n == 2: 
            return x
        x = F.selu(self.conv3a(x))
        x = F.selu(self.conv3b_bn(self.conv3b(x)))
        if n == 3: 
            return x
        x = F.selu(self.conv4a(x))
        x = F.selu(self.conv4b_bn(self.conv4b(x)))
        if n == 4: 
            return x
        x = F.selu(self.conv5a(x))
        if n == 5: 
            return x 
        x = self.gapool(x)
        if n == 6: 
            return x
        x = x.view(x.size(0), -1) # prep for linear layer
        #x = self.linear(x)
        x = F.selu(self.linear1(x))
        x = self.linear2(x)
        
        return x

class QNetwork_image(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size, seed, in_channels=3):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_image, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cuda_seed = torch.cuda.manual_seed(seed)
        self.alpha = 0.05
        s = 2
        s = 2**s

        filters = [16*s, 32*s, 64*s, 128*s] 
        kernel_size =  [3, 3, 2, 2] # [8, 4]
        stride = [3, 3, 2, 2] # [4, 2]
        
        k = filters[0] #(in_channels*(filters[0]//in_channels))
        self.conv1a = nn.Conv2d(in_channels, k, kernel_size[0], stride=1, padding=1) #, groups=in_channels)
        #self.conv1a_bn = nn.BatchNorm2d(k)
        self.conv1b = nn.Conv2d(k, k, kernel_size[0], stride=stride[0], padding=0) # , groups=in_channels)
        self.conv1b_bn = nn.BatchNorm2d(k)
        
        self.conv2a = nn.Conv2d(k, filters[1], kernel_size[1], stride=1, padding=1)
        #self.conv2a_bn = nn.BatchNorm2d(filters[1])
        self.conv2b = nn.Conv2d(filters[1], filters[1], kernel_size[1], stride=stride[1], padding=0)
        self.conv2b_bn = nn.BatchNorm2d(filters[1])
        
        self.conv3a = nn.Conv2d(filters[1], filters[2], kernel_size[2], stride=1, padding=1)
        #self.conv3a_bn = nn.BatchNorm2d(filters[2])
        self.conv3b = nn.Conv2d(filters[2], filters[2], kernel_size[2], stride=stride[2], padding=0)
        self.conv3b_bn = nn.BatchNorm2d(filters[2])
        
        self.conv4a = nn.Conv2d(filters[2], filters[3], kernel_size[3], stride=1, padding=1)
        #self.conv4a_bn = nn.BatchNorm2d(filters[3])
        self.conv4b = nn.Conv2d(filters[3], filters[3], kernel_size[3], stride=stride[3], padding=0)
        self.conv4b_bn = nn.BatchNorm2d(filters[3])

        self.conv5a = nn.Conv2d(filters[3], action_size*s, kernel_size[3], stride=stride[3], padding=0)
        #self.conv5a_bn = nn.BatchNorm2d(action_size*s)        
        # Global Average Pooling 
        self.gapool = nn.AdaptiveAvgPool2d(1)

        #self.linear = nn.Linear(action_size, action_size)
        self.linear1 = nn.Linear(action_size*s, action_size*s)
        self.linear2 = nn.Linear(action_size*s, action_size)

    def forward(self, x, a=None, n=None):
        ## TODO: Define the feedforward behavior of this model
        x = F.selu(self.conv1a(x))
        x = F.selu(self.conv1b_bn(self.conv1b(x)))
        if n == 1: 
            return x
        x = F.selu(self.conv2a(x))
        x = F.selu(self.conv2b_bn(self.conv2b(x)))
        if n == 2: 
            return x
        x = F.selu(self.conv3a(x))
        x = F.selu(self.conv3b_bn(self.conv3b(x)))
        if n == 3: 
            return x
        x = F.selu(self.conv4a(x))
        x = F.selu(self.conv4b_bn(self.conv4b(x)))
        if n == 4: 
            return x
        x = F.selu(self.conv5a(x))
        if n == 5: 
            return x 
        x = self.gapool(x)
        if n == 6: 
            return x
        x = x.view(x.size(0), -1) # prep for linear layer
        #x = self.linear(x)
        x = F.selu(self.linear1(x))
        x = self.linear2(x)
        
        return x

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cuda_seed = torch.cuda.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)