# This is file with models for deep convolution general adverisal network
# on mnist dataset

# General import for neural network

from torch.nn import Module
from torch.nn import Sequential
from torch.nn import BCELoss
# from torch.optim import Adam

# Import for generator

# from torch.nn import Unflatten
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torch.nn import ReLU
from torch.nn import Tanh

# Import for discriminator

from torch.nn import Conv2d
from torch.nn import LeakyReLU
from torch.nn import Flatten
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import Sigmoid

from torch import ones

## --- MODELS --- ##

### --- DISCRIMINATOR --- ###

class Discriminator(Module):
    
    '''
    Discriminator network is typical deep convolution neural network with
    conv layers for feature selection and dense layer for classification
    '''
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = Sequential(
            
            Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=1),
            BatchNorm2d(64),
            LeakyReLU(0.2),
            
            Conv2d(in_channels=64, out_channels=256, kernel_size=4, stride=2),
            BatchNorm2d(256),
            LeakyReLU(0.2),
            
            Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2),
            BatchNorm2d(512),
            LeakyReLU(0.2),
            
            Flatten(),
            
            Linear(8192, 1024),
            LeakyReLU(0.2),
            Dropout(0.32),
            
            Linear(1024, 64),
            LeakyReLU(0.2),
            Dropout(0.2),
            
            Linear(64, 1),
            Sigmoid(),
            )
        
    def forward(self, inputs):
        return self.main(inputs)
    
### --- GENERATOR --- ###

class Generator(Module):
    
    '''
    Generator network is based on deconvolution layers
    '''
    
    def __init__(self):
        super(Generator, self).__init__()
        self.main = Sequential(
 
            ConvTranspose2d(128, 112, 4, 1, 0),
            BatchNorm2d(112),
            ReLU(),
            
            ConvTranspose2d(112, 56, 3, 2, 1),
            BatchNorm2d(56),
            ReLU(),
            
            ConvTranspose2d(56, 28, 4, 2, 1),
            BatchNorm2d(28),
            ReLU(),
            
            ConvTranspose2d(28, 1, 4, 2, 1),
            Tanh()
            
            )
        
    def forward(self, inputs):
        return self.main(inputs)
    
    
### --- LOSS FUNCTION FOR GENERATOR AND DISCRIMINATOR --- ###


def disc_loss(inputs, targets):
    
    return BCELoss()(inputs, targets)


def gen_loss(inputs, device):
    
    '''
    As device we use CPU like medieval monkeys.
    '''
    targets = ones([inputs.shape[0], 1])
    targets = targets.to(device)
    
    return BCELoss()(inputs, targets)
    