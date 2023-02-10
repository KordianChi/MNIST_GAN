#This is file with pytorch GAN model and utils function

from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import BCELoss

from torch import ones

class Discriminator(Module):
    
    '''
    Discriminator network, basic dense feed forward network, 28x28=784 is
    dimension of MNIST picture (flat input), alpha parameter in LeakyReLU 0.2
    '''
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = Sequential(
            Linear(784, 256),
            LeakyReLU(0.2),
            Linear(256, 256),
            LeakyReLU(0.2),
            Linear(256, 1),
            Sigmoid()
        )
        
    '''
    Dense feedforward network in pytorch need only this method.
    '''

    def forward(self, input):
        return self.main(input)
    
    

class Generator(Module):
    
    '''
    Generator network, latent space dim is 128, 784 output need reshape
    '''
    
    def __init__(self):
        super(Generator, self).__init__()
        self.main = Sequential(
            Linear(128, 1024),
            ReLU(),
            Linear(1024, 1024),
            ReLU(),
            Linear(1024, 784), # Reshape to proper output shape
            Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
    
### --- LOSS FUNCTION FOR GENERATOR AND DISCRIMINATOR --- ###

'''
This part is one of crucial probably, after testing many approaches,
conclusion is that network shape is less problem than proper training
procedure, we use  Binary cross entropy.
'''

def disc_loss(inputs, targets):
    
    return BCELoss()(inputs, targets)


def gen_loss(inputs, device):
    
    '''
    As device we use CPU like medieval monkeys.
    '''
    targets = ones([inputs.shape[0], 1])
    targets = targets.to(device)
    
    return BCELoss()(inputs, targets)
    




    

