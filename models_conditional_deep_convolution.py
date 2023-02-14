

from torch.nn import Module
from torch.nn import Sequential

from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import BatchNorm2d
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Linear
from torch.nn import Embedding
from torch.nn import LeakyReLU
from torch.nn import Flatten
from torch.nn import Dropout
from torch.nn import Sigmoid

from torch import cat

### --- DISCRIMINATOR CLASS --- ###


class Discriminator(Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.label_emb = Sequential(
            Embedding(10, 10),
            Linear(10, 784)
            )
        
        self.main = Sequential(
            Conv2d(in_channels=2, out_channels=64, kernel_size=4, stride=1),
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
    
    def forward(self, img, label):
        
        y = self.label_emb(label)
        y = y.view(-1, 1, 28, 28)
        X = cat((img, y), 1)
        outputs = self.main(X)
        return outputs.squeeze()
        
### --- GENERATOR CLASS --- ###

class Generator(Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.label_emb = Sequential(
            Embedding(10, 10),
            Linear(10, 16)
            )
        
        self.latent = Sequential(
            Linear(128, 2048),
            LeakyReLU(0.2)
            )
            
        self.main = Sequential(
            
            ConvTranspose2d(129, 56, 4, 1, 0),
            BatchNorm2d(56),
            ReLU(),
            
            ConvTranspose2d(56, 28, 4, 2, 1),
            BatchNorm2d(28),
            ReLU(),

            ConvTranspose2d(28, 1, 4, 2, 1),
            Tanh()
            
            )
        
    def forward(self, noise, label):
        
        X = self.latent(noise)
        X = X.view(-1, 128, 4, 4)
        y = self.label_emb(label)
        y = y.view(-1, 1, 4, 4)
        X = cat([X, y], 1)
        outputs = self.main(X)
        return outputs
