# This is conditional General Adversial Network, based on dense netowork


from torch.nn import Module
from torch.nn import Sequential

from torch.nn import Embedding
from torch.nn import LeakyReLU
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Tanh
from torch import cat


### --- DISCRIMINATOR CLASS --- ###

class Discriminator(Module):
    
    def __init__(self):
        super().__init__()
        
        self.label_emb = Sequential(
            Embedding(10, 10)
        )
        
        self.model = Sequential(
            Linear(794, 1024),
            LeakyReLU(0.2),
            Dropout(0.3),
            Linear(1024, 512),
            LeakyReLU(0.2),
            Dropout(0.3),
            Linear(512, 256),
            LeakyReLU(0.2),
            Dropout(0.3),
            Linear(256, 1),
            Sigmoid()
        )
    
    def forward(self, img, labels):

        y = self.label_emb(labels)
        X = img.reshape(-1, 784)
        X = cat((X, y), 1)
        outputs = self.model(X)
        return outputs.squeeze()

    
    


### --- GENERATOR CLASS --- ###

class Generator(Module):
    
    def __init__(self):
        super().__init__()
        
        self.label_emb = Embedding(10, 10)
        
        self.model = Sequential(
            Linear(138, 256),
            LeakyReLU(0.2),
            Linear(256, 512),
            LeakyReLU(0.2),
            Linear(512, 1024),
            LeakyReLU(0.2),
            Linear(1024, 784),
            Tanh()
        )
    
    def forward(self, Z, y):
        Z = Z.view(Z.size(0), 128)
        c = self.label_emb(y)
        Z = cat([Z, c], 1)
        outputs = self.model(Z)
        return outputs.view(Z.size(0), 28, 28)


