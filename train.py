'''
Training for vanilla GAN 
'''

import time

from model import Discriminator
from model import Generator

from model import disc_loss
from model import gen_loss

from torch import cuda

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from torch import optim
from torch import ones
from torch import zeros
from torch import rand
from torch import cat
from torch import save

from matplotlib import pyplot as plt

from os import environ

environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
LOSS!!!
"""
'''
STEP - 1 - DATA PREPARATION
'''

device = 'cuda:0' if cuda.is_available() else 'cpu'

batch_size = 64

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

latent_space_samples = [((rand(1, 128)-0.5) / 0.5).to(device)
                        for k in range(25)]

mnist_dataset = MNIST('mnist/', train=True,
                               download=True, transform=transform)

mnist_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)


'''
STEP - 2 - MODEL INSTANTIATE
'''

discriminator_network = Discriminator().to(device)
generator_network = Generator().to(device)


'''
STEP - 3 - OPTIMIZIER INSTANTIATE
'''

learning_rate = 0.0002

disc_optim = optim.Adam(discriminator_network.parameters(),
                         lr=learning_rate, betas=(0.5, 0.999))


gen_optim = optim.Adam(generator_network.parameters(),
                         lr=learning_rate, betas=(0.5, 0.999))


'''
STEP - 4 - TRAINING LOOP
'''

save(generator_network, r'models\model_epoch_000')

epochs = 100
k = 1

disc_loss_data = []
gen_loss_data = []

for _ in range(epochs):
    
    disc_epoch_loss = []
    gen_epoch_loss = []
    
    start = time.time()

    '''
    Main training loop
    '''
    for step, data in enumerate(mnist_loader):
        
        '''
        Loop over batch in epoch
        '''
        
        ### DISCRIMINATOR TRAINING ###
        
        X_true_in = data[0].to(device)
        X_true_in = X_true_in.view(-1, 784)
        
        y_true_out = discriminator_network(X_true_in)
        y_true = ones(X_true_in.shape[0], 1).to(device)
        
        
        noise = (rand(X_true_in.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        X_fake_in = generator_network(noise)
        y_fake_out = discriminator_network(X_fake_in)
        y_fake = zeros(X_fake_in.shape[0], 1).to(device)
        
        disc_outputs = cat((y_true_out, y_fake_out), 0)
        disc_targets = cat((y_true, y_fake), 0)

        ### zero the parameter gradients - MOST IMPORTANT ###

        disc_optim.zero_grad()
        
        ### backprop ###
        
        d_loss = disc_loss(disc_outputs, disc_targets)
        d_loss.backward()
        disc_epoch_loss.append(d_loss.item())
        disc_optim.step()
        
        ### GENERATOR TRAINING ###
        
        noise = (rand(X_true_in.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        
        X_fake_in = generator_network(noise)
        y_fake_out = discriminator_network(X_fake_in)
        
        g_loss = gen_loss(y_fake_out, device)
        gen_optim.zero_grad()
        g_loss.backward()
        gen_epoch_loss.append(g_loss.item())
        gen_optim.step()
        
    end = time.time()
    print('EPOCH OVER')
    print(k)
    print('Time: ', end - start)

    save(generator_network, fr'models\model_epoch_{str(k).zfill(3)}')
    k = k + 1
    disc_loss_data.append(sum(disc_epoch_loss) / len(disc_epoch_loss))
    gen_loss_data.append(sum(gen_epoch_loss) / len(gen_epoch_loss))


save(generator_network, fr'models\model_epoch_{str(k).zfill(3)}')
print('MODEL SAVED')

plt.plot(list(range(epochs)), disc_loss_data, label='Discriminator')
plt.plot(list(range(epochs)), gen_loss_data, label='Generator')
plt.legend(loc="upper right")
plt.xticks(list(range(1, epochs, 5)))
plt.xlabel('Epoch')
plt.ylabel('Loss')


