
import time
from numpy.random import randint

from models_conditional import Discriminator
from models_conditional import Generator

from torch.nn import BCELoss

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
from torch import Tensor
from torch import IntTensor

from matplotlib import pyplot as plt

from os import environ


###


environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = 'cuda:0' if cuda.is_available() else 'cpu'


###


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

batch_size = 60

mnist_dataset = MNIST('mnist/', train=True,
                               download=True, transform=transform)

mnist_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)


###


discriminator_network = Discriminator().to(device)
generator_network = Generator().to(device)


###


learning_rate = 0.0002

disc_optim = optim.Adam(discriminator_network.parameters(),
                         lr=learning_rate, betas=(0.5, 0.999))

gen_optim = optim.Adam(generator_network.parameters(),
                         lr=learning_rate, betas=(0.5, 0.999))

criterion = BCELoss()

###

save(generator_network, r'models\model_epoch_000')

epochs = 2
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
    for step, (image, label) in enumerate(mnist_loader):
        step_start = time.time()
        '''
        Loop over batch in epoch
        '''
        
        ### DISCRIMINATOR TRAINING ###
        
        X_true_in = image.to(device).view(-1, 784)
        y_true_in = label.to(device)
        
        y_true_out = discriminator_network(X_true_in, y_true_in).reshape(
                     batch_size, 1)
        
        y_true = ones(X_true_in.shape[0], 1).to(device)
        true_loss = criterion(y_true_out, y_true)

        
        noise = (rand(X_true_in.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        y_fake_in = IntTensor(randint(0, 10, batch_size)).to(device)
        
        X_fake_in = generator_network(noise, y_fake_in)
        
        y_fake_out = discriminator_network(X_fake_in, y_fake_in).reshape(
                     batch_size, 1)
        

        y_fake = zeros(X_fake_in.shape[0], 1).to(device).to(device)
        fake_loss = criterion(y_fake_out, y_fake)
        
        disc_outputs = cat((y_true_out, y_fake_out), 0)
        disc_targets = cat((y_true, y_fake), 0)

        ### zero the parameter gradients - MOST IMPORTANT ###

        disc_optim.zero_grad()
        
        ### backprop ###
        
        d_loss = criterion(disc_outputs, disc_targets)
        d_loss.backward()
        disc_epoch_loss.append(d_loss.item())
        disc_optim.step()
        
        ### GENERATOR TRAINING ###
        
        noise = (rand(X_true_in.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        
        y_fake_in = IntTensor(randint(0, 10, batch_size)).to(device)
        
        X_fake_in = generator_network(noise, y_fake_in)
        y_fake_out = discriminator_network(X_fake_in, y_fake_in).reshape(
                     batch_size, 1)
        
        g_loss = criterion(y_fake_out, y_true)
        gen_optim.zero_grad()
        g_loss.backward()
        gen_epoch_loss.append(g_loss.item())
        gen_optim.step()
        step_end = time.time()
        print(step_end - step_start, step)
        
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

