import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from PIL import Image as PImage
device="cpu"
trainloss=0
testloss=0
def weights_init(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight)
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encode=nn.Sequential(
        nn.Linear(22,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720)
        )
        self.fc_mu = nn.Linear(720, z_dim)
        self.fc_logvar = nn.Linear(720, z_dim)
        # Decoder
        self.decode=nn.Sequential(
        nn.Linear(z_dim, 720),
        nn.ReLU(),
        nn.Linear(720, 720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.ReLU(),
        nn.Linear(720,720),
        nn.Tanh(),
        nn.Linear(720,22)
        )
        nn.init.kaiming_uniform_(self.fc_mu.weight)
        nn.init.kaiming_uniform_(self.fc_logvar.weight)
        self.encode.apply(weights_init)
        self.decode.apply(weights_init)

        
    def encoder(self, x):
        a=self.encode(x)
        return self.fc_mu(a),self.fc_logvar(a)     

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        return self.decode(z)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        return self.decoder(z), mu, logvar
  
vae = VAE(z_dim=20).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
def loss_function(recon_x, x, mu, logvar):
    myloss=nn.MSELoss()
    MSE=myloss(recon_x,x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def trainortest(myvector):
    optimizer.zero_grad()
    recon_batch, mu, logvar = vae(myvector)
    loss = loss_function(recon_batch, myvector, mu, logvar)
    if(epoch<34000):
        loss.backward()
        myloss=loss.item()
        optimizer.step()
    else:
        myloss=loss.item()
    print(myloss)
    return myloss
with open('x_y (1).txt', 'r') as f: 
    lines = f.readlines()
epoch=0

trainloss=0
testloss=0
#batch_size will be 60
for line in lines:
        epoch+=1
        print(epoch)
        x, y = line.split('=')[0], line.split('=')[1]
        x, y = x.split(' '), y.split(' ')
        x = [i for i in x if i]
        y = [i for i in y if i]
        x[0] = x[0][1:]
        y[0] = y[0][1:]
        x[-1] = x[-1][:-1]
        y[-1] = y[-1][:-1]

        x = [float(i) for i in x if i]
        y = [float(i) for i in y if i]
        S=np.zeros(360, dtype='complex_')
        i=0
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=359
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=1
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=358
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=2
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=357
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=3
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=356
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=4
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=355
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        i=5
        for k in range(360):
            a=x[k]
            b=y[k]
            tmp = ((-2j*np.pi*i*k)) /360
            S[i] += (complex(a,b)) * np.exp(tmp)
        S[i]=S[i]/360
        input_list=[float(np.real(S[355])),float(np.real(S[356])),float(np.real(S[357])),float(np.real(S[358])), float(np.real(S[359])), float(np.real(S[0])), float(np.real(S[1])),float(np.real(S[2])),float(np.real(S[3])),float(np.real(S[4])),float(np.real(S[5])), float(np.imag(S[355])),float(np.imag(S[356])),float(np.imag(S[357])), float(np.imag(S[358])), float(np.imag(S[359])), float(np.imag(S[0])), float(np.imag(S[1])), float(np.imag(S[2])), float(np.imag(S[3])),float(np.imag(S[4])),float(np.imag(S[5]))]
        input_tensor=torch.tensor(input_list)
        answer=trainortest(input_tensor)

        if(epoch<34000):
            trainloss+=answer
        else:
            testloss+=answer

print(trainloss/34000)
print(testloss/(epoch-34000))
        
