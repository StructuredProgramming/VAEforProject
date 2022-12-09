# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:20:09 2022

@author: abhay
"""

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import os
import ast
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image as PImage
device="cpu"
def weights_init(m):
    if isinstance(m, nn.Linear):
      torch.nn.init.kaiming_uniform_(m.weight)
class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encode=nn.Sequential(
        nn.Linear(720,540),
        nn.ReLU(),
        nn.Linear(540,400),
        nn.ReLU(),
        nn.Linear(400,300),
        nn.ReLU(),
        nn.Linear(300,225),
        nn.ReLU(),
        nn.Linear(225,150),
        nn.ReLU(),
        nn.Linear(150,100),
        nn.ReLU(),
        nn.Linear(100,60),
        nn.ReLU(),
        nn.Linear(60,45)
        )
        self.fc_mu = nn.Linear(45, z_dim)
        self.fc_logvar = nn.Linear(45, z_dim)
        # Decoder
        self.decode=nn.Sequential(
        nn.Linear(z_dim, 45),
        nn.ReLU(),
        nn.Linear(45, 60),
        nn.ReLU(),
        nn.Linear(60,100),
        nn.ReLU(),
        nn.Linear(100,150),
        nn.ReLU(),
        nn.Linear(150,225),
        nn.ReLU(),
        nn.Linear(225,300),
        nn.ReLU(),
        nn.Linear(300,400),
        nn.ReLU(),
        nn.Linear(400,540),
        nn.ReLU(),
        nn.Linear(540,720)
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
vae.load_state_dict(torch.load("AbhayFourierTransformVAEWeights", map_location=torch.device('cpu')))
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(20, 90)
        nn.init.kaiming_uniform_(self.layer_1.weight)
        self.layer_2 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_2.weight)
        self.layer_3 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_3.weight)
        self.layer_4 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_4.weight)
        self.layer_5 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_5.weight)
        self.layer_6 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_6.weight)
        self.layer_7 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_7.weight)
        self.layer_8 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_8.weight)
        self.layer_9 = nn.Linear(90, 90)
        nn.init.kaiming_uniform_(self.layer_9.weight)
        self.layer_10 = nn.Linear(90,6)
        nn.init.kaiming_uniform_(self.layer_10.weight)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):

        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.relu(self.layer_5(x))
        x = self.relu(self.layer_6(x))
        x = self.relu(self.layer_7(x))
        x = self.relu(self.layer_8(x))
        x = self.relu(self.layer_9(x))
        x = self.tanh(self.layer_10(x))

        return x
def coords(m):
    for i in range(2, len(m) - 3):
        if m[i] == ',':
            return m[2:i],m[(i+2):(len(m)-2)]
with open('x_y (1).txt', 'r') as f: 
    lines = f.readlines()
epoch=0
model2=NeuralNetwork()
running_loss=0
optimizer = torch.optim.Adam(model2.parameters(), 1e-3)
for line in lines:
        epoch+=1
        print(epoch)
        x, y = line.split('=')[0], line.split('=')[1]
        x, y = x.split(' '), y.split(' ')
        a=line.split('=')
        joint1=a[2]
        joint2=a[3]
        joint3=a[4]
        joint4=a[5]
        joint5=a[6]
        x1,y1=coords(joint1)
        x2,y2=coords(joint2)
        x3,y3=coords(joint3)
        x4,y4=coords(joint4)
        x5,y5=coords(joint5)
        x = [i for i in x if i]
        y = [i for i in y if i]
        x[0] = x[0][1:]
        y[0] = y[0][1:]
        x[-1] = x[-1][:-1]
        y[-1] = y[-1][:-1]

        x = [float(i) for i in x if i]
        y = [float(i) for i in y if i]
        S=np.zeros(360, dtype='complex_')
        for i in range(0,360):
            for k in range(360):
                a=x[k]
                b=y[k]
                tmp = ((-2j*np.pi*i*k)) /360
                S[i] += (complex(a,b)) * np.exp(tmp)
            S[i]=S[i]/360
        input_list=[]
        for i in range(0,360):
            input_list.append(np.real(S[i]))
            input_list.append(np.imag(S[i]))
        input_list=torch.FloatTensor(input_list)
        latent_vector=vae.encoder(input_list)
        myvector=latent_vector[1]
        prediction=model2(myvector)
        output_list=[float(x2),float(x4),float(x5),float(y2),float(y4),float(y5)]
        output_tensor=torch.tensor(output_list)
        loss_function2=nn.MSELoss()
        loss=loss_function2(prediction,output_tensor)
        running_loss+=loss
        if(epoch==1500):
            print("Current avg loss")
            print(running_loss/1500)
        if(epoch>2000 and epoch<2010):
            print("New epoch")
            print("Joint 1")
            print(prediction[0])
            print(prediction[3])
            print(output_list[0])
            print(output_list[3])
            print("Joint 4")
            print(prediction[1])
            print(prediction[4])
            print(output_list[1])
            print(output_list[4])
            print("Joint 5")
            print(prediction[2])
            print(prediction[5])
            print(output_list[2])
            print(output_list[5])
        if(epoch<34000):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())
