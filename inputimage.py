# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:28:39 2022

@author: abhay
"""
#Uses old dataset, the one with the images of Coupler Curves, not the new x,y txt file
#Issue with iterating through the folders hits after a few hundred epochs
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
os.chdir('D:/abhay/')
SAVEDIR = "D:/abhay/wavelets/image/output/"
IMAGEDIR="D:/abhay/wavelets/dataset"
import skimage.io
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, z_dim)
        self.fc2_logvar = nn.Linear(h_dim, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim, x_dim)

    def encoder(self, x):
        x1= torchvision.transforms.functional.to_tensor(x)
        h = F.relu(self.fc1(x1))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def sampling(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample

    def decoder(self, z):
        h = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encoder(np.reshape(x,(-1, 4096)))
        z = self.sampling(mu, logvar)
        return self.decoder(z), mu, logvar

vae = VAE(x_dim=4096, h_dim=400, z_dim=20).to(device)

optimizer = optim.Adam(vae.parameters(), lr=0.001)
# reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, logvar):
    x= torchvision.transforms.functional.to_tensor(x)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4096), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


dirpath = os.listdir(IMAGEDIR)
def train(epoch):
    train_loss = 0
    for subdir in dirpath:
        full_subdir= IMAGEDIR + '/' + subdir + '/'
        imageList = os.listdir(full_subdir)
        for image in imageList:
            fullpath = full_subdir + image
            inputImage=PImage.open(fullpath)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(inputImage)
            recon_batch1=recon_batch
            recon_batch=recon_batch.view(-1,4096)
            loss = loss_function(recon_batch, inputImage, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
               
            print(f"loss: {loss:>7f}")


train(1)
