import random
import os.path
import numpy as np 

import torch
import torchvision
import torch.nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network import G, D


class Pix2Pix:
    def __init__(self, batch_size, epoch_iter, lr, betas, src_path, trg_path, save_model_path, restore_model_path, gpu):
        self.batch_size = batch_size
        self.epoch_iter = epoch_iter

        self.lr = lr
        self.betas = betas

        self.src_path = src_path
        self.trg_path = trg_path
        self.save_model_path = save_model_path
        self.restore_model_path = restore_model_path

        self.D = D()
        self.G = G()

        self.gpu = gpu

    def train(self):
        src_loader = None # get data from dataset 
        trg_loader = None # get data from dataset
        n_step = int(self.batch_size / len(src_loader))
        print('Dataset Load Success!')

        D_adam = optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        G_adam = optim.Adam(self.G.parameters(), lr=lr, betas=betas)

        if self.gpu:
            self.D = self.D.cuda()
            self.G = self.G.cuda()
            ones, zeros = Variable(torch.ones(batch_size, 1).cuda()), Variable(torch.zeros(batch_size, 1).cuda())
            BCE_loss = nn.BCE_loss().cuda()
        else:
            ones, zeros = Variable(torch.ones(batch_size, 1)), Variable(torch.zeros(batch_size, 1))
            BCE_loss = nn.BCE_loss()

        D.train()
        G.train()
        print('Training Start')
        for epoch in range(self.epoch_iter):
            for step in range(n_step):
                src_data = src_loader.next()
                trg_data = trg_loader.next()
                
                self.D.zero_grad()
                self.G.zero_grad()

                src_input = Variable(src_data)
                trg_input = Variable(trg_data)

                src_generated = self.G(src_input)

                D_src_generated = self.D(src_generated)
                D_trg_input = self.D(trg_input)

                # training D
                D_fake_loss = BCE_loss(d_src_generated, zeros)
                D_real_loss = BCE_loss(d_trg_input, ones)
                D_loss = d_fake_loss + d_real_loss
                D_loss.backward()
                D_adam.step()

                for p in D.parameters():
                    p.requires_grad = False
                
                # training G
                G_loss = BCE_loss(D_src_generated, ones)
                G_loss.backward()
                G_adam.step()
                
                for p in G.parameters():
                    p.requires_grad = True

                # logging losses
                if step % 20 == 0:
                    print(f"Epoch: {epoch} & Step: {step} => D-fake Loss: {D_fake_loss}, D-real Loss: {D_real_loss}, G Loss: {G_loss}")

            # save model
            if epoch % 20 == 0 and epoch != 0:
                torch.save(self.D.state_dict(), os.path.join(self.save_model_path, str(epoch) + 'D' + '.pth'))
                torch.save(self.G.state_dict(), os.path.join(self.save_model_path, str(epoch) + 'G' + '.pth'))
