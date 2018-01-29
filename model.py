import random
import os.path
import numpy as np 

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.utils as vutils

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network import G, D
from dataset import get_loader


class Pix2Pix:
    def __init__(self, batch_size, epoch_iter, lr, src_path, trg_path, sample_img_path, save_model_path, restore_model_path, gpu):
        self.batch_size = batch_size
        self.epoch_iter = epoch_iter
        self.lr = lr

        self.src_path = src_path
        self.trg_path = trg_path
        self.sample_img_path = sample_img_path
        self.save_model_path = save_model_path
        self.restore_model_path = restore_model_path

        self.D = D()
        self.G = G()

        self.gpu = gpu

        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])

    def load_dataset(self):
        src_data = dset.ImageFolder(self.src_path, self.transformations)
        self.src_loader = DataLoader(src_data, batch_size=self.batch_size)
        trg_data = dset.ImageFolder(self.trg_path, self.transformations)
        self.trg_loader = DataLoader(trg_data, batch_size=self.batch_size)

    def train(self):
        data_loader = get_loader(self.batch_size, self.src_path, self.trg_path, self.transform)
        print('Dataset Load Success!')

        D_adam = optim.Adam(self.D.parameters(), lr=self.lr)
        G_adam = optim.Adam(self.G.parameters(), lr=self.lr)

        if self.gpu:
            self.D = self.D.cuda()
            self.G = self.G.cuda()
            ones, zeros = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
            BCE_loss = nn.BCELoss().cuda()
        else:
            ones, zeros = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))
            BCE_loss = nn.BCELoss()

        self.D.train()
        self.G.train()
        print('Training Start')
        for epoch in range(self.epoch_iter):
            for step, (src, trg) in enumerate(data_loader):
                src_data = src
                trg_data = trg

                self.D.zero_grad()
                self.G.zero_grad()

                src_input = Variable(src_data.cuda())
                trg_input = Variable(trg_data.cuda())

                src_generated = self.G(src_input)

                D_src_generated = self.D(src_generated, trg_input)
                D_trg_input = self.D(trg_input, trg_input)

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
                    
                # save sample images 
                if step % 100 == 0:
                    vutils.save_image(src_input[0], os.paht.join(self.sample_img_path, f'epoch-{epoch}-step-{step}-src_input.jpg'), normalize = True)
                    vutils.save_image(trg_input[0], os.paht.join(self.sample_img_path, f'epoch-{epoch}-step-{step}-trg_input.jpg'), normalize = True)
                    vutils.save_image(src_generated.data[0], os.paht.join(self.sample_img_path, f'epoch-{epoch}-step-{step}-generated.jpg'), normalize = True)

            # save model
            if epoch % 100 == 0 and epoch != 0:
                torch.save(self.D.state_dict(), os.path.join(self.save_model_path, str(epoch) + 'D' + '.pth'))
                torch.save(self.G.state_dict(), os.path.join(self.save_model_path, str(epoch) + 'G' + '.pth'))
