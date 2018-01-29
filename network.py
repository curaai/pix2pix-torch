import torch
import torchvision
import torch.nn as nn 
import numpy as np


def conv_block(idx, name, in_c, out_c, activation, kernel_size=4, stride=2, padding=1, transpose=False, bn=True, bias=True, ):
    block = nn.Sequential()

    if not transpose:
        block.add_module(name + ' Conv2d' + str(idx), nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=bias))
    else:
        block.add_module(name + ' Conv2d_Transpose' + str(idx), nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding, bias=bias))
    if bn:
        block.add_module(name + ' Batch_norm' + str(idx), nn.BatchNorm2d(out_c))
    if activation == 'relu':
        block.add_module(name + ' ReLU' + str(idx), nn.ReLU(inplace=True))
    elif activation == 'leaky_relu':
        block.add_module(name + ' Leaky_ReLU' + str(idx), nn.LeakyReLU(0.2, inplace=True))
    elif activation == 'sigmoid':
        block.add_module(name + ' Sigmoid' + str(idx), nn.Sigmoid())
    elif activation == 'tanh':
        block.add_module(name + ' Tanh' + str(idx), nn.Tanh())
    
    return block

class Network(nn.Moudle):
    def __init__(self):
        self.build()


# input is 4 x 4 x in_c
class D(nn.Module):
    """
    input : ? x 4 x 4 x 6(3 x 2)
    layer0 : ? x 8 x 8 x 32 
    layer1 : ? x 16 x 16 x 64 
    layer2 : ? x 32 x 32 x 128 
    layer3 : ? x 31 x 31 x 256 
    output : ? x 30 x 30 x 1 
    """
    def __init__(self, in_c):
        self.name = "D"
        
        self.build()

    def build(self):
        activation = 'leaky_relu'

        layer0 = conv_block(0, self.name, 6, 32, activation, bn=False)
        layer1 = conv_block(1, self.name, 32, 64, activation)
        layer2 = conv_block(2, self.name, 64, 128, activation)
        layer3 = conv_block(3, self.name, 128, 256, activation, kernel_size=2, stride=1, padding=0) 
        layer4 = conv_block(4, self.name, 256, 1, 'sigmoid', kernel_size=2, stride=1, padding=0)

        self.layer0 = layer0
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

    def forward(self, src_input, trg_input):
        x = torch.cat((src_input, trg_input, 0)
        out0 = self.layer0(x)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4
        
    