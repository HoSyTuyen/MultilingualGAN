import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layers(x)


class VDCNN(nn.Module):
    def __init__(self, in_nc = 3, out_nc = 3, nf = 64, nb = 18):
        super(VDCNN, self).__init__()
        
        vdcnn_blocks = []
        vdcnn_blocks.append(conv_block(in_nc, nf, 3, 1, 1))
        
        for _ in range(nb):
            vdcnn_blocks.append(conv_block(nf, nf, 3, 1, 1))
            
        vdcnn_blocks.append(nn.Conv2d(nf, out_nc, 3, 1, 1))
       
        self.vdcnn_network = nn.Sequential(*vdcnn_blocks)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        identity = x.clone()
        x = self.vdcnn_network(x)
        x += identity
        x = self.relu(x)
        return x
        
        
class GeneratorVDCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4):
        super(GeneratorVDCNN, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3), 
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1), 
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1),
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), 
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.vdcnn_blocks = VDCNN(in_nc = nf * 4, out_nc = nf * 4)

        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1),
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3),
            nn.Tanh(),
        )

        utils.initialize_weights(self)

    def forward(self, input):
        x = self.down_convs(input)
        x = self.vdcnn_blocks(x)
        output = self.up_convs(x)
        return output        
