import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class residual_block(nn.Module):
    def __init__(self, nf = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.BatchNorm2d(nf)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.layers(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_nc = 3, out_nc = 3, nf = 64, nb = 16):
        super(ResNet, self).__init__()
        
        resnet_blocks_1 = []
        resnet_blocks_2 = []
        resnet_blocks_3 = []
        
        # Block group 1
        resnet_blocks_1.append(nn.Conv2d(in_nc, nf, 3, 1, 1))
        resnet_blocks_1.append(nn.ReLU(True))
        self.resnet_network_1 = nn.Sequential(*resnet_blocks_1)
        
        # Block group 2
        resnet_blocks_2.append(nn.Conv2d(in_nc, nf, 3, 1, 1))
        resnet_blocks_2.append(nn.ReLU(True))
        for _ in range(nb):
            resnet_blocks_2.append(residual_block(nf))
        resnet_blocks_2.append(nn.Conv2d(nf, nf, 3, 1, 1))
        resnet_blocks_2.append(nn.BatchNorm2d(nf))
        self.resnet_network_2 = nn.Sequential(*resnet_blocks_2)
        
        # Block group 3
        resnet_blocks_3.append(nn.Conv2d(nf, 256, 3, 1, 1))
        resnet_blocks_3.append(nn.ReLU(True))
        resnet_blocks_3.append(nn.Conv2d(256, 256, 3, 1, 1))
        resnet_blocks_3.append(nn.ReLU(True)) 
        resnet_blocks_3.append(nn.Conv2d(256, out_nc, 3, 1, 1))  
        self.resnet_network_3 = nn.Sequential(*resnet_blocks_3)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output_1 = self.resnet_network_1(x)
        output_2 = self.resnet_network_2(x) 
 
        output_2 += output_1
        output_2 = self.relu(output_2)
          
        output_3 = self.resnet_network_3(output_2)
        return output_3
        
        
class GeneratorResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=32, nb=4):
        super(GeneratorResNet, self).__init__()
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

        self.resnet_blocks = ResNet(in_nc = nf * 4, out_nc = nf * 4)

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
        x = self.resnet_blocks(x)
        output = self.up_convs(x)
        return output        